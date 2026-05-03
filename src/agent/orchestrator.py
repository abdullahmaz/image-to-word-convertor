"""Agent orchestrator: Observe → Interpret → Decide → Act → Reflect → Learn.

This is the single entry point used by ``app.py``. It coordinates the
tool registry, memory, LLM client, and explainability log.

The loop is intentionally split into staged methods so the Streamlit UI
can pause between stages and surface HITL panels (e.g. user edits OCR
text before the formatting step continues).

Stages:
  observe()    – PIL image in → image-quality metrics
  interpret()  – run OCR + layout + PII scan + (LLM) classification
  decide()     – pick formatting strategy based on doc-type + prefs +
                 confidence; may consult LLM ``tool_suggest_formatting``
  act()        – render the .docx via ``tool_build_docx``
  reflect()    – record outcome metrics; persist log
  learn()      – record user feedback / overrides into long-term memory

A typical "autonomous" run executes all stages back-to-back. In
"assisted" mode the UI surfaces HITL prompts whenever
``RunResult.hitl_required`` is set after a stage.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from src.agent.explainability import RunLog
from src.agent.llm_client import LlmClient
from src.agent.llm_tools import register_llm_tools
from src.agent.memory import LongTermMemory, ShortTermMemory, UserPrefs
from src.agent.policies import Policies, scan_pii
from src.agent.tools import ToolRegistry, ToolResult, default_registry


AutonomyLevel = Literal["manual", "assisted", "autonomous"]


@dataclass
class RunResult:
    """Snapshot of agent state. Returned after every stage; the UI reads
    ``stage_completed`` and ``hitl_required`` to decide what to render
    next."""

    stage_completed: str = "init"
    hitl_required: bool = False
    hitl_kind: Optional[str] = None        # pre_ocr | post_ocr | post_classify | pre_build | post_export
    hitl_message: Optional[str] = None
    image_assessment: Optional[Dict[str, Any]] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    layout: Any = None
    layout_summary: Optional[Dict[str, Any]] = None
    pii_findings: List[Dict[str, Any]] = field(default_factory=list)
    pii_acknowledged: bool = False
    doc_type: Optional[str] = None
    doc_type_confidence: Optional[float] = None
    cleaned_text: Optional[str] = None
    structured_blocks: Optional[List[Dict[str, Any]]] = None
    docx_bytes: Optional[bytes] = None
    error: Optional[str] = None


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class AgentOrchestrator:
    """Stateful orchestrator. One instance per Streamlit session is
    enough; the Streamlit cache keeps it alive across reruns."""

    def __init__(
        self,
        *,
        ocr_engine: Any,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[LlmClient] = None,
        long_term: Optional[LongTermMemory] = None,
        policies: Optional[Policies] = None,
        autonomy: AutonomyLevel = "assisted",
    ) -> None:
        self.policies = policies or Policies()
        self.ocr_engine = ocr_engine
        self.registry = registry or default_registry()
        self.llm = llm_client or LlmClient(policies=self.policies)
        register_llm_tools(self.registry, self.llm)
        self.long_term = long_term or LongTermMemory()
        self.autonomy: AutonomyLevel = autonomy
        self.short_term = ShortTermMemory()
        self.log = RunLog()
        self.result = RunResult()

    # ------------------------------------------------------------------
    # Stage 1 — Observe: cheap perception, no model inference
    # ------------------------------------------------------------------
    def observe(self, image: Image.Image) -> RunResult:
        t0 = time.perf_counter()
        res = self.registry.call("tool_assess_image", image=image, policies=self.policies)
        self.log.log(
            "observe", res.name,
            rationale=res.rationale or "image assessed",
            inputs={"image_size": image.size},
            outputs={k: v for k, v in res.data.items() if k != "layout"},
            confidence=res.confidence,
            elapsed_ms=_ms_since(t0),
        )
        self.short_term.set("image_assessment", res.data)
        self.result.image_assessment = res.data
        self.result.stage_completed = "observe"

        # HITL gate: very low quality and not autonomous → ask user.
        if not res.data.get("quality_ok", True) and self.autonomy != "autonomous":
            self.result.hitl_required = True
            self.result.hitl_kind = "pre_ocr"
            self.result.hitl_message = (
                "Image quality issues detected: "
                + ", ".join(res.data.get("quality_notes") or [])
                + ". OCR accuracy may suffer. Continue?"
            )
        return self.result

    # ------------------------------------------------------------------
    # Stage 2 — Interpret: OCR + layout + PII + classification
    # ------------------------------------------------------------------
    def interpret(self, image: Image.Image, *, max_new_tokens: int = 1024) -> RunResult:
        # OCR — cache by content hash so the user can iterate on settings
        # (autonomy slider, doc-type override, prefs) without re-running
        # the slow vision model.
        img_hash = hashlib.sha1(image.tobytes()).hexdigest()
        cache_key = f"ocr_cache::{img_hash}::{max_new_tokens}::{self.policies.ocr_max_long_side_px}"
        cached = self.short_term.get(cache_key)
        t0 = time.perf_counter()
        if cached is not None:
            ocr_res = cached
            self.log.log(
                "interpret", "tool_extract_text",
                rationale="OCR result reused from in-session cache (no model call).",
                outputs={"char_count": ocr_res.data.get("char_count"), "line_count": ocr_res.data.get("line_count")},
                confidence=ocr_res.confidence,
                elapsed_ms=_ms_since(t0),
            )
        else:
            ocr_res = self.registry.call(
                "tool_extract_text",
                engine=self.ocr_engine,
                image=image,
                max_new_tokens=max_new_tokens,
                max_long_side_px=self.policies.ocr_max_long_side_px,
            )
            self.short_term.set(cache_key, ocr_res)
            self.log.log(
                "interpret", ocr_res.name,
                rationale=ocr_res.rationale or "ocr ran",
                outputs={
                    "char_count": ocr_res.data.get("char_count"),
                    "line_count": ocr_res.data.get("line_count"),
                    "ocr_input_size": ocr_res.data.get("ocr_input_size"),
                },
                confidence=ocr_res.confidence,
                elapsed_ms=_ms_since(t0),
            )
        if not ocr_res.ok:
            self.result.error = ocr_res.error
            return self.result
        ocr_text = ocr_res.data.get("text", "")
        self.short_term.set("ocr_text", ocr_text)
        self.result.ocr_text = ocr_text
        self.result.ocr_confidence = ocr_res.confidence

        # Layout
        t1 = time.perf_counter()
        layout_res = self.registry.call("tool_analyze_layout", image=image, policies=self.policies)
        self.log.log(
            "interpret", layout_res.name,
            rationale=layout_res.rationale or "layout analysed",
            outputs={"line_count": layout_res.data.get("line_count"), "alignment_summary": layout_res.data.get("alignment_summary")},
            confidence=layout_res.confidence,
            elapsed_ms=_ms_since(t1),
        )
        self.short_term.set("layout", layout_res.data.get("layout"))
        self.short_term.set("layout_summary", layout_res.data.get("alignment_summary"))
        self.result.layout = layout_res.data.get("layout")
        self.result.layout_summary = {
            "line_count": layout_res.data.get("line_count"),
            "alignment_summary": layout_res.data.get("alignment_summary"),
        }

        # PII scan
        t2 = time.perf_counter()
        pii_res = self.registry.call("tool_scan_pii", text=ocr_text)
        self.log.log(
            "interpret", pii_res.name,
            rationale=pii_res.rationale or "pii scanned",
            outputs={"has_pii": pii_res.data.get("has_pii"), "kinds": pii_res.data.get("kinds")},
            elapsed_ms=_ms_since(t2),
        )
        self.result.pii_findings = pii_res.data.get("findings") or []

        # Combined classification + formatting plan (one LLM call, not two).
        # We run it here so the user's HITL screens already see the proposed
        # doc_type and block count without waiting for stage 3.
        t3 = time.perf_counter()
        prefs_for_llm = {
            "alignment_bias": self.long_term.prefs.preferred_alignment_bias,
        }
        plan_res = self.registry.call(
            "tool_plan_document",
            text=ocr_text,
            layout_summary=self.result.layout_summary,
            prefs=prefs_for_llm,
        )
        plan_outputs: Dict[str, Any] = {
            "doc_type": plan_res.data.get("doc_type"),
            "block_count": len(plan_res.data.get("blocks") or []),
        }
        if not plan_res.ok and plan_res.data.get("raw"):
            # Truncated by the log sanitiser, but enough to diagnose
            # malformed JSON, refusals, or empty completions.
            plan_outputs["raw_llm_reply"] = plan_res.data.get("raw")
            plan_outputs["error"] = plan_res.error
        self.log.log(
            "interpret", plan_res.name,
            rationale=plan_res.rationale or "plan decided",
            outputs=plan_outputs,
            confidence=plan_res.confidence,
            elapsed_ms=_ms_since(t3),
        )
        if plan_res.ok:
            self.result.doc_type = plan_res.data.get("doc_type")
            self.result.doc_type_confidence = plan_res.data.get("doc_type_confidence")
            self.short_term.set("plan_blocks", plan_res.data.get("blocks"))
            self.short_term.set("plan_format_confidence", plan_res.data.get("format_confidence"))
        else:
            self.result.doc_type = "other"
            self.result.doc_type_confidence = 0.0
            self.short_term.set("plan_blocks", None)

        self.result.stage_completed = "interpret"

        # HITL gates: PII or low confidence (when not fully autonomous).
        if self.result.pii_findings and self.autonomy != "autonomous":
            self.result.hitl_required = True
            self.result.hitl_kind = "post_ocr_pii"
            self.result.hitl_message = (
                "Possible personal data detected ("
                + ", ".join(sorted({f["kind"] for f in self.result.pii_findings}))
                + "). Acknowledge before continuing — under PECA-2016 / GDPR you are responsible "
                "for ensuring you have consent to process this content."
            )
        elif (
            self.result.ocr_confidence is not None
            and self.result.ocr_confidence < self.policies.ocr_confidence_floor
            and self.autonomy == "manual"
        ):
            # Only block in manual mode; in assisted we surface the editor non-blocking.
            self.result.hitl_required = True
            self.result.hitl_kind = "post_ocr"
            self.result.hitl_message = (
                f"OCR confidence {self.result.ocr_confidence:.2f} is below threshold "
                f"{self.policies.ocr_confidence_floor:.2f}. Review the extracted text."
            )
        return self.result

    # ------------------------------------------------------------------
    # Stage 3 — Decide: pick a formatting strategy
    # ------------------------------------------------------------------
    def decide(self) -> RunResult:
        """Decide is now a no-LLM step: it consumes the combined plan
        produced in ``interpret`` and only triggers an extra LLM call if
        the user has explicitly enabled OCR cleanup (slow, opt-in)."""
        ocr_text = self.short_term.get("ocr_text") or ""
        doc_type = self.result.doc_type or "other"

        # Cleanup is OPT-IN (`enable_llm_cleanup` pref). It's the heaviest
        # call (long output) and most documents don't need it.
        wants_cleanup = bool(getattr(self.long_term.prefs, "enable_llm_cleanup", False))
        if wants_cleanup:
            t0 = time.perf_counter()
            clean_res = self.registry.call(
                "tool_clean_text_llm",
                text=ocr_text,
                doc_type=doc_type,
            )
            self.log.log(
                "decide", clean_res.name,
                rationale=clean_res.rationale or "cleanup attempted",
                outputs={"changes_made": clean_res.data.get("changes_made")},
                confidence=clean_res.confidence,
                elapsed_ms=_ms_since(t0),
            )
            cleaned = clean_res.data.get("cleaned_text", ocr_text)
            self.result.cleaned_text = cleaned
            self.short_term.set("text_for_build", cleaned)
        else:
            self.log.log(
                "decide", "skip_cleanup",
                rationale="LLM cleanup disabled (default fast path); using raw OCR text.",
            )
            self.result.cleaned_text = ocr_text
            self.short_term.set("text_for_build", ocr_text)

        # Block structure: reuse the plan from interpret() — no second LLM call.
        plan_blocks = self.short_term.get("plan_blocks")
        plan_conf = self.short_term.get("plan_format_confidence") or 0.0
        if plan_blocks and plan_conf >= self.policies.formatting_confidence_floor:
            self.result.structured_blocks = plan_blocks
            self.log.log(
                "decide", "use_plan_blocks",
                rationale=f"Using LLM plan blocks ({len(plan_blocks)}) at confidence {plan_conf:.2f}.",
                confidence=plan_conf,
            )
        else:
            self.result.structured_blocks = None
            if plan_blocks:
                why = f"plan confidence {plan_conf:.2f} below floor {self.policies.formatting_confidence_floor:.2f}"
            elif not self.llm.available:
                why = "LLM unavailable (no HF_TOKEN)"
            else:
                why = "LLM call failed or returned unparseable JSON"
            self.log.log(
                "decide", "fallback_regex_parser",
                rationale=f"Falling back to regex markdown parser ({why}).",
            )
        self.result.stage_completed = "decide"
        return self.result

    # ------------------------------------------------------------------
    # Stage 4 — Act: build the .docx
    # ------------------------------------------------------------------
    def act(self, *, title: str = "") -> RunResult:
        t0 = time.perf_counter()
        text_for_build = self.short_term.get("text_for_build") or self.result.ocr_text or ""
        layout = self.short_term.get("layout") or self.result.layout
        build_res = self.registry.call(
            "tool_build_docx",
            ocr_text=text_for_build,
            layout=layout,
            user_prefs=self.long_term.prefs,
            title=title,
            structured_blocks=self.result.structured_blocks,
        )
        self.log.log(
            "act", build_res.name,
            rationale=build_res.rationale or "docx built",
            outputs={"byte_count": build_res.data.get("byte_count")},
            confidence=build_res.confidence,
            elapsed_ms=_ms_since(t0),
        )
        if not build_res.ok:
            self.result.error = build_res.error
            return self.result
        self.result.docx_bytes = build_res.data.get("docx_bytes")
        self.result.stage_completed = "act"
        return self.result

    # ------------------------------------------------------------------
    # Stage 5 — Reflect: persist the run log
    # ------------------------------------------------------------------
    def reflect(self) -> RunResult:
        self.log.log(
            "reflect", "summary",
            rationale=(
                f"Completed run for doc_type={self.result.doc_type}, "
                f"ocr_conf={self.result.ocr_confidence}, "
                f"used_llm_blocks={self.result.structured_blocks is not None}, "
                f"pii={bool(self.result.pii_findings)}."
            ),
            outputs={"docx_bytes": (self.result.docx_bytes is not None) and len(self.result.docx_bytes)},
        )
        self.log.persist()
        self.result.stage_completed = "reflect"
        return self.result

    # ------------------------------------------------------------------
    # Stage 6 — Learn: write feedback into long-term memory
    # ------------------------------------------------------------------
    def learn(
        self,
        *,
        accepted: Optional[bool] = None,
        edited_text: Optional[str] = None,
        overridden_doc_type: Optional[str] = None,
    ) -> RunResult:
        with self.long_term.deferred_save():
            if accepted is not None:
                self.long_term.record_feedback("overall_run", accepted)
                self.log.log(
                    "learn", "record_feedback",
                    rationale=f"User marked run as {'accepted' if accepted else 'rejected'}.",
                    outputs={"acceptance_rate_overall": self.long_term.acceptance_rate("overall_run")},
                )
            if overridden_doc_type and overridden_doc_type != self.result.doc_type:
                self.long_term.record_feedback("doc_type_classifier", False)
                self.long_term.record_doc_type_override(
                    overridden_doc_type, "last_seen", time.time()
                )
                self.log.log(
                    "learn", "record_doc_type_override",
                    rationale=f"User changed doc_type from '{self.result.doc_type}' to '{overridden_doc_type}'.",
                )
                self.result.doc_type = overridden_doc_type
            if edited_text and edited_text != (self.result.cleaned_text or self.result.ocr_text):
                self.long_term.record_feedback("ocr_cleanup", False)
                self.log.log(
                    "learn", "record_text_edit",
                    rationale="User edited OCR/cleaned text — counted as cleanup miss.",
                )
                self.result.cleaned_text = edited_text
                self.short_term.set("text_for_build", edited_text)
        self.log.persist()
        self.result.stage_completed = "learn"
        return self.result

    # ------------------------------------------------------------------
    # Convenience: run all stages back-to-back (autonomous mode)
    # ------------------------------------------------------------------
    def run_all(self, image: Image.Image, *, title: str = "", max_new_tokens: int = 1024) -> RunResult:
        self.observe(image)
        if self.result.hitl_required and self.autonomy != "autonomous":
            return self.result
        self.interpret(image, max_new_tokens=max_new_tokens)
        if self.result.hitl_required and self.autonomy != "autonomous":
            return self.result
        self.decide()
        self.act(title=title)
        self.reflect()
        return self.result
