"""LLM-driven tools registered on top of the deterministic registry.

Kept in a separate module so ``tools.py`` has no dependency on a token
or a network. The orchestrator calls ``register_llm_tools(reg, client)``
once an ``LlmClient`` is constructed; if the client is unavailable, the
LLM tools degrade to no-ops with a clear rationale, and the orchestrator
falls back to deterministic behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.llm_client import LlmClient
from src.agent.tools import ToolRegistry, ToolResult, ToolSpec


# Doc types are constrained to a small enum so downstream code can switch on them.
DOC_TYPES = ("letter", "form", "article", "notes", "handwriting", "receipt", "other")

CLEANUP_SYS = (
    "You fix obvious OCR errors WITHOUT inventing content. Return STRICT JSON: "
    '{"cleaned_text": "<full corrected text preserving line breaks>", '
    '"changes_made": <int>, "notes": "<brief notes>"}. '
    "Rules: only fix character-level OCR mistakes (e.g. rn→m, 0↔O, l↔1). "
    "Do not paraphrase, summarise, translate, add headings, or remove content. "
    "Preserve every newline."
)

PLAN_SYS = (
    "You are a document-understanding agent. In ONE response you must classify "
    "the document AND propose its block structure for a Word export. "
    "Return STRICT JSON ONLY (no prose, no markdown fences): "
    '{"doc_type": "<letter|form|article|notes|handwriting|receipt|other>", '
    '"doc_type_confidence": <float 0-1>, '
    '"blocks": [{"type": "paragraph|heading|ulist|olist|code|table", '
    '"text": "<text or empty>", "level": <1-6 for headings only>, '
    '"items": ["..."], "table": [["..."]]}], '
    '"format_confidence": <float 0-1>}. '
    "Rules: preserve content verbatim from the OCR text; do not invent text; "
    "use the layout summary to bias choices (centered short lines → headings; "
    "bullet markers → ulist; numbered lines → olist; pipe rows → tables)."
)


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def make_tool_clean_text_llm(client: LlmClient) -> ToolSpec:
    def run(*, text: str, doc_type: str = "other") -> ToolResult:
        if not client.available:
            return ToolResult(
                name="tool_clean_text_llm",
                ok=False,
                data={"cleaned_text": text},  # passthrough so caller can keep working
                error=client.status_message,
                rationale="LLM unavailable; using raw OCR text.",
            )
        user = (
            f"Document type: {doc_type}\n"
            f"Raw OCR text:\n{_truncate(text, 6000)}\n\n"
            "Return cleaned text."
        )
        resp = client.chat_json(system=CLEANUP_SYS, user=user, max_new_tokens=1500)
        if not resp.ok or not isinstance(resp.parsed, dict):
            return ToolResult(
                name="tool_clean_text_llm",
                ok=False,
                data={"cleaned_text": text, "raw": resp.text},
                error=resp.error or "Unparseable cleanup output",
                rationale="LLM cleanup parse failed; using raw OCR text.",
            )
        cleaned = str(resp.parsed.get("cleaned_text", text))
        # Safety: if cleaned text is wildly shorter than raw, it probably summarised.
        # Reject and fall back.
        if len(cleaned) < 0.6 * len(text):
            return ToolResult(
                name="tool_clean_text_llm",
                ok=False,
                data={"cleaned_text": text, "rejected": cleaned},
                error="Cleanup output too short — possible summarisation.",
                rationale="Rejected LLM output (length drop > 40%); using raw OCR text.",
            )
        changes = int(resp.parsed.get("changes_made") or 0)
        return ToolResult(
            name="tool_clean_text_llm",
            ok=True,
            data={"cleaned_text": cleaned, "changes_made": changes},
            confidence=0.8 if changes > 0 else 0.6,
            rationale=f"LLM applied ~{changes} character-level corrections.",
        )

    return ToolSpec(
        name="tool_clean_text_llm",
        description="LLM cleans obvious OCR errors without paraphrasing. Falls back to raw text on any doubt.",
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "doc_type": {"type": "string"},
            },
            "required": ["text"],
        },
        run=run,
    )


def make_tool_plan_document(client: LlmClient) -> ToolSpec:
    """Single-shot classifier + formatter. Replaces two separate LLM
    round-trips with one — the dominant latency win on HF Inference."""

    def run(
        *,
        text: str,
        layout_summary: Optional[Dict[str, Any]] = None,
        prefs: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if not client.available:
            return ToolResult(
                name="tool_plan_document",
                ok=False,
                error=client.status_message,
                rationale="LLM unavailable; orchestrator falls back to defaults + regex parser.",
            )
        user = (
            f"Layout summary: {layout_summary or {}}\n"
            f"User prefs: {prefs or {}}\n"
            f"OCR text (preserve verbatim):\n{_truncate(text, 4000)}\n\n"
            "Return the combined classification + block structure now."
        )
        # 1024 is enough for typical single-page documents; we shrink
        # input above to keep total round-trip well under the HF timeout.
        resp = client.chat_json(system=PLAN_SYS, user=user, max_new_tokens=1024)
        if not resp.ok or not isinstance(resp.parsed, dict):
            return ToolResult(
                name="tool_plan_document",
                ok=False,
                data={"raw": resp.text},
                error=resp.error or "Unparseable plan output",
                rationale="LLM did not return parseable JSON; falling back.",
            )
        doc_type = str(resp.parsed.get("doc_type", "other")).lower()
        if doc_type not in DOC_TYPES:
            doc_type = "other"
        doc_conf = float(resp.parsed.get("doc_type_confidence", 0.5) or 0.5)
        blocks = resp.parsed.get("blocks") or []
        fmt_conf = float(resp.parsed.get("format_confidence", 0.5) or 0.5)
        if not isinstance(blocks, list):
            blocks = []
        return ToolResult(
            name="tool_plan_document",
            ok=True,
            data={
                "doc_type": doc_type,
                "doc_type_confidence": doc_conf,
                "blocks": blocks,
                "format_confidence": fmt_conf,
            },
            confidence=min(doc_conf, fmt_conf),
            rationale=(
                f"Classified as '{doc_type}' (conf {doc_conf:.2f}); "
                f"proposed {len(blocks)} blocks (conf {fmt_conf:.2f})."
            ),
        )

    return ToolSpec(
        name="tool_plan_document",
        description="Single-shot LLM call: returns doc_type + suggested blocks together.",
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "layout_summary": {"type": "object"},
                "prefs": {"type": "object"},
            },
            "required": ["text"],
        },
        run=run,
    )


def register_llm_tools(registry: ToolRegistry, client: LlmClient) -> None:
    """Add (or replace) the LLM-backed tools on the registry. Safe to
    call even if ``client`` is unavailable — the tools will simply
    return ``ok=False`` at call time."""
    registry.register(make_tool_plan_document(client))   # one-shot classify + format
    registry.register(make_tool_clean_text_llm(client))  # opt-in OCR cleanup pass
