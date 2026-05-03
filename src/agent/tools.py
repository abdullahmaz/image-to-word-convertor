"""Tool registry exposed to the agent.

Tools are the agent's *actions*. Each one has a stable name, a JSON
schema describing its arguments, and a ``run`` callable. The orchestrator
chooses tools based on the LLM's plan (or a deterministic fallback) and
records every invocation in the run log.

The wrappers here intentionally avoid duplicating logic — they delegate
to the existing Phase-1 functions in ``src.ocr_lighton``, ``src.layout``,
and ``src.docx_writer``. New capabilities (image assessment, LLM-driven
cleanup/classification/formatting) are implemented in this module
because they're agent-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.agent.policies import Policies, scan_pii


@dataclass
class ToolResult:
    """Uniform return type for every tool. Agents read ``ok`` first; on
    failure ``error`` is set and ``data`` may still hold partial output."""

    name: str
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None  # short string for the run log


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    run: Callable[..., ToolResult]


class ToolRegistry:
    """Dictionary-backed registry. Allows the orchestrator (and the LLM
    planner) to enumerate available tools and call them by name."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def manifest(self) -> List[Dict[str, Any]]:
        """Compact JSON-serialisable manifest used in LLM planning prompts."""
        return [
            {"name": s.name, "description": s.description, "schema": s.schema}
            for s in self._tools.values()
        ]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self.get(name)
        try:
            return spec.run(**kwargs)
        except Exception as exc:  # tools must never crash the agent loop
            return ToolResult(
                name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                rationale="Tool raised an exception; agent should fall back or stop.",
            )


# ---------------------------------------------------------------------------
# Deterministic, non-LLM tools
# ---------------------------------------------------------------------------


def _tool_assess_image_impl(*, image: Image.Image, policies: Optional[Policies] = None) -> ToolResult:
    """Perception step. Inspects the uploaded image without any model
    inference and returns quality / risk metrics. Cheap so we always run
    it first."""
    p = policies or Policies()

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    short_side = min(h, w)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    contrast_std = float(np.std(gray))
    sharpness_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Skew estimate via Hough on edges (best-effort, fails silently).
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        hough = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        skew_deg = 0.0
        if hough is not None:
            angles = []
            for line in hough[:50]:
                rho, theta = line[0]
                deg = (theta * 180 / math.pi) - 90
                if -45 <= deg <= 45:
                    angles.append(deg)
            if angles:
                skew_deg = float(np.median(angles))
    except cv2.error:
        skew_deg = 0.0

    # Dominant-script heuristic: ratio of non-ASCII glyph likelihood.
    # We can't read text yet, so we just flag the image as "ambiguous"
    # and let the OCR step refine.
    notes: List[str] = []
    if short_side < p.quality_low_resolution_px:
        notes.append(f"low_resolution(short_side={short_side}px)")
    if contrast_std < p.quality_low_contrast_std:
        notes.append(f"low_contrast(std={contrast_std:.1f})")
    if sharpness_var < p.quality_low_sharpness_var:
        notes.append(f"blurry(laplacian_var={sharpness_var:.1f})")
    if abs(skew_deg) > p.quality_high_skew_deg:
        notes.append(f"skewed({skew_deg:+.1f}deg)")

    quality_ok = len(notes) == 0
    rationale = (
        "Image looks clean — proceeding without preprocessing warnings."
        if quality_ok
        else "Image flagged for: " + ", ".join(notes) + ". Agent will warn the user."
    )

    return ToolResult(
        name="tool_assess_image",
        ok=True,
        data={
            "width": w,
            "height": h,
            "short_side_px": short_side,
            "contrast_std": contrast_std,
            "sharpness_var": sharpness_var,
            "skew_deg": skew_deg,
            "quality_notes": notes,
            "quality_ok": quality_ok,
        },
        confidence=1.0 if quality_ok else 0.5,
        rationale=rationale,
    )


def _tool_extract_text_impl(
    *,
    engine: Any,  # avoid a hard import-time dep on torch in this module
    image: Image.Image,
    max_new_tokens: int = 1024,
    max_long_side_px: Optional[int] = None,
) -> ToolResult:
    """Wraps ``OcrLighton.ocr``. ``engine`` must be an already-loaded
    ``LightOnOcrEngine`` (cached in the Streamlit app).

    Auto-downscales the image when its long side exceeds
    ``max_long_side_px``. On CPU this is the dominant latency saver —
    OCR runtime scales roughly with pixel count.
    """
    original_size = image.size
    resized = image
    if max_long_side_px and max(image.size) > max_long_side_px:
        long_side = max(image.size)
        scale = max_long_side_px / float(long_side)
        new_size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
    text = engine.ocr(image=resized, max_new_tokens=max_new_tokens)

    # Confidence proxy: longer, multi-line outputs with reasonable
    # alphabetic ratio are more trustworthy than 1-token noise. This is
    # a coarse signal, not a logprob — clearly labelled as a heuristic.
    if not text:
        confidence = 0.0
    else:
        alpha = sum(1 for c in text if c.isalpha())
        ratio = alpha / max(1, len(text))
        length_score = min(1.0, len(text) / 200.0)
        confidence = round(0.5 * ratio + 0.5 * length_score, 3)

    resize_note = (
        f" (downscaled {original_size[0]}x{original_size[1]} → {resized.size[0]}x{resized.size[1]})"
        if resized.size != original_size
        else ""
    )
    return ToolResult(
        name="tool_extract_text",
        ok=True,
        data={
            "text": text,
            "char_count": len(text),
            "line_count": text.count("\n") + 1 if text else 0,
            "ocr_input_size": list(resized.size),
            "ocr_input_resized": resized.size != original_size,
        },
        confidence=confidence,
        rationale=f"OCR produced {len(text)} chars; heuristic confidence {confidence:.2f}{resize_note}.",
    )


def _tool_analyze_layout_impl(*, image: Image.Image, policies: Optional[Policies] = None) -> ToolResult:
    from src.layout import analyze_layout  # local import to keep module light

    layout = analyze_layout(image, policies=policies)
    return ToolResult(
        name="tool_analyze_layout",
        ok=True,
        data={
            "layout": layout,
            "line_count": len(layout.lines),
            "alignment_summary": _summarise_alignments(layout),
        },
        confidence=1.0 if layout.lines else 0.2,
        rationale=f"Detected {len(layout.lines)} text-line boxes via OpenCV heuristics.",
    )


def _summarise_alignments(layout: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {"left": 0, "center": 0, "right": 0}
    for ln in layout.lines:
        counts[ln.style.alignment] = counts.get(ln.style.alignment, 0) + 1
    return counts


def _tool_scan_pii_impl(*, text: str) -> ToolResult:
    findings = scan_pii(text)
    has_pii = bool(findings)
    return ToolResult(
        name="tool_scan_pii",
        ok=True,
        data={
            "has_pii": has_pii,
            "findings": [{"kind": k, "match": _redact(v)} for k, v in findings],
            "kinds": sorted({k for k, _ in findings}),
        },
        confidence=1.0,
        rationale=(
            "No PII patterns matched in extracted text."
            if not has_pii
            else f"Matched PII patterns: {sorted({k for k, _ in findings})}. HITL acknowledgement required."
        ),
    )


def _redact(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _tool_build_docx_impl(
    *,
    ocr_text: str,
    layout: Any,
    user_prefs: Optional[Any] = None,
    title: str = "",
    structured_blocks: Optional[List[Dict[str, Any]]] = None,
) -> ToolResult:
    """Wraps ``build_docx``. When ``structured_blocks`` is provided (LLM
    formatting suggestion), it overrides the regex parser; otherwise the
    Phase-1 markdown parser is used."""
    from src.docx_writer import build_docx

    docx_bytes = build_docx(
        title=title,
        ocr_text=ocr_text,
        layout=layout,
        user_prefs=user_prefs,
        structured_blocks=structured_blocks,
    )
    return ToolResult(
        name="tool_build_docx",
        ok=True,
        data={"docx_bytes": docx_bytes, "byte_count": len(docx_bytes)},
        confidence=1.0,
        rationale=(
            "Built docx from agent-suggested structure."
            if structured_blocks
            else "Built docx from regex-parsed markdown blocks (Phase-1 path)."
        ),
    )


def default_registry() -> ToolRegistry:
    """Registry containing the deterministic tools. The LLM-backed tools
    are added in ``src.agent.llm_tools.register_llm_tools`` once the
    ``LlmClient`` is constructed (so we don't require a token to import
    this module)."""
    reg = ToolRegistry()

    reg.register(ToolSpec(
        name="tool_assess_image",
        description="Inspect the uploaded image and return quality + skew metrics. Cheap; always run first.",
        schema={"type": "object", "properties": {"image": {"type": "PIL.Image"}}, "required": ["image"]},
        run=_tool_assess_image_impl,
    ))
    reg.register(ToolSpec(
        name="tool_extract_text",
        description="Run LightOnOCR-1B on the image to extract text. Returns text + heuristic confidence.",
        schema={
            "type": "object",
            "properties": {
                "engine": {"type": "LightOnOcrEngine"},
                "image": {"type": "PIL.Image"},
                "max_new_tokens": {"type": "integer", "default": 1024},
            },
            "required": ["engine", "image"],
        },
        run=_tool_extract_text_impl,
    ))
    reg.register(ToolSpec(
        name="tool_analyze_layout",
        description="Run OpenCV layout analysis (line boxes, alignment, bold/italic heuristics).",
        schema={"type": "object", "properties": {"image": {"type": "PIL.Image"}}, "required": ["image"]},
        run=_tool_analyze_layout_impl,
    ))
    reg.register(ToolSpec(
        name="tool_scan_pii",
        description="Scan extracted text for personally-identifiable patterns (CNIC, phone, email, IBAN, card).",
        schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        run=_tool_scan_pii_impl,
    ))
    reg.register(ToolSpec(
        name="tool_build_docx",
        description="Render the final .docx using OCR text, layout, optional LLM-suggested structure, and user prefs.",
        schema={
            "type": "object",
            "properties": {
                "ocr_text": {"type": "string"},
                "layout": {"type": "LayoutAnalysis"},
                "user_prefs": {"type": "UserPrefs"},
                "structured_blocks": {"type": "array", "items": {"type": "object"}},
                "title": {"type": "string"},
            },
            "required": ["ocr_text", "layout"],
        },
        run=_tool_build_docx_impl,
    ))
    return reg
