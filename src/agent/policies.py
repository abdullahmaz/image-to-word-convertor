"""Centralised constants, thresholds, and PII patterns for the agent.

Phase-1 had thresholds scattered across ``src/layout.py``. The agent must
be able to introspect and (later) adapt these, so we collect them here.
PII patterns reflect Pakistan's PECA-2016 / GDPR awareness: identifiers
that, if leaked, would constitute personal data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass(frozen=True)
class Policies:
    """Numerical knobs the agent reads from. Frozen — agent proposes
    changes via memory updates, not by mutating policy in place."""

    # ---- Layout heuristics (lifted from layout.py) ----
    bold_ink_density_threshold: float = 0.18
    italic_min_angle_deg: float = 10.0
    italic_max_angle_deg: float = 25.0
    align_center_margin_diff: float = 0.08
    align_center_min_margin: float = 0.10
    align_right_max_right_margin: float = 0.06
    align_right_min_left_margin: float = 0.18
    contour_min_width_ratio: float = 0.08
    contour_min_height_px: int = 8
    morph_kernel_min_width: int = 20
    morph_kernel_width_divisor: int = 40
    morph_kernel_height: int = 3
    line_merge_height_ratio: float = 0.35
    line_merge_min_height_px: int = 10
    line_merge_max_gap_px: int = 20

    # ---- Image quality assessment ----
    quality_low_resolution_px: int = 600         # short side below this = low res
    quality_low_contrast_std: float = 35.0       # grayscale std below this = low contrast
    quality_low_sharpness_var: float = 100.0     # Laplacian variance below this = blurry
    quality_high_skew_deg: float = 5.0           # |skew| above this = needs deskew warning

    # ---- Confidence floors that drive HITL ----
    ocr_confidence_floor: float = 0.55           # below → trigger post-OCR review
    classification_confidence_floor: float = 0.6 # below → ask user for doc type
    formatting_confidence_floor: float = 0.5     # below → preview block-by-block

    # ---- LLM behaviour ----
    # Qwen2.5-1.5B is small, fast, AND much more reliable at strict JSON
    # output than 1B Llama. Override via AGENT_LLM_MODEL env var.
    llm_default_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.2
    llm_request_timeout_s: float = 45.0

    # ---- OCR pre-processing ----
    # LightOnOCR-1B on CPU scales ~linearly with pixel count. Most A4
    # scans don't need >1600px on the long side for readable OCR.
    ocr_max_long_side_px: int = 1600
    ocr_default_max_new_tokens: int = 768


# PII patterns are NOT in the dataclass because compiled regexes are
# unhashable; they're module-level singletons.

PII_PATTERNS: Dict[str, Pattern[str]] = {
    # Pakistan CNIC: 5-7-1 with dashes
    "cnic": re.compile(r"\b\d{5}-\d{7}-\d\b"),
    # Pakistan mobile / landline (E.164-ish, allowing common local forms)
    "phone_pk": re.compile(r"\b(?:\+92|0092|0)\s*3\d{2}[\s-]?\d{7}\b"),
    "phone_generic": re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban": re.compile(r"\bPK\d{2}[A-Z0-9]{4}\d{16}\b"),
}


def scan_pii(text: str) -> List[Tuple[str, str]]:
    """Returns list of (pattern_name, matched_text) tuples found in text.

    Used by the perception step to flag whether HITL acknowledgement is
    required before further processing.
    """
    findings: List[Tuple[str, str]] = []
    if not text:
        return findings
    for name, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            findings.append((name, m.group(0)))
    return findings


# ---- Autonomy levels ----
# manual: every step requires user click
# assisted: agent runs, but pauses at HITL checkpoints (default)
# autonomous: agent runs end-to-end unless PII or hard-fail detected
AUTONOMY_LEVELS: Tuple[str, ...] = ("manual", "assisted", "autonomous")
DEFAULT_AUTONOMY: str = "assisted"

# ---- HITL checkpoint kinds (used by orchestrator + UI) ----
HITL_KINDS: Tuple[str, ...] = (
    "pre_ocr",
    "post_ocr",
    "post_ocr_pii",
    "post_classify",
    "pre_build",
    "post_export",
)

# ---- Agent loop stages, in order ----
STAGE_ORDER: Tuple[str, ...] = (
    "init",
    "observe",
    "interpret",
    "decide",
    "act",
    "reflect",
    "learn",
)
