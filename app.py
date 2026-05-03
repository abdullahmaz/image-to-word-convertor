from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up HF_TOKEN from .env if present
except ImportError:
    pass

# Streamlit secrets (HF Spaces / Streamlit Cloud) → environment.
# Wrapped broadly because st.secrets raises different exception types
# across versions when no secrets file exists.
try:
    for _key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "AGENT_LLM_MODEL"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from src.agent.llm_client import LlmClient
from src.agent.llm_tools import DOC_TYPES
from src.agent.memory import LongTermMemory
from src.agent.orchestrator import AgentOrchestrator, AutonomyLevel
from src.agent.policies import (
    AUTONOMY_LEVELS,
    DEFAULT_AUTONOMY,
    HITL_KINDS,
    Policies,
    STAGE_ORDER,
)
from src.docx_writer import build_docx
from src.layout import analyze_layout, render_layout_debug
from src.ocr_lighton import LightOnOcrEngine


@dataclass
class AppConfig:
    model_id: str = "lightonai/LightOnOCR-1B-1025"
    max_new_tokens: int = 768
    ocr_max_long_side_px: int = 1600


# ---------------------------------------------------------------------------
# Cached / session-scoped state
# ---------------------------------------------------------------------------


def _load_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


@st.cache_resource(show_spinner=False)
def _get_engine(model_id: str) -> LightOnOcrEngine:
    return LightOnOcrEngine(model_id=model_id)


@st.cache_resource(show_spinner=False)
def _get_long_term_memory() -> LongTermMemory:
    return LongTermMemory()


def _get_orchestrator(
    engine: LightOnOcrEngine,
    autonomy: AutonomyLevel,
    ocr_max_long_side_px: int,
) -> AgentOrchestrator:
    """One orchestrator per (autonomy, OCR-resize) combination — rebuilt
    when the user changes either, so the cached OCR result invalidates
    and the new policy takes effect."""
    key = f"_orch_{autonomy}_{ocr_max_long_side_px}"
    if key not in st.session_state:
        from dataclasses import replace
        policies = replace(Policies(), ocr_max_long_side_px=int(ocr_max_long_side_px))
        st.session_state[key] = AgentOrchestrator(
            ocr_engine=engine,
            long_term=_get_long_term_memory(),
            policies=policies,
            llm_client=LlmClient(policies=policies),
            autonomy=autonomy,
        )
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar(cfg: AppConfig) -> dict:
    st.sidebar.title("Agent controls")

    agent_enabled = st.sidebar.toggle(
        "Enable agent",
        value=True,
        help="Off = deterministic Phase-1 pipeline (no perception, no LLM, no memory).",
    )

    autonomy: AutonomyLevel = st.sidebar.select_slider(  # type: ignore[assignment]
        "Autonomy level",
        options=list(AUTONOMY_LEVELS),
        value=DEFAULT_AUTONOMY,
        help=(
            "manual: every step needs a click. "
            "assisted: agent runs but pauses at HITL checkpoints. "
            "autonomous: end-to-end, only stops on PII or hard errors."
        ),
    )

    cfg.max_new_tokens = int(
        st.sidebar.slider("Max OCR tokens", 256, 4096, cfg.max_new_tokens, step=128)
    )
    cfg.ocr_max_long_side_px = int(
        st.sidebar.slider(
            "OCR input size (long-side px)",
            min_value=800, max_value=3200, value=cfg.ocr_max_long_side_px, step=200,
            help=(
                "Images larger than this are downscaled before OCR. "
                "On CPU, OCR runtime scales with pixel count — lower = much faster. "
                "1200–1800 is usually enough for printed pages."
            ),
        )
    )
    show_debug = st.sidebar.checkbox("Show layout overlay", value=False)

    st.sidebar.divider()
    st.sidebar.caption("**User preferences (long-term memory)**")
    ltm = _get_long_term_memory()
    prefs = ltm.prefs
    new_font = st.sidebar.text_input("Default font", value=prefs.default_font)
    new_size = st.sidebar.number_input("Font size (pt)", min_value=8, max_value=24, value=int(prefs.default_font_size_pt))
    new_spacing = st.sidebar.number_input("Line spacing", min_value=0.8, max_value=3.0, value=float(prefs.default_line_spacing), step=0.05)
    new_bias = st.sidebar.selectbox(
        "Alignment bias",
        ["auto", "left", "center", "right"],
        index=["auto", "left", "center", "right"].index(prefs.preferred_alignment_bias),
    )
    enable_cleanup = st.sidebar.checkbox(
        "LLM OCR cleanup (slow, opt-in)",
        value=getattr(prefs, "enable_llm_cleanup", False),
        help=(
            "Off (default): fast path — one LLM call total (~5–15s). "
            "On: adds a second LLM call to fix character-level OCR errors "
            "(~30–90s extra). Disable if speed matters."
        ),
    )
    if (
        new_font != prefs.default_font
        or new_size != prefs.default_font_size_pt
        or abs(new_spacing - prefs.default_line_spacing) > 1e-6
        or new_bias != prefs.preferred_alignment_bias
        or enable_cleanup != getattr(prefs, "enable_llm_cleanup", False)
    ):
        ltm.update_prefs(
            default_font=new_font,
            default_font_size_pt=int(new_size),
            default_line_spacing=float(new_spacing),
            preferred_alignment_bias=new_bias,
            enable_llm_cleanup=bool(enable_cleanup),
        )
        st.sidebar.success("Preferences saved.")

    if st.sidebar.button("Reset long-term memory", type="secondary"):
        ltm.reset()
        st.sidebar.warning("Long-term memory reset.")

    st.sidebar.divider()
    st.sidebar.caption("**Ethics, privacy & legal notice**")
    st.sidebar.markdown(
        "- Images are processed in-memory only and never written to disk.\n"
        "- Extracted text (no images) may be sent to the Hugging Face Inference API "
        "for cleanup/classification — disable in Privacy mode.\n"
        "- You are responsible for having lawful authority to process the document "
        "(PECA-2016 / GDPR).\n"
        "- The agent flags personal data (CNIC, phone, email) and requires acknowledgement.\n"
        "- Decisions are logged with rationale; see _Show reasoning_ below output.\n"
        "- ACM / IEEE Code of Ethics: avoid harm, respect privacy, be honest about limitations."
    )

    return {
        "agent_enabled": agent_enabled,
        "autonomy": autonomy,
        "show_debug": show_debug,
        "ocr_max_long_side_px": cfg.ocr_max_long_side_px,
    }


# ---------------------------------------------------------------------------
# Phase-1 deterministic path (kept for the "agent off" toggle)
# ---------------------------------------------------------------------------


def _run_phase1(engine: LightOnOcrEngine, image: Image.Image, cfg: AppConfig, show_debug: bool, file_stem: str) -> None:
    with st.spinner("Running OCR..."):
        text = engine.ocr(image=image, max_new_tokens=cfg.max_new_tokens)
    with st.spinner("Analyzing layout..."):
        layout = analyze_layout(image)
    if show_debug:
        st.subheader("Layout debug")
        st.image(render_layout_debug(image, layout), use_container_width=True)
    with st.spinner("Generating .docx..."):
        docx_bytes = build_docx(title="", ocr_text=text, layout=layout)
    st.download_button(
        "Download .docx",
        data=docx_bytes,
        file_name=f"{file_stem}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Agentic path
# ---------------------------------------------------------------------------


def _stage_index(name: Optional[str]) -> int:
    if name is None or name not in STAGE_ORDER:
        return 0
    return STAGE_ORDER.index(name)


def _run_agentic(
    engine: LightOnOcrEngine,
    image: Image.Image,
    cfg: AppConfig,
    autonomy: AutonomyLevel,
    show_debug: bool,
    file_stem: str,
) -> None:
    orch = _get_orchestrator(engine, autonomy, cfg.ocr_max_long_side_px)

    # Surface LLM availability up front.
    st.caption(orch.llm.status_message)

    # Stage 1 — Observe
    if _stage_index(orch.result.stage_completed) < _stage_index("observe"):
        with st.spinner("Observing image..."):
            orch.observe(image)

    if orch.result.hitl_required and orch.result.hitl_kind == "pre_ocr":
        st.warning(orch.result.hitl_message)
        col_a, col_b = st.columns(2)
        if col_a.button("Continue anyway"):
            orch.result.hitl_required = False
            orch.result.hitl_kind = None
            st.rerun()
        if col_b.button("Cancel"):
            st.stop()
        st.stop()

    # Stage 2 — Interpret
    if _stage_index(orch.result.stage_completed) < _stage_index("interpret"):
        with st.spinner("Reading text and analysing layout..."):
            orch.interpret(image, max_new_tokens=cfg.max_new_tokens)

    if orch.result.error:
        st.error(orch.result.error)
        st.stop()

    if show_debug and orch.result.layout is not None:
        with st.expander("Layout overlay", expanded=False):
            st.image(render_layout_debug(image, orch.result.layout), use_container_width=True)

    # PII gate
    if orch.result.hitl_required and orch.result.hitl_kind == "post_ocr_pii" and not orch.result.pii_acknowledged:
        st.error(orch.result.hitl_message)
        with st.expander("Detected matches (redacted)"):
            for f in orch.result.pii_findings:
                st.write(f"- **{f['kind']}**: `{f['match']}`")
        ack = st.checkbox(
            "I confirm I have lawful authority and consent to process this content (PECA-2016 / GDPR).",
            key="pii_ack",
        )
        if ack:
            orch.result.pii_acknowledged = True
            orch.result.hitl_required = False
            orch.result.hitl_kind = None
            st.rerun()
        st.stop()

    # Post-OCR review (assisted-only soft prompt)
    st.subheader("Extracted text")
    if orch.result.ocr_confidence is not None:
        st.caption(f"OCR heuristic confidence: **{orch.result.ocr_confidence:.2f}**")
    edited_text = st.text_area(
        "Review / edit the OCR output before formatting:",
        value=orch.result.ocr_text or "",
        height=240,
        key="ocr_edit_box",
    )

    # Doc-type override
    detected = orch.result.doc_type or "other"
    st.subheader("Document type")
    st.caption(
        f"Agent classified this as **{detected}** "
        f"(confidence {orch.result.doc_type_confidence or 0:.2f})."
    )
    options = list(DOC_TYPES)
    chosen_type = st.selectbox(
        "Override classification:",
        options=options,
        index=options.index(detected) if detected in options else options.index("other"),
        key="doc_type_select",
    )

    # Stage 3 — Decide (run on every render so user edits propagate)
    if (
        edited_text != (orch.result.cleaned_text or orch.result.ocr_text)
        or chosen_type != orch.result.doc_type
    ):
        # The user edited something — feed back into memory before deciding.
        orch.learn(
            edited_text=edited_text if edited_text != orch.result.ocr_text else None,
            overridden_doc_type=chosen_type if chosen_type != orch.result.doc_type else None,
        )

    if _stage_index(orch.result.stage_completed) < _stage_index("decide"):
        with st.spinner("Deciding formatting strategy..."):
            orch.decide()

    # Show reasoning so far
    with st.expander("Show reasoning (agent decision trace)", expanded=False):
        st.markdown(orch.log.render_markdown())

    # Block preview if LLM produced one
    if orch.result.structured_blocks:
        with st.expander(f"Agent-suggested blocks ({len(orch.result.structured_blocks)})", expanded=False):
            for i, blk in enumerate(orch.result.structured_blocks, 1):
                summary = blk.get("text") or blk.get("items") or blk.get("table") or ""
                if isinstance(summary, list):
                    summary = " · ".join(str(s) for s in summary[:3]) + ("…" if len(summary) > 3 else "")
                st.write(f"**{i}. {blk.get('type', 'paragraph')}** — {str(summary)[:160]}")
    else:
        st.caption("Agent fell back to the deterministic regex parser for block structure.")

    # Stage 4 — Act
    if st.button("Build .docx", type="primary"):
        with st.spinner("Generating .docx..."):
            orch.act(title="")
            orch.reflect()
        st.session_state["_built"] = True

    if st.session_state.get("_built") and orch.result.docx_bytes:
        st.download_button(
            "Download .docx",
            data=orch.result.docx_bytes,
            file_name=f"{file_stem}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

        st.divider()
        st.caption("Help the agent learn — was this result correct?")
        col_y, col_n = st.columns(2)
        if col_y.button("👍 Accept", use_container_width=True):
            orch.learn(accepted=True)
            st.success("Recorded acceptance — thank you.")
        if col_n.button("👎 Reject", use_container_width=True):
            orch.learn(accepted=False)
            st.info("Recorded rejection — the agent will weigh this in future runs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Image → Word (Agentic)", layout="wide")
    cfg = AppConfig()

    st.title("Image → Word Converter — Agentic Edition")
    st.caption(
        "An LLM-driven agent perceives the image, decides how to format it, calls "
        "OCR / layout / docx tools, and learns from your edits. Disable the agent "
        "in the sidebar to see the deterministic Phase-1 pipeline."
    )

    sb = _sidebar(cfg)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if not uploaded:
            st.info("Upload an image to begin.")
            st.stop()
        image = _load_image(uploaded)
        st.image(image, caption="Input image", use_container_width=True)

    file_stem = (uploaded.name or "document").rsplit(".", 1)[0]

    # New upload → reset agent state
    if st.session_state.get("_last_filename") != uploaded.name:
        for k in list(st.session_state.keys()):
            if k.startswith("_orch_") or k in ("_built", "pii_ack"):
                del st.session_state[k]
        st.session_state["_last_filename"] = uploaded.name

    with col_right:
        with st.spinner("Loading OCR model (first run only)..."):
            engine = _get_engine(cfg.model_id)

        if not sb["agent_enabled"]:
            st.info("Agent disabled — running deterministic Phase-1 pipeline.")
            if st.button("Convert to .docx", type="primary"):
                _run_phase1(engine, image, cfg, sb["show_debug"], file_stem)
            return

        cfg.ocr_max_long_side_px = sb["ocr_max_long_side_px"]
        _run_agentic(engine, image, cfg, sb["autonomy"], sb["show_debug"], file_stem)


if __name__ == "__main__":
    main()
