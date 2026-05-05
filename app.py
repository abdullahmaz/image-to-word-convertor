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
    st.sidebar.header("⚙️ Settings")
    st.sidebar.caption("Tune how the converter works. Sensible defaults — change when you need to.")

    agent_enabled = st.sidebar.toggle(
        "Use AI agent",
        value=True,
        help="On (recommended): an LLM agent classifies the document, cleans the text, and chooses formatting. Off: a fast, rules-only pipeline with no LLM calls.",
    )

    autonomy: AutonomyLevel = st.sidebar.select_slider(  # type: ignore[assignment]
        "How much should the agent decide on its own?",
        options=list(AUTONOMY_LEVELS),
        value=DEFAULT_AUTONOMY,
        help=(
            "**Manual** — confirm every step. "
            "**Assisted** — agent runs but pauses for review at key moments. "
            "**Autonomous** — end-to-end, only stops on personal data or errors."
        ),
    )

    with st.sidebar.expander("🔍 OCR & quality", expanded=False):
        cfg.max_new_tokens = int(
            st.slider(
                "Maximum text length",
                256, 4096, cfg.max_new_tokens, step=128,
                help="Cap on how much text the OCR model can produce. Raise this for long, dense pages.",
            )
        )
        cfg.ocr_max_long_side_px = int(
            st.slider(
                "Image resolution",
                min_value=800, max_value=3200, value=cfg.ocr_max_long_side_px, step=200,
                help="Bigger images mean slower OCR. 1200–1800 px is plenty for printed pages; raise it only for fine print.",
            )
        )
        show_debug = st.checkbox(
            "Show layout overlay",
            value=False,
            help="Visualises the detected paragraphs and alignment on top of your image.",
        )

    with st.sidebar.expander("🎨 Document defaults", expanded=False):
        st.caption("These remembered preferences shape every Word file you generate.")
        ltm = _get_long_term_memory()
        prefs = ltm.prefs
        new_font = st.text_input("Font", value=prefs.default_font)
        new_size = st.number_input(
            "Font size (pt)", min_value=8, max_value=24,
            value=int(prefs.default_font_size_pt),
        )
        new_spacing = st.number_input(
            "Line spacing", min_value=0.8, max_value=3.0,
            value=float(prefs.default_line_spacing), step=0.05,
        )
        new_bias = st.selectbox(
            "Text alignment",
            ["auto", "left", "center", "right"],
            index=["auto", "left", "center", "right"].index(prefs.preferred_alignment_bias),
            help="‘Auto’ lets the agent decide based on the layout it sees.",
        )
        enable_cleanup = st.checkbox(
            "Polish OCR with LLM (slower, more accurate)",
            value=getattr(prefs, "enable_llm_cleanup", False),
            help="Adds an extra AI pass to fix typos and broken words. Costs ~30–90s. Leave off if speed matters more than precision.",
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
            st.success("Preferences saved.")

        if st.button("Forget my preferences", type="secondary", use_container_width=True):
            ltm.reset()
            st.warning("Preferences cleared — back to defaults.")

    with st.sidebar.expander("🔒 Privacy & responsible use", expanded=False):
        st.markdown(
            "- **Your images stay in memory** — never written to disk.\n"
            "- **Extracted text only** may be sent to Hugging Face for cleanup and classification.\n"
            "- **You confirm** you have the right to process this document (PECA-2016 / GDPR).\n"
            "- **Personal data flagged** — CNIC, phone, and email patterns require acknowledgement before continuing.\n"
            "- **Every decision logged** with reasoning — open _Show reasoning_ under the output.\n"
            "- Aligned with the **ACM / IEEE Code of Ethics**: avoid harm, respect privacy, be honest about limits."
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
    with st.spinner("Reading the text…"):
        text = engine.ocr(image=image, max_new_tokens=cfg.max_new_tokens)
    with st.spinner("Working out the layout…"):
        layout = analyze_layout(image)
    if show_debug:
        st.subheader("Detected layout")
        st.image(render_layout_debug(image, layout), use_container_width=True)
    with st.spinner("Building your Word document…"):
        docx_bytes = build_docx(title="", ocr_text=text, layout=layout)
    st.success("Done — your document is ready.")
    st.download_button(
        "⬇️ Download Word document",
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
        with st.spinner("Taking a first look at your image…"):
            orch.observe(image)

    if orch.result.hitl_required and orch.result.hitl_kind == "pre_ocr":
        st.warning(orch.result.hitl_message)
        col_a, col_b = st.columns(2)
        if col_a.button("Continue anyway", use_container_width=True):
            orch.result.hitl_required = False
            orch.result.hitl_kind = None
            st.rerun()
        if col_b.button("Cancel", use_container_width=True):
            st.stop()
        st.stop()

    # Stage 2 — Interpret
    if _stage_index(orch.result.stage_completed) < _stage_index("interpret"):
        with st.spinner("Reading the text and figuring out the layout…"):
            orch.interpret(image, max_new_tokens=cfg.max_new_tokens)

    if orch.result.error:
        st.error(orch.result.error)
        st.stop()

    if show_debug and orch.result.layout is not None:
        with st.expander("📐 Detected layout overlay", expanded=False):
            st.image(render_layout_debug(image, orch.result.layout), use_container_width=True)

    # PII gate
    if orch.result.hitl_required and orch.result.hitl_kind == "post_ocr_pii" and not orch.result.pii_acknowledged:
        st.error(orch.result.hitl_message)
        with st.expander("What we found (redacted)"):
            for f in orch.result.pii_findings:
                st.write(f"- **{f['kind']}**: `{f['match']}`")
        ack = st.checkbox(
            "I confirm I have the legal right and consent to process this content (PECA-2016 / GDPR).",
            key="pii_ack",
        )
        if ack:
            orch.result.pii_acknowledged = True
            orch.result.hitl_required = False
            orch.result.hitl_kind = None
            st.rerun()
        st.stop()

    # Post-OCR review (assisted-only soft prompt)
    st.subheader("📝 Review the extracted text")
    if orch.result.ocr_confidence is not None:
        conf = orch.result.ocr_confidence
        emoji = "🟢" if conf >= 0.75 else "🟡" if conf >= 0.5 else "🔴"
        st.caption(f"{emoji} Confidence: **{conf:.0%}** — edit anything that looks off below.")
    edited_text = st.text_area(
        "Make corrections here before we build your document:",
        value=orch.result.ocr_text or "",
        height=240,
        key="ocr_edit_box",
    )

    # Doc-type override
    detected = orch.result.doc_type or "other"
    st.subheader("📂 Document type")
    st.caption(
        f"Looks like a **{detected}** to me "
        f"(confidence {orch.result.doc_type_confidence or 0:.0%}). Change it if I got it wrong."
    )
    options = list(DOC_TYPES)
    chosen_type = st.selectbox(
        "Document type",
        options=options,
        index=options.index(detected) if detected in options else options.index("other"),
        key="doc_type_select",
        label_visibility="collapsed",
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
        with st.spinner("Choosing how to format your document…"):
            orch.decide()

    # Show reasoning so far
    with st.expander("🧠 See how the agent decided", expanded=False):
        st.markdown(orch.log.render_markdown())

    # Block preview if LLM produced one
    if orch.result.structured_blocks:
        with st.expander(f"🧱 Suggested layout ({len(orch.result.structured_blocks)} blocks)", expanded=False):
            for i, blk in enumerate(orch.result.structured_blocks, 1):
                summary = blk.get("text") or blk.get("items") or blk.get("table") or ""
                if isinstance(summary, list):
                    summary = " · ".join(str(s) for s in summary[:3]) + ("…" if len(summary) > 3 else "")
                st.write(f"**{i}. {blk.get('type', 'paragraph')}** — {str(summary)[:160]}")
    else:
        st.caption("ℹ️ Using the rules-based layout parser for this one — the agent didn't suggest a structure.")

    st.divider()

    # Stage 4 — Act
    if st.button("✨ Build my Word document", type="primary", use_container_width=True):
        with st.spinner("Putting your Word document together…"):
            orch.act(title="")
            orch.reflect()
        st.session_state["_built"] = True

    if st.session_state.get("_built") and orch.result.docx_bytes:
        st.success("All done — your document is ready below.")
        st.download_button(
            "⬇️ Download Word document",
            data=orch.result.docx_bytes,
            file_name=f"{file_stem}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

        st.divider()
        st.caption("**How did we do?** Your feedback helps the agent improve next time.")
        col_y, col_n = st.columns(2)
        if col_y.button("👍 Looks great", use_container_width=True):
            orch.learn(accepted=True)
            st.success("Thanks — noted!")
        if col_n.button("👎 Not quite right", use_container_width=True):
            orch.learn(accepted=False)
            st.info("Got it — the agent will adjust for next time.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -0.015em; }
  h1 { font-weight: 700; }
  .hero {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 60%, #FAF5FF 100%);
    border: 1px solid #E0E7FF;
    margin-bottom: 1.25rem;
  }
  .hero-title { margin: 0 0 .35rem 0; font-size: 1.85rem; font-weight: 700; color: #1E1B4B; }
  .hero-sub { margin: 0; color: #475569; font-size: 1rem; line-height: 1.5; }
  .hero-pill {
    display: inline-block; padding: 2px 10px; margin-bottom: .6rem;
    background: #EEF2FF; color: #4338CA; border-radius: 999px;
    font-size: .75rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
  }
  [data-testid="stFileUploader"] section {
    border: 2px dashed #C7D2FE !important; border-radius: 12px; background: #FAFAFF;
    transition: border-color .15s ease, background .15s ease;
  }
  [data-testid="stFileUploader"] section:hover {
    border-color: #818CF8 !important; background: #F5F3FF;
  }
  .stButton button[kind="primary"], .stDownloadButton button {
    border-radius: 10px; font-weight: 600;
  }
  .stDownloadButton button { background: #10B981; color: #fff; border: 0; }
  .stDownloadButton button:hover { background: #059669; color: #fff; }
  section[data-testid="stSidebar"] { background: #FAFAFB; }
  .step-label {
    display: inline-block; padding: 2px 9px; margin-right: 6px;
    background: #EEF2FF; color: #4338CA; border-radius: 6px;
    font-size: .72rem; font-weight: 700; letter-spacing: .03em;
  }
  .footer {
    margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #E2E8F0;
    color: #64748B; font-size: .82rem; text-align: center;
  }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Image → Word",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    cfg = AppConfig()

    st.markdown(
        """
        <div class="hero">
          <span class="hero-pill">AI · OCR · Word</span>
          <div class="hero-title">📄 Image → Word</div>
          <p class="hero-sub">
            Drop in a scanned page or photo and get back a clean, editable Word document.
            An on-device agent reads the text, infers the layout, rebuilds it as <code>.docx</code>,
            and learns from your edits.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sb = _sidebar(cfg)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="step-label">STEP 1</span> **Upload your image**', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or browse — JPG or PNG, single page",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if not uploaded:
            st.info("👆 Choose a JPG or PNG to get started. Best results on flat, well-lit scans.")
            st.markdown('<div class="footer">Built by the team · LightOnOCR-1B · Streamlit</div>', unsafe_allow_html=True)
            st.stop()
        image = _load_image(uploaded)
        st.image(image, caption="Your image", use_container_width=True)

    file_stem = (uploaded.name or "document").rsplit(".", 1)[0]

    # New upload → reset agent state
    if st.session_state.get("_last_filename") != uploaded.name:
        for k in list(st.session_state.keys()):
            if k.startswith("_orch_") or k in ("_built", "pii_ack"):
                del st.session_state[k]
        st.session_state["_last_filename"] = uploaded.name

    with col_right:
        st.markdown('<span class="step-label">STEP 2</span> **Convert**', unsafe_allow_html=True)
        with st.spinner("Warming up the OCR model — this only happens on the first run…"):
            engine = _get_engine(cfg.model_id)

        if not sb["agent_enabled"]:
            st.info("Agent is **off** — using the deterministic pipeline (faster, no LLM).")
            if st.button("Convert to Word", type="primary", use_container_width=True):
                _run_phase1(engine, image, cfg, sb["show_debug"], file_stem)
            st.markdown('<div class="footer">Built by the team · LightOnOCR-1B · Streamlit</div>', unsafe_allow_html=True)
            return

        cfg.ocr_max_long_side_px = sb["ocr_max_long_side_px"]
        _run_agentic(engine, image, cfg, sb["autonomy"], sb["show_debug"], file_stem)
        st.markdown('<div class="footer">Built by the team · LightOnOCR-1B · Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
