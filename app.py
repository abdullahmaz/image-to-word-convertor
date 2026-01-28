from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from PIL import Image

from src.docx_writer import build_docx
from src.layout import analyze_layout, render_layout_debug
from src.ocr_lighton import LightOnOcrEngine


@dataclass
class AppConfig:
    model_id: str = "lightonai/LightOnOCR-1B-1025"
    max_new_tokens: int = 1024


def _load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return image


@st.cache_resource(show_spinner=False)
def _get_engine(model_id: str) -> LightOnOcrEngine:
    return LightOnOcrEngine(model_id=model_id)


def main() -> None:
    st.set_page_config(page_title="Image → Word (OCR)", layout="wide")
    cfg = AppConfig()

    st.title("Image → Word Converter (LightOnOCR-1B)")
    st.caption(
        "Upload a JPG/PNG, extract text with LightOnOCR-1B, then download a formatted .docx "
        "(basic paragraphs + alignment + rough bold/italic detection)."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if not uploaded:
            st.stop()
        image = _load_image(uploaded)
        st.image(image, caption="Input image", use_container_width=True)

        with st.expander("Advanced"):
            cfg.max_new_tokens = int(
                st.slider(
                    "Max generated tokens",
                    min_value=256,
                    max_value=4096,
                    value=cfg.max_new_tokens,
                    step=128,
                )
            )
            show_debug = st.checkbox("Show detected layout overlay", value=True)

    with col_right:
        run = st.button("Convert to .docx", type="primary", use_container_width=True)
        if not run:
            st.info("Click **Convert to .docx** to run OCR and generate the Word file.")
            st.stop()

        with st.spinner("Loading model (first run can take a while)..."):
            engine = _get_engine(cfg.model_id)

        with st.spinner("Running OCR..."):
            text = engine.ocr(image=image, max_new_tokens=cfg.max_new_tokens)

        with st.spinner("Analyzing layout & formatting..."):
            layout = analyze_layout(image)

        if show_debug:
            st.subheader("Layout debug")
            debug_img = render_layout_debug(image, layout)
            st.image(debug_img, caption="Detected lines + inferred alignment/bold/italic", use_container_width=True)

        with st.spinner("Generating .docx..."):
            docx_bytes = build_docx(
                title=(uploaded.name or "document").rsplit(".", 1)[0],
                ocr_text=text,
                layout=layout,
            )

        st.download_button(
            "Download .docx",
            data=docx_bytes,
            file_name=f"{(uploaded.name or 'document').rsplit('.', 1)[0]}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

