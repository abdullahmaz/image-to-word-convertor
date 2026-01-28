---
title: Image to Word Converter
emoji: "\U0001F4C4"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Image → Word Converter (LightOnOCR-1B)

MVP app that:

- Accepts **JPG/PNG** scans
- Extracts text using **`lightonai/LightOnOCR-1B-1025`**
- Detects **basic layout** from the image (paragraphs, alignment, rough bold/italic)
- Exports a formatted **`.docx`**

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face Spaces (Docker)

This repo includes a `Dockerfile` that runs Streamlit on port **7860** (as required by Spaces).

