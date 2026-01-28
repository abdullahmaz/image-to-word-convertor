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

## Deploy (recommended for MVP submission)

### Option A — Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new Streamlit app pointing to `app.py`.

### Option B — Hugging Face Spaces (Streamlit)

1. Create a Space (SDK: **Streamlit**).
2. Upload this repository files (or connect the GitHub repo).
3. The Space will install `requirements.txt` and run `app.py`.

## Notes

- First run downloads the model weights (can take time).
- CPU-only deployments will be slow, but functional.

