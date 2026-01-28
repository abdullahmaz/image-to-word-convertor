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

## Notes

- First run downloads the model weights (can take time).
- CPU-only deployments will be slow, but functional.

