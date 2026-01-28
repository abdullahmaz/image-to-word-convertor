from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, LightOnOCRForConditionalGeneration


class LightOnOcrEngine:
    """
    Thin wrapper around LightOnOCR-1B inference (transformers).

    Notes:
    - This model class is provided by the transformers fork pinned in requirements.txt.
    - On CPU, this will be slow but still functional.
    """

    def __init__(self, model_id: str = "lightonai/LightOnOCR-1B-1025") -> None:
        self.model_id = model_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # The notebook uses `dtype=` and `device_map=device`; we keep it robust across CPU/GPU.
        if self.device == "cuda":
            self.model = LightOnOCRForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map="auto",
                attn_implementation="sdpa",
            )
        else:
            self.model = LightOnOCRForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                attn_implementation="sdpa",
            ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def ocr(self, image: Image.Image, max_new_tokens: int = 1024) -> str:
        # Match the official notebook prompt.
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        output_ids = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        input_length = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_length:]

        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

