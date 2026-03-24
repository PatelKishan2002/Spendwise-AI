"""Donut-based receipt parser (CORD fine-tune)."""

import torch
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MAX_IMAGE_SIZE = 1280


def _preprocess_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) <= MAX_IMAGE_SIZE:
        return image
    scale = MAX_IMAGE_SIZE / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class ReceiptParser:

    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.task_prompt = "<s_cord-v2>"

    def parse(self, image) -> dict:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        image = _preprocess_image(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values, decoder_input_ids=decoder_input_ids,
                max_length=1024, early_stopping=True, num_beams=5,
                length_penalty=0.8,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        sequence = self.processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        result = self.processor.token2json(sequence)
        if isinstance(result, dict):
            result["parser"] = "donut"
        return result
