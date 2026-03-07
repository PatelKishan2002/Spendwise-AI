"""
Receipt Parser Module - SpendWise AI

Uses Donut (Document Understanding Transformer) for receipt parsing.
"""

import torch
import numpy as np
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


class ReceiptParser:
    """Production-ready receipt parser using Donut"""

    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                "Receipt OCR model could not be loaded. This usually happens when the machine "
                "has no internet access / DNS is failing, or the Hugging Face model cache is empty. "
                "Connect to the internet once to download the model, or pre-download it into the "
                "Hugging Face cache, then retry.\n\n"
                f"Original error: {e}"
            ) from e
        self.model.to(self.device)
        self.model.eval()
        self.task_prompt = "<s_cord-v2>"

    def parse(self, image) -> dict:
        """Parse receipt image to structured data"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values, decoder_input_ids=decoder_input_ids,
                max_length=512, early_stopping=True, num_beams=4,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        sequence = self.processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

        return self.processor.token2json(sequence)
