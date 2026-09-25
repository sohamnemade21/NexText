import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class ModelPrediction:

    def __init__(self):

        self.model_path = (
            "artifacts/model_trainer/pegasus-samsum-model"
        )

        self.tokenizer_path = (
            "artifacts/model_trainer/tokenizer"
        )

        # Automatically use GPU if available
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path
        )

        # Load trained model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path
        )

        # Remove conflicting generation setting
        self.model.generation_config.max_length = None

        # Move model to available device
        self.model = self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("Model loaded successfully!")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            summary_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                length_penalty=0.8,
                early_stopping=True
            )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary