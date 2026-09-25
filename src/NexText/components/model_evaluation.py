import os

import pandas as pd
import torch
from datasets import load_from_disk
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from NexText.entity.config_entity import ModelEvaluationConfig
from NexText.logging import logger


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(
        self,
        list_of_elements,
        batch_size
    ):
        """Split elements into smaller batches."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        model,
        tokenizer,
        batch_size=1,
        device=None,
        column_text="dialogue",
        column_summary="summary"
    ):
        """
        Generate summaries for the test dataset and calculate ROUGE scores.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        article_batches = self.generate_batch_sized_chunks(
            dataset[column_text],
            batch_size
        )

        target_batches = self.generate_batch_sized_chunks(
            dataset[column_summary],
            batch_size
        )

        predictions = []
        references = []

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            desc="Evaluating",
            total=(len(dataset) + batch_size - 1) // batch_size
        ):

            inputs = tokenizer(
                article_batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                summaries = model.generate(
                    **inputs,
                    length_penalty=0.8,
                    num_beams=4,
                    max_new_tokens=128
                )

            decoded_summaries = tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions.extend(decoded_summaries)
            references.extend(target_batch)

        # ROUGE calculation
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for prediction, reference in zip(
            predictions,
            references
        ):

            scores = scorer.score(
                reference,
                prediction
            )

            rouge1_scores.append(
                scores["rouge1"].fmeasure
            )

            rouge2_scores.append(
                scores["rouge2"].fmeasure
            )

            rougeL_scores.append(
                scores["rougeL"].fmeasure
            )

        rouge_dict = {
            "rouge1": sum(rouge1_scores) / len(rouge1_scores),
            "rouge2": sum(rouge2_scores) / len(rouge2_scores),
            "rougeL": sum(rougeL_scores) / len(rougeL_scores)
        }

        return rouge_dict

    def evaluate(self):

        # Automatically use available device
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Evaluation device: {device}")

        if torch.cuda.is_available():
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        # Load trained model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        )

        model = model.to(device)
        model.eval()

        logger.info("Trained model loaded successfully.")

        # Load transformed dataset
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        test_dataset = dataset_samsum_pt["test"]

        logger.info(
            f"Test samples: {len(test_dataset)}"
        )

        # Calculate ROUGE scores
        score = self.calculate_metric_on_test_ds(
            dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            device=device,
            column_text="dialogue",
            column_summary="summary"
        )

        logger.info(
            f"ROUGE-1: {score['rouge1']:.4f}"
        )

        logger.info(
            f"ROUGE-2: {score['rouge2']:.4f}"
        )

        logger.info(
            f"ROUGE-L: {score['rougeL']:.4f}"
        )

        # Save metrics
        os.makedirs(
            os.path.dirname(
                str(self.config.metric_file_name)
            ),
            exist_ok=True
        )

        df = pd.DataFrame(
            [score],
            index=["pegasus"]
        )

        df.to_csv(
            self.config.metric_file_name
        )

        logger.info(
            f"Metrics saved to: "
            f"{self.config.metric_file_name}"
        )

        return score