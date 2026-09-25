import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_from_disk

from NexText.logging import logger
from NexText.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt
        )

        # Load Pegasus model (Trainer automatically manages device placement & mixed precision)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        )

        # Data collator
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus
        )

        # Load transformed dataset
        logger.info("Loading transformed dataset...")

        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        logger.info(
            f"Train samples: {len(dataset_samsum_pt['train'])}"
        )

        logger.info(
            f"Validation samples: {len(dataset_samsum_pt['validation'])}"
        )

        # Training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adafactor",
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        # Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        logger.info("Starting model training...")

        trainer.train()

        logger.info("Model training completed.")

        # Save model
        model_path = os.path.join(
            self.config.root_dir,
            "pegasus-samsum-model"
        )

        model_pegasus.save_pretrained(model_path)

        # Save tokenizer
        tokenizer_path = os.path.join(
            self.config.root_dir,
            "tokenizer"
        )

        tokenizer.save_pretrained(tokenizer_path)

        logger.info(
            f"Model saved to: {model_path}"
        )

        logger.info(
            f"Tokenizer saved to: {tokenizer_path}"
        )
