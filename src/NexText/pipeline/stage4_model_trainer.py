from NexText.config.configuration import ConfigurationManager
from NexText.components.model_trainer import ModelTrainer
from NexText.logging import logger


STAGE_NAME = "Model Training"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(
                f"\n{'=' * 20} {STAGE_NAME} Started {'=' * 20}"
            )

            config = ConfigurationManager()

            model_trainer_config = (
                config.get_model_trainer_config()
            )

            model_trainer = ModelTrainer(
                config=model_trainer_config
            )

            model_trainer.train()

            logger.info(
                f"{'=' * 20} {STAGE_NAME} Completed {'=' * 20}"
            )

        except Exception as e:
            logger.exception(f"{STAGE_NAME} failed: {e}")
            raise e
