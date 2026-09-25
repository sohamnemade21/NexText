from NexText.config.configuration import ConfigurationManager
from NexText.components.data_transformation import DataTransformation
from NexText.logging import logger


STAGE_NAME = "Data Transformation"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(
                f"\n{'=' * 20} {STAGE_NAME} Started {'=' * 20}"
            )

            config = ConfigurationManager()

            data_transformation_config = (
                config.get_data_transformation_config()
            )

            data_transformation = DataTransformation(
                config=data_transformation_config
            )

            data_transformation.convert()

            logger.info(
                f"{'=' * 20} {STAGE_NAME} Completed {'=' * 20}"
            )

        except Exception as e:
            logger.exception(f"{STAGE_NAME} failed: {e}")
            raise e