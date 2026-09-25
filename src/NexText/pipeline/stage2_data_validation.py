from NexText.config.configuration import ConfigurationManager
from NexText.components.data_validation import DataValidation
from NexText.logging import logger


STAGE_NAME = "Data Validation"


class DataValidationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:
            logger.info(
                f"\n{'=' * 20} {STAGE_NAME} Started {'=' * 20}"
            )

            config = ConfigurationManager()

            data_validation_config = (
                config.get_data_validation_config()
            )

            data_validation = DataValidation(
                config=data_validation_config
            )

            validation_status = (
                data_validation.validate_all_files_exist()
            )

            if validation_status:
                logger.info(
                    "Data validation completed successfully."
                )
            else:
                logger.error(
                    "Data validation failed."
                )

            logger.info(
                f"{'=' * 20} {STAGE_NAME} Completed "
                f"{'=' * 20}"
            )

            return validation_status

        except Exception as e:
            logger.exception(
                f"{STAGE_NAME} failed: {e}"
            )
            raise e