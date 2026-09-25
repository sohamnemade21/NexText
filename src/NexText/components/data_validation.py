from pathlib import Path

from NexText.constants import PROJECT_ROOT
from NexText.logging import logger
from NexText.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:

        try:
            validation_status = True

            # Always use project root
            data_ingestion_path = (
                PROJECT_ROOT / "artifacts" / "data_ingestion"
            )

            if not data_ingestion_path.exists():
                raise FileNotFoundError(
                    f"Data ingestion directory not found: "
                    f"{data_ingestion_path}"
                )

            logger.info(
                f"Checking data in: {data_ingestion_path}"
            )

            dataset_dir = (
                data_ingestion_path / "samsum_dataset"
                if (data_ingestion_path / "samsum_dataset").exists()
                else data_ingestion_path
            )

            for file in self.config.ALL_REQUIRED_FILES:

                file_path = dataset_dir / file

                if not file_path.exists():
                    validation_status = False

                    logger.error(
                        f"Required file/folder not found: {file_path}"
                    )
                else:
                    logger.info(
                        f"Found required file/folder: {file_path}"
                    )

            # Status file
            status_file = (
                PROJECT_ROOT / self.config.STATUS_FILE
            )

            status_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                status_file,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(
                    f"Validation status: {validation_status}"
                )

            logger.info(
                f"Data validation status: {validation_status}"
            )

            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e