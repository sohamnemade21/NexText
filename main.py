from NexText.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline
from NexText.logging import logger


STAGE_NAME = "Data Ingestion"


if __name__ == "__main__":
    try:
        logger.info(
            f"\n{'=' * 20}{STAGE_NAME} Started{'=' * 20}"
        )

        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()

        logger.info(
            f"{'=' * 20}Stage One Completed{'=' * 20}"
        )

    except Exception as e:
        logger.exception(e)
        raise e