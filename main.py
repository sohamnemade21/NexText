from NexText.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline
)

from NexText.pipeline.stage2_data_validation import (
    DataValidationTrainingPipeline
)

from NexText.pipeline.stage3_data_transformation import (
    DataTransformationTrainingPipeline
)

from NexText.pipeline.stage4_model_trainer import (
    ModelTrainerTrainingPipeline
)

from NexText.pipeline.stage5_model_evaluation import (
    ModelEvaluationTrainingPipeline
)

from NexText.logging import logger


STAGE_NAME = "NexText Pipeline"


if __name__ == "__main__":

    try:

        # =========================================================
        # STAGE 1: DATA INGESTION
        # =========================================================

        logger.info(
            f"\n{'=' * 20} Data Ingestion Started "
            f"{'=' * 20}"
        )

        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()

        logger.info(
            f"{'=' * 20} Data Ingestion Completed "
            f"{'=' * 20}"
        )


        # =========================================================
        # STAGE 2: DATA VALIDATION
        # =========================================================

        logger.info(
            f"\n{'=' * 20} Data Validation Started "
            f"{'=' * 20}"
        )

        data_validation_pipeline = DataValidationTrainingPipeline()

        validation_status = data_validation_pipeline.main()

        if not validation_status:
            raise Exception(
                "Data validation failed. "
                "Please check the validation logs."
            )

        logger.info(
            f"{'=' * 20} Data Validation Completed "
            f"{'=' * 20}"
        )


        # =========================================================
        # STAGE 3: DATA TRANSFORMATION
        # =========================================================

        logger.info(
            f"\n{'=' * 20} Data Transformation Started "
            f"{'=' * 20}"
        )

        data_transformation_pipeline = (
            DataTransformationTrainingPipeline()
        )

        data_transformation_pipeline.main()

        logger.info(
            f"{'=' * 20} Data Transformation Completed "
            f"{'=' * 20}"
        )


        # =========================================================
        # STAGE 4: MODEL TRAINING
        # =========================================================

        logger.info(
            f"\n{'=' * 20} Model Training Started "
            f"{'=' * 20}"
        )

        model_trainer_pipeline = (
            ModelTrainerTrainingPipeline()
        )

        model_trainer_pipeline.main()

        logger.info(
            f"{'=' * 20} Model Training Completed "
            f"{'=' * 20}"
        )


        # =========================================================
        # STAGE 5: MODEL EVALUATION
        # =========================================================

        logger.info(
            f"\n{'=' * 20} Model Evaluation Started "
            f"{'=' * 20}"
        )

        model_evaluation_pipeline = (
            ModelEvaluationTrainingPipeline()
        )

        model_evaluation_pipeline.main()

        logger.info(
            f"{'=' * 20} Model Evaluation Completed "
            f"{'=' * 20}"
        )


        # =========================================================
        # PIPELINE COMPLETED
        # =========================================================

        logger.info(
            f"\n{'=' * 20} {STAGE_NAME} Completed "
            f"{'=' * 20}"
        )


    except Exception as e:

        logger.exception(
            f"{STAGE_NAME} failed: {e}"
        )

        raise e