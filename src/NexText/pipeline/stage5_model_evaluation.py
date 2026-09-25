from NexText.config.configuration import ConfigurationManager
from NexText.components.model_evaluation import ModelEvaluation
from NexText.logging import logger


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        try:
            logger.info(
                "\n========== Model Evaluation Started =========="
            )

            config = ConfigurationManager()

            model_evaluation_config = (
                config.get_model_evaluation_config()
            )

            model_evaluation = ModelEvaluation(
                config=model_evaluation_config
            )

            metrics = model_evaluation.evaluate()

            logger.info(
                f"Evaluation Results: {metrics}"
            )

            logger.info(
                "========== Model Evaluation Completed =========="
            )

            return metrics

        except Exception as e:
            logger.exception(
                f"Model Evaluation failed: {e}"
            )
            raise e