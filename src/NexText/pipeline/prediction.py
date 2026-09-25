from NexText.components.model_prediction import ModelPrediction
from NexText.logging import logger


class PredictionPipeline:

    def __init__(self):
        self.predictor = ModelPrediction()

    def predict(self, text: str) -> str:
        logger.info(f"Generating summary for text with length: {len(text)}")
        summary = self.predictor.predict(text)
        logger.info(f"Summary generated successfully with length: {len(summary)}")
        return summary
