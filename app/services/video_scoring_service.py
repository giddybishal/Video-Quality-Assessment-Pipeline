from app.ports.feature_extractor import FeatureExtractor
from app.ports.quality_predictor import QualityPredictor


class VideoScoringService:

    def __init__(self, extractor: FeatureExtractor, predictor: QualityPredictor):
        self.extractor = extractor
        self.predictor = predictor

    def score(self, video_path: str) -> float:

        features = self.extractor.extract(video_path)

        score = self.predictor.predict(features)

        return score
    