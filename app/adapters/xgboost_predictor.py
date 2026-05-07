import joblib
import numpy as np

from app.ports.quality_predictor import QualityPredictor


class XGBoostQualityPredictor(QualityPredictor):

    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, features: np.ndarray) -> float:

        features = features.reshape(1, -1)
        features = self.scaler.transform(features)

        score = self.model.predict(features)[0]

        return float(score)
    