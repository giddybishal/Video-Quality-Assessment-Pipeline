from adapters.resnet_extractor import ResNetFeatureExtractor
from adapters.xgboost_predictor import XGBoostQualityPredictor
from services.video_scoring_service import VideoScoringService

VIDEO_PATH = "your_video.mp4"

extractor = ResNetFeatureExtractor()
predictor = XGBoostQualityPredictor(
    model_path="models/model.pkl",
    scaler_path="models/scaler.pkl"
)

service = VideoScoringService(extractor, predictor)

score = service.score(VIDEO_PATH)

print("Predicted MOS:", score)
