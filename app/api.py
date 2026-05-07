from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil

from app.adapters.resnet_extractor import ResNetFeatureExtractor
from app.adapters.xgboost_predictor import XGBoostQualityPredictor
from app.services.video_scoring_service import VideoScoringService

app = FastAPI()

# Dependency setup (simple DI for now)
extractor = ResNetFeatureExtractor()
predictor = XGBoostQualityPredictor(
    model_path="models/model.pkl",
    scaler_path="models/scaler.pkl"
)

service = VideoScoringService(extractor, predictor)


@app.post("/score-video")
async def score_video(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    # Run pipeline
    score = service.score(temp_path)

    return {
        "filename": file.filename,
        "predicted_mos": score
    }
