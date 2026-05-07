from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import shutil
import os

from app.adapters.resnet_extractor import ResNetFeatureExtractor
from app.adapters.xgboost_predictor import XGBoostQualityPredictor
from app.adapters.whisper_semantic_scorer import WhisperSemanticScorer

from app.services.video_scoring_service import VideoScoringService
from app.services.final_scoring_service import FinalScoringService


app = FastAPI()


# -----------------------------
# Dependency Injection Setup
# -----------------------------

extractor = ResNetFeatureExtractor()

predictor = XGBoostQualityPredictor(
    model_path="models/model.pkl",
    scaler_path="models/scaler.pkl"
)

semantic_scorer = WhisperSemanticScorer()

vqa_service = VideoScoringService(
    extractor,
    predictor
)

final_service = FinalScoringService(
    vqa_service,
    semantic_scorer
)


# -----------------------------
# API Endpoint
# -----------------------------

@app.post("/score-video")
async def score_video(
    file: UploadFile = File(...),
    campaign_text: str = Form(...)
):

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:

        shutil.copyfileobj(file.file, tmp)

        temp_path = tmp.name

    try:

        # Run full multimodal scoring pipeline
        results = final_service.score(
            temp_path,
            campaign_text
        )

        return {
            "filename": file.filename,
            "campaign_text": campaign_text,
            "results": results
        }

    finally:

        # Cleanup temp file
        os.remove(temp_path)
        