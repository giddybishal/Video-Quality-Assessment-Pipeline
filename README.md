# 🎥 Multimodal Video Campaign Scoring System

## 📌 Overview

This project implements a multimodal AI pipeline for evaluating creator videos against campaign objectives.

The system combines:

- Visual Quality Assessment (VQA)
- Semantic Campaign Matching
- Unified Multimodal Scoring

to estimate how suitable a creator video is for a brand campaign.

The project is built using a Hexagonal (Ports & Adapters) Architecture for modularity and extensibility.

---

# 🧠 Features

## 1. Visual Quality Assessment (VQA)

- Extract frames from videos
- Generate deep visual embeddings using pretrained ResNet50
- Train XGBoost regression model on KonVid-1k dataset
- Predict MOS (Mean Opinion Score) for unseen videos

### Evaluation Results
- R² Score: ~0.50
- Spearman Correlation: ~0.68
- RMSE: ~0.26 MOS

---

## 2. Semantic Campaign Matching

- Transcribe uploaded videos using Whisper
- Encode transcript + campaign text using Sentence Transformers
- Compute semantic similarity using cosine similarity
- Apply lightweight keyword boosting for campaign relevance

Outputs a semantic alignment score between:
- creator video content
- brand campaign objective

---

## 3. Multimodal Score Fusion

Final score combines:
- Visual quality score
- Semantic relevance score

```text
Final Score =
0.6 * Visual Quality +
0.4 * Semantic Relevance
```

Final score is normalized and returned on a 0–5 scale.

---

# 🏗 Architecture

The project follows a simplified Hexagonal Architecture.

## Ports
Abstract interfaces:
- FeatureExtractor
- QualityPredictor
- SemanticScorer

## Adapters
Concrete implementations:
- ResNet50 feature extractor
- XGBoost predictor
- Whisper semantic scorer

## Services
Business orchestration:
- VideoScoringService
- FinalScoringService

## Entry Points
- CLI testing (`main.py`)
- FastAPI inference API (`api.py`)

---

# 🚀 API

FastAPI endpoint:

```text
POST /score-video
```

### Inputs
- Video file upload
- Campaign text

### Output

```json
{
  "filename": "video.mp4",
  "campaign_text": "...",
  "results": {
    "vqa_score": 3.31,
    "semantic_score": 0.83,
    "final_score": 3.65
  }
}
```

---

# 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- XGBoost
- Whisper
- Sentence Transformers
- FastAPI
- Scikit-learn

---

# 📂 Current Capabilities

✔ Video quality prediction  
✔ Semantic campaign matching  
✔ Multimodal score fusion  
✔ FastAPI inference API  
✔ Hexagonal architecture refactor  
✔ Modular ML service design  

---

# 🔭 Future Improvements

- Temporal-aware video models
- Better semantic reasoning
- Campaign-specific fine-tuning
- Batch video processing
- Async inference pipeline
- Frontend dashboard
- Cloud deployment

---

# 🎯 Goal

Build a scalable AI system capable of evaluating creator-generated videos for brand campaign suitability using both visual and semantic intelligence.
