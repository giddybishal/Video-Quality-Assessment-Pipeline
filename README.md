# 🎥 Video Quality Assessment Pipeline

## 📌 Overview

This project implements an end-to-end machine learning pipeline that predicts video quality (MOS score) from raw video input.

It uses pretrained deep visual features (ResNet50) combined with an XGBoost regression model and is currently being refactored into a Hexagonal Architecture for better modularity and extensibility.

---

## 🧠 Current System Architecture

The system is structured using a **Hexagonal (Ports & Adapters) Architecture**:

Video Input
↓
Feature Extractor (ResNet50 Adapter)
↓
Video Feature Vector (2048-dim)
↓
Quality Predictor (XGBoost Adapter)
↓
Predicted MOS Score


Core logic is isolated from infrastructure concerns such as model implementation and future API exposure.

---

## 🧠 What I Have Built So Far

### 1. Video Feature Extraction
- Extract frames from videos at fixed intervals
- Use pretrained ResNet50 to extract 2048-dimensional embeddings
- Aggregate frame-level features into a single video representation
- Save features as `.npy` for reuse and reproducibility

---

### 2. Dataset Preparation
- Match extracted video features with MOS labels from CSV
- Construct training dataset:
  - `X` → 2048-dimensional video embeddings  
  - `y` → MOS scores (human-labeled quality scores)

---

### 3. Model Training
- Train XGBoost regression model on extracted features
- Apply feature normalization using `StandardScaler`
- Evaluation metrics:
  - MSE
  - R² Score
  - Spearman Correlation (ranking consistency)

---

### 4. Inference Pipeline
- Built a local prediction pipeline for new videos
- Input: raw video file
- Output: predicted MOS score

---

### 5. Architecture Refactor (In Progress)
Refactored into Hexagonal Architecture:

- **Ports**
  - FeatureExtractor (video → embedding)
  - QualityPredictor (features → score)

- **Adapters**
  - ResNet50 feature extractor
  - XGBoost regression model

- **Service Layer**
  - VideoScoringService orchestrates full pipeline

- **Entry Points**
  - `main.py` → local testing (CLI runner)
  - Future: FastAPI (`api.py`) for production inference

---

## 📊 Current Results

- R² Score: ~0.50  
- Spearman Correlation: ~0.68  
- RMSE: ~0.26 MOS  

The model demonstrates strong ranking ability aligned with human-perceived video quality.

---

## 🚧 Challenges Faced

- Mapping video-level labels (MOS) to extracted frame features
- Handling computational cost of video processing locally
- Understanding and interpreting deep feature vectors (ResNet outputs)
- Ensuring consistent preprocessing across training and inference pipelines
- Managing pathing and modular structure during refactor

---

## 🔭 Next Steps

### 1. Improve Model Performance
- Improve temporal feature aggregation (better than simple averaging)
- Experiment with stronger encoders (CLIP / video transformers)
- Hyperparameter tuning for XGBoost

---

### 2. Extend Scoring System
- Add semantic / agenda matching layer for campaign relevance
- Combine multiple scoring signals:
  - Visual quality score
  - Semantic relevance score
  - Engagement proxy score
- Build unified video quality scoring function

---

### 3. API Layer (Next Implementation)
- Add FastAPI service (`api.py`)
- Enable video upload and real-time scoring
- Convert pipeline into deployable ML service

---

### 4. Architecture Completion
Continue evolving hexagonal structure:

- Domain → scoring logic
- Application → orchestration service
- Ports → abstract ML interfaces
- Adapters → ML models + API + data loaders

---

## 🏁 Current Status

✔ End-to-end ML pipeline working  
✔ Feature extraction + regression model complete  
✔ Hexagonal architecture introduced  
✔ Local inference working  
⏳ API layer pending (FastAPI integration next step)  

---

## 🚀 Goal

Build a modular, production-ready **Video Quality Assessment System** capable of:

- Predicting perceptual video quality (MOS)
- Evaluating campaign relevance
- Supporting scalable API-based inference
- Extensible multi-signal scoring system
