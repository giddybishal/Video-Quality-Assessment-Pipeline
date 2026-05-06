# Video Quality Assessment Pipeline

## 📌 Overview

This project implements an end-to-end machine learning pipeline to predict video quality (MOS score) from raw video input. It uses deep visual features extracted from a pretrained ResNet50 model and learns a regression model using XGBoost trained on the :contentReference[oaicite:0]{index=0}.

---

## 🧠 What I Built So Far

### 1. Video Feature Extraction
- Extract frames from videos at fixed intervals
- Use pretrained ResNet50 to extract 2048-dimensional embeddings
- Aggregate frame-level features into a single video representation
- Save features as `.npy` files for reuse

### 2. Dataset Preparation
- Match extracted video features with MOS labels from CSV
- Build training dataset:
  - `X` = 2048-dim video embeddings  
  - `y` = MOS scores  

### 3. Model Training
- Train XGBoost regression model on extracted features
- Apply feature normalization (StandardScaler)
- Evaluate using:
  - MSE
  - R² Score
  - Spearman Correlation (ranking quality)

### 4. Inference Pipeline
- Built local script to predict MOS for new videos
- Input: raw video file  
- Output: predicted quality score  

---

## 📊 Results
- R² Score: ~0.50  
- Spearman Correlation: ~0.68  
- RMSE: ~0.26 MOS  

The model shows strong ability to rank video quality in alignment with human perception.

---

## 🚧 Blockers Faced
- Mapping video features correctly to MOS labels
- Handling large-scale video processing locally
- Understanding and interpreting ResNet feature outputs
- Ensuring consistent preprocessing between training and inference pipelines

---

## 🔭 Next Steps

### 1. Improve Model Performance
- Better frame aggregation (temporal-aware features)
- Experiment with stronger encoders (e.g., CLIP or video models)
- Hyperparameter tuning for XGBoost

### 2. Extend Scoring System
- Add semantic/agenda matching layer for campaign relevance
- Combine visual quality + semantic relevance into a unified scoring system

### 3. Architecture Refactor
- Move toward Hexagonal Architecture for modularity:
  - Domain: scoring logic
  - Application: pipeline orchestration
  - Ports: feature extraction, model inference
  - Adapters: ML models, dataset loaders, API layer

---