# SahabaX AI

Machine learning system for predicting next-day rainfall in Australia using a complete end-to-end ML pipeline, benchmark evaluation, FastAPI backend, and interactive bilingual frontend.

---

## Kaggle Notebook

You can explore the full notebook version of the project on [Kaggle Notebook](https://www.kaggle.com/code/m0hamedramzy/rain-in-australia-prediction-with-ml-pipeline?utm_source=chatgpt.com)

The notebook includes:

* Full exploratory data analysis (EDA)
* Data preprocessing
* Feature engineering
* Benchmark comparison
* Model training
* Evaluation metrics
* Visualizations
* Final model selection

---

## Overview

| Item                  | Details                              |
| --------------------- | ------------------------------------ |
| Project Name          | SahabaX AI                           |
| Task                  | Rainfall Prediction                  |
| Dataset               | Rain in Australia (`weatherAUS.csv`) |
| Best Production Model | XGBoost                              |
| API Version           | 2.0.0                                |
| Backend               | FastAPI                              |
| Frontend              | HTML · CSS · Vanilla JavaScript      |
| ML Libraries          | scikit-learn · XGBoost · LightGBM    |
| Imbalance Handling    | SMOTE                                |
| Feature Selection     | SelectKBest (Chi-Square)             |

---

## Models Evaluated

The system benchmarks and compares nine machine learning algorithms:

1. Logistic Regression
2. SVM
3. Naive Bayes
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM
9. MLP Neural Network

Models are ranked using weighted F1-Score to better handle class imbalance.

---

## Project Structure

```text
.
├── api.py
├── train.py
├── benchmark.py
├── predict.py
├── requirements.txt
│
├── data/
│   └── Rain in Australia/
│       └── weatherAUS.csv
│
├── models/
│   └── rain_model.pkl
│
├── outputs/
│   ├── benchmark_results.json
│   ├── best_model_confusion_matrix.png
│   └── model_comparison.png
│
├── frontend/
│   └── index.html
│
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessor.py
    ├── feature_engineering.py
    ├── models.py
    ├── trainer.py
    ├── evaluator.py
    └── model_io.py
```

---

# System Architecture

```text
Dataset
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
SMOTE Balancing
   ↓
MinMax Scaling
   ↓
Feature Selection
   ↓
Model Benchmarking
   ↓
Best Model Selection
   ↓
Training Pipeline
   ↓
Saved Model (.pkl)
   ↓
FastAPI Backend
   ↓
Frontend Prediction Interface
```

---

# Quick Start

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd SahabaX-AI
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Benchmark

This compares all models and ranks them automatically.

```bash
python benchmark.py
```

Generated outputs:

* `outputs/benchmark_results.json`
* `outputs/model_comparison.png`
* `outputs/best_model_confusion_matrix.png`

---

## 4. Train Best Model

```bash
python train.py
```

Or train a specific model:

```bash
python train.py --model "LightGBM"
```

The trained pipeline will be saved to:

```text
models/rain_model.pkl
```

---

## 5. Start API Server

```bash
python api.py
```

Or:

```bash
uvicorn api:app --reload --port 8000
```

API Documentation:

```text
http://localhost:8000/docs
```

---

## 6. Start Frontend

```bash
cd frontend
python -m http.server 8080
```

Frontend URL:

```text
http://localhost:8080
```

---

# Machine Learning Pipeline

| Stage             | Description                                          |
| ----------------- | ---------------------------------------------------- |
| Load Data         | Read dataset from CSV                                |
| Preprocessing     | Clean missing values and encode categorical features |
| Split Dataset     | Stratified train/test split                          |
| SMOTE             | Handle class imbalance                               |
| Scaling           | MinMax normalization                                 |
| Feature Selection | Select top 15 features using Chi-Square              |
| Training          | Train selected model                                 |
| Evaluation        | Accuracy, Precision, Recall, F1                      |
| Saving            | Export trained pipeline                              |

---

# Selected Features

The pipeline automatically selects the most important features using:

```text
SelectKBest(chi2)
```

Default selected feature count:

```text
15 Features
```

---

# API Reference

Base URL:

```text
http://localhost:8000
```

---

## GET `/`

Returns API status and available routes.

### Response

```json
{
  "message": "Rain Prediction API is running",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

---

## GET `/health`

Returns model load status.

### Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "XGBoost"
}
```

---

## POST `/predict`

Predicts rainfall probability using weather observations.

### Request Example

```json
{
  "Humidity3pm": 55.0,
  "Humidity9am": 80.0,
  "Rainfall": 5.0,
  "Sunshine": 8.0,
  "Cloud3pm": 4.0,
  "Cloud9am": 5.0,
  "Pressure9am": 1012.0,
  "WindGustSpeed": 40.0,
  "Temp3pm": 22.0,
  "RainToday": 0
}
```

### Response Example

```json
{
  "prediction": "Yes",
  "probability": 0.72,
  "rain": true
}
```

---

# Frontend Features

The frontend interface includes:

* Arabic and English language support
* RTL/LTR switching
* Light and dark mode
* Responsive layout
* Weather prediction form
* Input validation
* Prediction probability visualization
* AI weather analysis panel
* Mobile support

---

# Configuration

Main configuration file:

```text
src/config.py
```

Configurable settings include:

* Dataset paths
* Model paths
* Random seed
* Number of selected features
* SMOTE strategy
* Hyperparameters
* Wind direction mappings

---

# Model Details

## XGBoost

Production model selected from benchmark ranking.

### Parameters

```text
Estimators    : 100
Max Depth     : 10
Learning Rate : 0.1
Subsample     : 0.8
```

---

## LightGBM

```text
Estimators    : 100
Learning Rate : 0.1
Tree Depth    : Unlimited
```

---

## Random Forest

```text
Estimators : 300
Max Depth  : 15
Class Weight : Balanced
```

---

## MLP Neural Network

```text
Architecture:
128 -> 64 -> 32
```

Activation:

```text
ReLU
```

Optimizer:

```text
Adam
```

---

# Benchmark Output

Running:

```bash
python benchmark.py
```

Produces:

| File                            | Description                |
| ------------------------------- | -------------------------- |
| benchmark_results.json          | Model ranking and metrics  |
| best_model_confusion_matrix.png | Confusion matrix           |
| model_comparison.png            | Benchmark comparison chart |

---

# Dependencies

| Package          | Purpose              |
| ---------------- | -------------------- |
| fastapi          | Backend framework    |
| uvicorn          | ASGI server          |
| pandas           | Data analysis        |
| numpy            | Numerical operations |
| scikit-learn     | ML utilities         |
| xgboost          | XGBoost model        |
| lightgbm         | LightGBM model       |
| imbalanced-learn | SMOTE                |
| matplotlib       | Visualization        |
| seaborn          | Statistical plotting |
| joblib           | Model serialization  |

---

# Troubleshooting

## Model Not Loaded

Train the model first:

```bash
python train.py
```

---

## API Connection Error

Ensure the API server is running on:

```text
http://localhost:8000
```

---

## CORS Issues

The API currently allows all origins:

```python
allow_origins=["*"]
```

---

## Feature Mismatch

The API expects the exact fields defined in:

```text
WeatherInput
```

---

# Extending the Project

## Add New Models

Edit:

```text
src/models.py
```

Then rerun:

```bash
python benchmark.py
```

---

## Customize Pipeline

Edit:

```text
src/trainer.py
```

or:

```text
src/config.py
```

Possible customizations:

* More selected features
* Different SMOTE strategy
* Hyperparameter tuning
* Alternative preprocessing techniques

---

# License

This project uses the Rain in Australia dataset.

Please ensure compliance with the dataset license and usage terms before redistribution or commercial use.
