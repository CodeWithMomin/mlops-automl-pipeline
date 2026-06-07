import os
import shutil
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Mock environment variables before importing config/app
os.environ["LOCAL_STORAGE_DIR"] = "./test_store"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["S3_ENDPOINT_URL"] = ""  # Force local fallback

from src import config
from src.ml import dataset, drift, automl
from src.api.main import app, INFERENCE_LOG_PATH

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    # Setup test directories
    os.makedirs(config.LOCAL_STORAGE_DIR, exist_ok=True)
    yield
    # Teardown
    if os.path.exists(config.LOCAL_STORAGE_DIR):
        shutil.rmtree(config.LOCAL_STORAGE_DIR)


def test_dataset_generation():
    df_base = dataset.generate_base_dataset(n_samples=100, seed=42)
    assert len(df_base) == 100
    assert "target" in df_base.columns
    assert "credit_score" in df_base.columns
    
    df_drift = dataset.generate_drifted_dataset(n_samples=100, shift_type="covariate", seed=42)
    assert len(df_drift) == 100
    # Drifted credit score mean should be lower
    assert df_drift["credit_score"].mean() < df_base["credit_score"].mean()


def test_drift_detection():
    df_base = dataset.generate_base_dataset(n_samples=1500, seed=42)
    df_clean = dataset.generate_base_dataset(n_samples=800, seed=100)
    df_drifted = dataset.generate_drifted_dataset(n_samples=800, shift_type="covariate", seed=200)
    
    detector = drift.DriftDetector(df_base)
    
    # 1. Clean dataset should NOT trigger overall drift
    report_clean = detector.check_drift(df_clean)
    assert report_clean["overall_drift"] is False
    
    # 2. Drifted dataset SHOULD trigger overall drift
    report_drifted = detector.check_drift(df_drifted)
    assert report_drifted["overall_drift"] is True
    assert len(report_drifted["drifted_features"]) > 0


def test_automl_training():
    df = dataset.generate_base_dataset(n_samples=200, seed=42)
    # Using 2 trials for faster testing
    trainer = automl.AutoMLTrainer(n_trials=2)
    results = trainer.train(df)
    
    assert "model_name" in results
    assert "model_object" in results
    assert "scaler_object" in results
    assert "metrics" in results
    assert "f1_score" in results["metrics"]


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_auth(client):
    # Attempt prediction without credentials
    response = client.post("/predict", json={"samples": []})
    assert response.status_code == 401
    
    # Attempt token retrieval with bad credentials
    response = client.post("/token", data={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    
    # Valid credentials
    response = client.post("/token", data={"username": config.API_USERNAME, "password": config.API_PASSWORD})
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


def test_api_prediction_and_drift_flow(client):
    # 1. Get Token
    response = client.post("/token", data={"username": config.API_USERNAME, "password": config.API_PASSWORD})
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Submit Predictions
    sample_payload = {
        "samples": [
            {
                "age": 30.0,
                "income": 50000.0,
                "credit_score": 700.0,
                "debt_to_income": 20.0,
                "employment_years": 3.0,
                "prior_defaults": 0
            }
        ]
    }
    
    # Reset/clean logged inference before prediction
    if os.path.exists(INFERENCE_LOG_PATH):
        os.remove(INFERENCE_LOG_PATH)
        
    response = client.post("/predict", json=sample_payload, headers=headers)
    assert response.status_code == 200
    pred_data = response.json()
    assert len(pred_data["predictions"]) == 1
    assert "prediction" in pred_data["predictions"][0]
    assert "probability" in pred_data["predictions"][0]
    
    # Verify inference is logged
    assert os.path.exists(INFERENCE_LOG_PATH)
    
    # 3. Retrieve Drift Report
    response = client.get("/drift-report", headers=headers)
    assert response.status_code == 200
    drift_data = response.json()
    assert "overall_drift" in drift_data
    assert "feature_reports" in drift_data


def test_api_feedback_and_retraining(client):
    # 1. Get Token
    response = client.post("/token", data={"username": config.API_USERNAME, "password": config.API_PASSWORD})
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Send feedback data
    feedback_payload = {
        "samples": [
            {
                "age": 45.0,
                "income": 30000.0,
                "credit_score": 550.0,
                "debt_to_income": 45.0,
                "employment_years": 1.0,
                "prior_defaults": 1,
                "target": 0  # Likely default, but let's say target=0 to cause performance issues if model predicts 1
            }
        ]
    }
    
    # Check feedback endpoint
    response = client.post("/feedback", json=feedback_payload, headers=headers)
    assert response.status_code == 200
    feedback_data = response.json()
    assert "performance_f1" in feedback_data
    assert "degradation_detected" in feedback_data
    
    # 3. Manually trigger retraining
    response = client.post("/retrain", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] in ["triggered", "ignored"]
