import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List

from src import config
from src.api import auth, schemas
from src.storage.s3_manager import S3Manager
from src.ml import dataset, drift, automl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "model": None,
    "scaler": None,
    "model_name": "unknown",
    "model_version": "none",
    "metrics": {},
    "reference_data": None,
    "s3_manager": None,
    "is_retraining": False,
    "retrain_count": 0
}

# CSV path for logging inference queries for drift analysis
INFERENCE_LOG_PATH = os.path.join(config.LOCAL_STORAGE_DIR, "inference_log.csv")

def ensure_initial_model(s3_manager: S3Manager):
    """
    Checks if a model exists in S3/local storage. If not, generates base data and trains one.
    """
    logger.info("Initializing baseline model check...")
    try:
        # Try loading metadata
        metadata = s3_manager.load_pkl("model_metadata.pkl")
        model = s3_manager.load_pkl("model.pkl")
        scaler = s3_manager.load_pkl("scaler.pkl")
        
        # We also need reference data to perform drift detection
        ref_data_key = "reference_data.csv"
        local_ref_path = os.path.join(config.LOCAL_STORAGE_DIR, "reference_data.csv")
        s3_manager.download_file(ref_data_key, local_ref_path)
        ref_df = pd.read_csv(local_ref_path)
        
        if metadata and model and scaler and not ref_df.empty:
            logger.info(f"Loaded existing model '{metadata['model_name']}' version '{metadata['version']}' from store.")
            app_state.update({
                "model": model,
                "scaler": scaler,
                "model_name": metadata["model_name"],
                "model_version": metadata["version"],
                "metrics": metadata.get("metrics", {}),
                "reference_data": ref_df
            })
            return
    except Exception as e:
        logger.info(f"Could not load existing model from storage ({e}). Will train a new baseline.")

    # Train a new baseline model
    logger.info("No valid baseline model found. Generating synthetic training data...")
    ref_df = dataset.generate_base_dataset(n_samples=2000, seed=42)
    
    # Save training dataset locally and upload
    local_ref_path = os.path.join(config.LOCAL_STORAGE_DIR, "reference_data.csv")
    os.makedirs(os.path.dirname(local_ref_path), exist_ok=True)
    ref_df.to_csv(local_ref_path, index=False)
    s3_manager.upload_file(local_ref_path, "reference_data.csv")
    
    # Train AutoML
    trainer = automl.AutoMLTrainer()
    results = trainer.train(ref_df)
    
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metadata = {
        "model_name": results["model_name"],
        "version": version,
        "metrics": results["metrics"],
        "training_samples": results["training_samples"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Save artifacts
    s3_manager.save_pkl(results["model_object"], "model.pkl")
    s3_manager.save_pkl(results["scaler_object"], "scaler.pkl")
    s3_manager.save_pkl(metadata, "model_metadata.pkl")
    
    # Update app state
    app_state.update({
        "model": results["model_object"],
        "scaler": results["scaler_object"],
        "model_name": results["model_name"],
        "model_version": version,
        "metrics": results["metrics"],
        "reference_data": ref_df
    })
    logger.info(f"Successfully trained and uploaded baseline model: {results['model_name']} (v {version})")


def run_retraining_job():
    """
    Retrains the model in the background, combining the base reference dataset
    with any logged drift/feedback data if available.
    """
    app_state["is_retraining"] = True
    s3_manager = app_state["s3_manager"]
    try:
        logger.info("Background retraining job started.")
        # Load baseline training data
        ref_df = app_state["reference_data"].copy()
        
        # Load accumulated feedback data if it exists to enhance retraining
        feedback_path = os.path.join(config.LOCAL_STORAGE_DIR, "feedback_log.csv")
        if os.path.exists(feedback_path):
            try:
                feedback_df = pd.read_csv(feedback_path)
                # Ensure it matches schema columns
                cols = list(ref_df.columns)
                feedback_df = feedback_df[cols]
                # Combine datasets
                ref_df = pd.concat([ref_df, feedback_df], ignore_index=True)
                logger.info(f"Augmented training data with {len(feedback_df)} samples of feedback data.")
            except Exception as e:
                logger.warning(f"Could not load feedback data during retraining: {e}")
        
        # Trigger AutoML training
        trainer = automl.AutoMLTrainer()
        results = trainer.train(ref_df)
        
        # Save artifacts with new version timestamp
        new_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_name": results["model_name"],
            "version": new_version,
            "metrics": results["metrics"],
            "training_samples": results["training_samples"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        s3_manager.save_pkl(results["model_object"], "model.pkl")
        s3_manager.save_pkl(results["scaler_object"], "scaler.pkl")
        s3_manager.save_pkl(metadata, "model_metadata.pkl")
        
        # Update active model in memory
        app_state.update({
            "model": results["model_object"],
            "scaler": results["scaler_object"],
            "model_name": results["model_name"],
            "model_version": new_version,
            "metrics": results["metrics"]
        })
        app_state["retrain_count"] += 1
        logger.info(f"Background retraining completed successfully. New active model: {results['model_name']} (v {new_version})")
        
    except Exception as e:
        logger.error(f"Error occurred during background retraining: {e}", exc_info=True)
    finally:
        app_state["is_retraining"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize S3Manager and Model
    s3_manager = S3Manager()
    app_state["s3_manager"] = s3_manager
    ensure_initial_model(s3_manager)
    yield
    # Shutdown: Clean up if needed
    logger.info("Application shutting down.")

app = FastAPI(
    title="AutoML Self-Healing Pipeline",
    description="A secure MLOps pipeline featuring AutoML tuning, drift detection, and self-healing mechanisms.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/token", response_model=schemas.TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates username/password and issues a JWT token.
    """
    if form_data.username != config.API_USERNAME or form_data.password != config.API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict(request: schemas.PredictionRequest, current_user: str = Depends(auth.get_current_user)):
    """
    Executes feature preprocessing and model prediction. Logs inputs for drift analysis.
    """
    if app_state["model"] is None or app_state["scaler"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized/loaded."
        )
        
    try:
        # Convert request schemas to DataFrame
        samples_dict = [sample.model_dump() for sample in request.samples]
        df_inference = pd.DataFrame(samples_dict)
        
        # Log to file for drift analysis (add timestamp)
        df_log = df_inference.copy()
        df_log["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        os.makedirs(os.path.dirname(INFERENCE_LOG_PATH), exist_ok=True)
        # Append mode or write fresh
        if not os.path.exists(INFERENCE_LOG_PATH):
            df_log.to_csv(INFERENCE_LOG_PATH, index=False)
        else:
            df_log.to_csv(INFERENCE_LOG_PATH, mode='a', header=False, index=False)
            
        # Preprocess features (scaler expects columns in specific order)
        cols_continuous = ["age", "income", "credit_score", "debt_to_income", "employment_years"]
        cols_categorical = ["prior_defaults"]
        
        X_cont = df_inference[cols_continuous].values
        X_cat = df_inference[cols_categorical].values
        
        X_cont_scaled = app_state["scaler"].transform(X_cont)
        X_processed = np.hstack([X_cont_scaled, X_cat])
        
        # Predict
        preds = app_state["model"].predict(X_processed)
        
        # Predict Probabilities
        if hasattr(app_state["model"], "predict_proba"):
            probs = app_state["model"].predict_proba(X_processed)[:, 1]
        else:
            # Fallback for models without predict_proba
            probs = preds.astype(float)
            
        outputs = []
        for pred, prob in zip(preds, probs):
            outputs.append(schemas.PredictionOutput(prediction=int(pred), probability=float(prob)))
            
        return schemas.PredictionResponse(
            predictions=outputs,
            model_name=app_state["model_name"],
            model_version=app_state["model_version"]
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process prediction: {str(e)}"
        )


@app.get("/drift-report")
async def get_drift_report(current_user: str = Depends(auth.get_current_user)):
    """
    Computes data drift statistics (KS-test and PSI) comparing logged inference requests
    against the reference training data.
    """
    if app_state["reference_data"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference dataset is not available."
        )
        
    if not os.path.exists(INFERENCE_LOG_PATH):
        return {
            "drift_detected": False,
            "message": "No inference data logged yet. Execute predictions first.",
            "average_psi": 0.0,
            "drift_fraction": 0.0,
            "drifted_features": [],
            "feature_reports": {}
        }
        
    try:
        # Load logged inference data
        df_inference = pd.read_csv(INFERENCE_LOG_PATH)
        # Drop timestamp
        if "timestamp" in df_inference.columns:
            df_inference = df_inference.drop(columns=["timestamp"])
            
        # Calculate drift
        detector = drift.DriftDetector(app_state["reference_data"])
        report = detector.check_drift(df_inference)
        return report
    except Exception as e:
        logger.error(f"Error computing drift report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate drift report: {str(e)}"
        )


@app.post("/feedback", response_model=schemas.FeedbackResponse)
async def submit_feedback(
    request: schemas.FeedbackRequest, 
    background_tasks: BackgroundTasks, 
    current_user: str = Depends(auth.get_current_user)
):
    """
    Receives ground truth labels for recent inputs to evaluate active model performance.
    Triggers background retraining (self-healing) if accuracy/F1 drops below threshold
    or if severe data drift is detected.
    """
    if app_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
        
    try:
        samples_dict = [sample.model_dump() for sample in request.samples]
        df_feedback = pd.DataFrame(samples_dict)
        
        # Save feedback data to local feedback store (to augment retraining later)
        feedback_path = os.path.join(config.LOCAL_STORAGE_DIR, "feedback_log.csv")
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        if not os.path.exists(feedback_path):
            df_feedback.to_csv(feedback_path, index=False)
        else:
            df_feedback.to_csv(feedback_path, mode='a', header=False, index=False)
            
        # Preprocess features
        cols_continuous = ["age", "income", "credit_score", "debt_to_income", "employment_years"]
        cols_categorical = ["prior_defaults"]
        
        X_cont = df_feedback[cols_continuous].values
        X_cat = df_feedback[cols_categorical].values
        
        X_cont_scaled = app_state["scaler"].transform(X_cont)
        X_processed = np.hstack([X_cont_scaled, X_cat])
        
        y_true = df_feedback["target"].values
        
        # Predict
        y_pred = app_state["model"].predict(X_processed)
        
        # Calculate performance metrics
        from sklearn.metrics import f1_score as calc_f1, accuracy_score as calc_acc
        f1 = float(calc_f1(y_true, y_pred, zero_division=0))
        acc = float(calc_acc(y_true, y_pred))
        
        # Check drift status using the new feedback features
        df_features = df_feedback.drop(columns=["target"])
        detector = drift.DriftDetector(app_state["reference_data"])
        drift_report = detector.check_drift(df_features)
        
        drift_detected = drift_report["overall_drift"]
        degradation_detected = f1 < config.PERFORMANCE_THRESHOLD_F1
        
        retraining_triggered = False
        message = "Model performing within normal parameters."
        
        # Self-healing logic
        if degradation_detected or drift_detected:
            if not app_state["is_retraining"]:
                background_tasks.add_task(run_retraining_job)
                retraining_triggered = True
                message = f"Degradation or drift detected. Self-healing retraining triggered in the background. " \
                          f"Reason: [Degradation: {degradation_detected} (F1: {f1:.4f} < {config.PERFORMANCE_THRESHOLD_F1}), " \
                          f"Drift: {drift_detected} (Avg PSI: {drift_report['average_psi']:.4f})]"
            else:
                message = "Degradation or drift detected, but retraining is already in progress."
                
        return schemas.FeedbackResponse(
            performance_f1=f1,
            performance_accuracy=acc,
            threshold_f1=config.PERFORMANCE_THRESHOLD_F1,
            degradation_detected=degradation_detected,
            drift_detected=drift_detected,
            retraining_triggered=retraining_triggered,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Feedback processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(e)}"
        )


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, current_user: str = Depends(auth.get_current_user)):
    """
    Manually triggers the AutoML retraining pipeline in the background.
    """
    if app_state["is_retraining"]:
        return {"status": "ignored", "message": "Retraining is already in progress."}
        
    background_tasks.add_task(run_retraining_job)
    return {"status": "triggered", "message": "AutoML retraining job has been triggered in the background."}


@app.get("/health")
async def health():
    """
    Public health check endpoint displaying service health and model stats.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": app_state["model"] is not None,
        "model_name": app_state["model_name"],
        "model_version": app_state["model_version"],
        "model_metrics_f1": app_state["metrics"].get("f1_score"),
        "model_metrics_accuracy": app_state["metrics"].get("accuracy"),
        "is_retraining": app_state["is_retraining"],
        "background_retrain_count": app_state["retrain_count"]
    }


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Serves a beautiful, interactive frontend MLOps dashboard.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌌 Antigravity AutoML MLOps Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-base: #060713;
            --bg-surface: rgba(17, 19, 40, 0.75);
            --bg-card: rgba(27, 30, 62, 0.5);
            --border-glow: rgba(139, 92, 246, 0.2);
            --border-hover: rgba(139, 92, 246, 0.45);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --primary: #8b5cf6;
            --primary-gradient: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            scroll-behavior: smooth;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
        }

        /* Login Screen Overlay */
        #login-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(6, 7, 19, 0.95);
            backdrop-filter: blur(12px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.5s ease;
        }

        .login-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-glow);
            padding: 2.5rem;
            border-radius: 1.5rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 20px rgba(139, 92, 246, 0.1);
            text-align: center;
        }

        .login-card h2 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #a78bfa, #8b5cf6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .login-card p {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }

        .input-group {
            text-align: left;
            margin-bottom: 1.25rem;
        }

        .input-group label {
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .input-group input {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(6, 7, 19, 0.6);
            border: 1px solid var(--border-glow);
            border-radius: 0.75rem;
            color: white;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s;
        }

        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 8px rgba(139, 92, 246, 0.3);
        }

        .btn {
            display: inline-block;
            width: 100%;
            padding: 0.85rem;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 0.75rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
        }

        .error-message {
            color: var(--danger);
            font-size: 0.85rem;
            margin-top: 1rem;
            display: none;
        }

        /* Dashboard Container */
        .dashboard-container {
            display: none;
            flex-direction: column;
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            gap: 2rem;
        }

        /* Navbar Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-glow);
            padding-bottom: 1.5rem;
        }

        .logo-section h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #c084fc, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-section p {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        .logout-btn {
            background: transparent;
            border: 1px solid var(--border-glow);
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s;
        }

        .logout-btn:hover {
            border-color: var(--danger);
            color: var(--danger);
            background: rgba(239, 68, 68, 0.05);
        }

        /* Overview Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-glow);
            border-radius: 1.25rem;
            padding: 1.5rem;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            border-color: var(--border-hover);
            transform: translateY(-2px);
        }

        .card-header {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-value {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .card-subtext {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* Status Badge Colors */
        .status-pill {
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .status-safe {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-drifted {
            background: rgba(239, 68, 68, 0.15);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
            animation: pulse-border 2s infinite;
        }

        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .dot-pulse {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: currentColor;
            display: inline-block;
            animation: pulse-opacity 1.5s infinite;
        }

        @keyframes pulse-opacity {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        /* Two Column Layout */
        .main-layout {
            display: grid;
            grid-template-columns: 1.6fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
        }

        /* Sections */
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Tables */
        .table-container {
            width: 100%;
            overflow-x: auto;
            margin-top: 1rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        th {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-glow);
            color: var(--text-secondary);
            font-weight: 500;
        }

        td {
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        tr:last-child td {
            border-bottom: none;
        }

        /* Charts panel */
        .chart-box {
            height: 300px;
            position: relative;
            margin-top: 1rem;
        }

        /* Form Inputs & Playgrounds */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .playground-form .input-group {
            margin-bottom: 0.85rem;
        }

        .playground-form input, .playground-form select {
            width: 100%;
            padding: 0.6rem 0.85rem;
            background: rgba(6, 7, 19, 0.4);
            border: 1px solid var(--border-glow);
            border-radius: 0.6rem;
            color: white;
            font-size: 0.9rem;
            outline: none;
        }

        .playground-form input:focus {
            border-color: var(--primary);
        }

        .prediction-result {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 0.75rem;
            background: rgba(6, 7, 19, 0.5);
            border: 1px dashed var(--border-glow);
            text-align: center;
            display: none;
        }

        .prediction-header {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .prediction-label {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .probability-bar-container {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 99px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .probability-bar {
            height: 100%;
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 99px;
        }

        .high-risk {
            color: var(--danger);
        }

        .low-risk {
            color: var(--success);
        }

        /* Controls Panel */
        .controls-card {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .control-row {
            display: flex;
            gap: 0.75rem;
        }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--primary);
            color: var(--text-primary);
        }

        .btn-outline:hover {
            background: rgba(139, 92, 246, 0.1);
            border-color: var(--primary-hover);
        }

        /* Retrain Overlay loader */
        .retrain-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(6, 7, 19, 0.85);
            backdrop-filter: blur(6px);
            z-index: 10;
            border-radius: 1.25rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.4s ease;
        }

        .retrain-overlay.active {
            opacity: 1;
            pointer-events: auto;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3.5px solid rgba(139, 92, 246, 0.25);
            border-top: 3.5px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-glow);
            padding: 1rem 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            z-index: 2000;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transform: translateY(150%);
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .toast.active {
            transform: translateY(0);
        }
    </style>
</head>
<body>

    <!-- Login Dialog Overlay -->
    <div id="login-overlay">
        <div class="login-card">
            <h2>🌌 MLOps Console</h2>
            <p>Sign in with administrative credentials</p>
            
            <form id="login-form">
                <div class="input-group">
                    <label>Username</label>
                    <input type="text" id="username" value="admin" required autocomplete="username">
                </div>
                <div class="input-group">
                    <label>Password</label>
                    <input type="password" id="password" value="admin123" required autocomplete="current-password">
                </div>
                <button type="submit" class="btn">Authenticate</button>
            </form>
            <div id="login-error" class="error-message">Invalid username or password.</div>
        </div>
    </div>

    <!-- Main Dashboard -->
    <div class="dashboard-container" id="dashboard">
        <header>
            <div class="logo-section">
                <h1>AutoML MLOps Pipeline</h1>
                <p>Intelligent Self-Healing Retraining & Continuous Drift Monitor</p>
            </div>
            <button onclick="logout()" class="logout-btn">Sign Out</button>
        </header>

        <!-- Top Overview metrics -->
        <div class="metrics-grid">
            <!-- Metric Card 1: Active Model -->
            <div class="card" id="card-active-model">
                <div class="card-header">
                    <span>Active Production Model</span>
                    <span class="status-pill status-safe" id="model-loaded-badge">
                        <span class="dot-pulse"></span> Loaded
                    </span>
                </div>
                <div class="card-value" id="val-model-name">Loading...</div>
                <div class="card-subtext">Version: <span id="val-model-version">N/A</span></div>
            </div>

            <!-- Metric Card 2: Performance F1 -->
            <div class="card">
                <div class="card-header">
                    <span>Validation Accuracy & F1</span>
                </div>
                <div class="card-value" id="val-model-score">0.00%</div>
                <div class="card-subtext">Baseline F1 Target: &gt; 80.0%</div>
            </div>

            <!-- Metric Card 3: Drift Alarm Status -->
            <div class="card" id="card-drift-status">
                <div class="card-header">
                    <span>Drift Detection Status</span>
                </div>
                <div class="card-value" id="val-drift-status">Loading...</div>
                <div class="card-subtext">Avg Population Stability Index: <span id="val-avg-psi">0.00</span></div>
            </div>

            <!-- Metric Card 4: System Retrains -->
            <div class="card">
                <div class="card-header">
                    <span>Self-Healing Retrains</span>
                </div>
                <div class="card-value" id="val-retrain-count">0</div>
                <div class="card-subtext">Active Tuning Tasks: <span id="val-is-retraining">No</span></div>
            </div>
        </div>

        <!-- Main Workspace Two Columns -->
        <div class="main-layout">
            <!-- Left Side: Data Drift Details & Charts -->
            <div class="column-left" style="display: flex; flex-direction: column; gap: 2rem;">
                <!-- Drift Analysis Table -->
                <div class="card" style="position: relative;">
                    <!-- Retrain Background Spinner Overlay -->
                    <div class="retrain-overlay" id="retrain-spinner">
                        <div class="spinner"></div>
                        <h3 style="font-weight: 600; color: white;">AutoML Retraining Process Active</h3>
                        <p style="color: var(--text-secondary); font-size: 0.85rem;">Searching hyperparameter space with Optuna...</p>
                    </div>

                    <div class="section-title">📊 Feature Drift Analysis (KS-Test & PSI)</div>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">
                        Compares inference datasets dynamically with reference training data.
                    </p>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Feature Name</th>
                                    <th>PSI Value</th>
                                    <th>PSI Status</th>
                                    <th>KS-Test p-value</th>
                                    <th>KS Status</th>
                                    <th>Overall Status</th>
                                </tr>
                            </thead>
                            <tbody id="drift-table-body">
                                <tr><td colspan="6" style="text-align: center;">Waiting for prediction queries...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Drift Distribution Chart -->
                <div class="card">
                    <div class="section-title">📈 Feature Shift Comparison (PSI Charts)</div>
                    <div class="chart-box">
                        <canvas id="psiChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Right Side: Playground & Self-Healing Controllers -->
            <div class="column-right" style="display: flex; flex-direction: column; gap: 2rem;">
                <!-- Prediction playground -->
                <div class="card">
                    <div class="section-title">🔮 Prediction Playground</div>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem;">
                        Submit features manually to calculate real-time borrower default risk.
                    </p>
                    
                    <form id="playground-form" class="playground-form">
                        <div class="form-grid">
                            <div class="input-group">
                                <label>Age</label>
                                <input type="number" id="play-age" min="18" max="100" value="38" required>
                            </div>
                            <div class="input-group">
                                <label>Income ($)</label>
                                <input type="number" id="play-income" min="0" value="60000" required>
                            </div>
                            <div class="input-group">
                                <label>Credit Score</label>
                                <input type="number" id="play-credit" min="300" max="850" value="680" required>
                            </div>
                            <div class="input-group">
                                <label>Debt-to-Income (%)</label>
                                <input type="number" id="play-dti" min="0" max="100" value="25" required>
                            </div>
                            <div class="input-group">
                                <label>Employment Years</label>
                                <input type="number" id="play-employment" min="0" value="5" step="0.5" required>
                            </div>
                            <div class="input-group">
                                <label>Prior Defaults</label>
                                <select id="play-defaults">
                                    <option value="0">0 (No)</option>
                                    <option value="1">1 (Yes)</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="btn" style="margin-top: 1rem;">Run Model Inference</button>
                    </form>

                    <!-- Prediction Result Overlay -->
                    <div class="prediction-result" id="play-result-box">
                        <div class="prediction-header">Model output</div>
                        <div class="prediction-label" id="play-prediction-label">Low Default Risk</div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">
                            Probability of Default: <span id="play-prob-value">12.5%</span>
                        </div>
                        <div class="probability-bar-container">
                            <div class="probability-bar" id="play-prob-bar"></div>
                        </div>
                    </div>
                </div>

                <!-- Self-Healing Control Actions -->
                <div class="card controls-card">
                    <div class="section-title">⚙️ MLOps Orchestrations & Actions</div>
                    <p style="color: var(--text-secondary); font-size: 0.85rem;">
                        Trigger self-healing cycles manually or inject mock drifting telemetry datasets.
                    </p>
                    
                    <button onclick="triggerManualRetrain()" class="btn btn-outline">
                        🔄 Trigger AutoML Retrain
                    </button>
                    
                    <button onclick="injectDriftedFeedback()" class="btn btn-outline" style="border-color: var(--warning); color: var(--warning);">
                        ⚡ Inject Drifted Feedback Batch
                    </button>
                    
                    <p style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                        Injecting drifted feedback sends features with shifted variables (low income, high debt) along with mismatching ground truth labels. This immediately triggers self-healing automated retraining.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Notification Toast -->
    <div id="toast" class="toast">
        <span id="toast-message">Success!</span>
    </div>

    <script>
        let psiChartInstance = null;

        document.getElementById("login-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const usernameInput = document.getElementById("username").value;
            const passwordInput = document.getElementById("password").value;
            const errorDiv = document.getElementById("login-error");
            
            errorDiv.style.display = "none";
            
            const formData = new URLSearchParams();
            formData.append("username", usernameInput);
            formData.append("password", passwordInput);
            
            try {
                const response = await fetch("/token", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem("mldashboard_token", data.access_token);
                    showDashboard();
                } else {
                    errorDiv.style.display = "block";
                }
            } catch (err) {
                console.error(err);
                errorDiv.innerText = "Connection error to FastAPI backend server.";
                errorDiv.style.display = "block";
            }
        });

        function showToast(message, isError = false) {
            const toast = document.getElementById("toast");
            const msgSpan = document.getElementById("toast-message");
            toast.style.borderColor = isError ? "var(--danger)" : "var(--border-glow)";
            msgSpan.style.color = isError ? "var(--danger)" : "var(--success)";
            msgSpan.innerText = message;
            toast.classList.add("active");
            setTimeout(() => {
                toast.classList.remove("active");
            }, 3000);
        }

        function getHeaders() {
            const token = localStorage.getItem("mldashboard_token");
            return {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            };
        }

        function checkAuth() {
            const token = localStorage.getItem("mldashboard_token");
            if (token) {
                showDashboard();
            } else {
                document.getElementById("login-overlay").style.display = "flex";
                document.getElementById("dashboard").style.display = "none";
            }
        }

        function logout() {
            localStorage.removeItem("mldashboard_token");
            document.getElementById("login-overlay").style.display = "flex";
            document.getElementById("dashboard").style.display = "none";
        }

        function showDashboard() {
            document.getElementById("login-overlay").style.display = "none";
            document.getElementById("dashboard").style.display = "flex";
            refreshDashboardData();
            // Start polling health status
            setInterval(pollSystemHealth, 3000);
        }

        async function pollSystemHealth() {
            try {
                const response = await fetch("/health");
                if (response.ok) {
                    const data = await response.json();
                    
                    // Retraining indicator
                    const isRetraining = data.is_retraining;
                    const overlay = document.getElementById("retrain-spinner");
                    if (isRetraining) {
                        overlay.classList.add("active");
                    } else {
                        if (overlay.classList.contains("active")) {
                            overlay.classList.remove("active");
                            showToast("Model retraining completed! Dynamic reload active.");
                            refreshDashboardData();
                        }
                    }
                    
                    document.getElementById("val-retrain-count").innerText = data.background_retrain_count;
                    document.getElementById("val-is-retraining").innerText = isRetraining ? "Yes" : "No";
                }
            } catch (err) {
                console.error("Health polling failed", err);
            }
        }

        async function refreshDashboardData() {
            try {
                // Fetch Health
                const healthRes = await fetch("/health");
                if (healthRes.ok) {
                    const h = await healthRes.json();
                    document.getElementById("val-model-name").innerText = h.model_name.replace("_", " ").toUpperCase();
                    document.getElementById("val-model-version").innerText = h.model_version;
                    
                    const f1 = h.model_metrics_f1 ? (h.model_metrics_f1 * 100).toFixed(2) : "0.00";
                    const acc = h.model_metrics_accuracy ? (h.model_metrics_accuracy * 100).toFixed(2) : "0.00";
                    document.getElementById("val-model-score").innerText = `${f1}% / ${acc}%`;
                }

                // Fetch Drift Report
                const driftRes = await fetch("/drift-report", {
                    method: "GET",
                    headers: getHeaders()
                });
                
                if (driftRes.status === 401) {
                    logout();
                    return;
                }

                if (driftRes.ok) {
                    const r = await driftRes.json();
                    
                    // Set drift card
                    const driftBadge = document.getElementById("val-drift-status");
                    const cardDrift = document.getElementById("card-drift-status");
                    document.getElementById("val-avg-psi").innerText = r.average_psi.toFixed(4);
                    
                    if (r.message !== undefined) {
                        // Empty logs state
                        driftBadge.innerText = "NO TELEMETRY";
                        driftBadge.className = "status-pill status-safe";
                        document.getElementById("drift-table-body").innerHTML = `
                            <tr>
                                <td colspan="6" style="text-align: center; color: var(--text-secondary);">
                                    Waiting for prediction queries. Perform prediction playground actions to generate drift analysis.
                                </td>
                            </tr>`;
                        return;
                    }

                    if (r.overall_drift) {
                        driftBadge.innerText = "DRIFT DETECTED";
                        driftBadge.className = "status-pill status-drifted";
                    } else {
                        driftBadge.innerText = "SAFE";
                        driftBadge.className = "status-pill status-safe";
                    }

                    // Render feature table
                    let tableHTML = "";
                    const reports = r.feature_reports;
                    const features = Object.keys(reports);
                    
                    const chartLabels = [];
                    const chartPSIValues = [];
                    
                    features.forEach(f => {
                        const rep = reports[f];
                        chartLabels.push(f);
                        chartPSIValues.push(rep.psi);
                        
                        const psiText = rep.psi.toFixed(4);
                        const psiStatus = rep.psi_drift ? 
                            `<span style="color: var(--danger); font-weight:600;">Drift (&ge;0.25)</span>` : 
                            `<span style="color: var(--success);">Stable</span>`;
                            
                        const ksText = rep.ks_p_value !== null ? rep.ks_p_value.toExponential(3) : "N/A";
                        const ksStatus = rep.ks_drift ? 
                            `<span style="color: var(--danger); font-weight:600;">Drift (&lt;0.05)</span>` : 
                            `<span style="color: var(--success);">Stable</span>`;
                            
                        const overall = rep.drift_detected ? 
                            `<span class="status-pill status-drifted" style="font-size:0.7rem; padding:0.1rem 0.4rem;">Drift</span>` : 
                            `<span class="status-pill status-safe" style="font-size:0.7rem; padding:0.1rem 0.4rem;">Stable</span>`;
                            
                        tableHTML += `
                            <tr>
                                <td style="font-weight: 500; color: #e5e7eb;">${f}</td>
                                <td>${psiText}</td>
                                <td>${psiStatus}</td>
                                <td>${ksText}</td>
                                <td>${ksStatus}</td>
                                <td>${overall}</td>
                            </tr>
                        `;
                    });
                    
                    document.getElementById("drift-table-body").innerHTML = tableHTML;
                    renderChart(chartLabels, chartPSIValues);
                }
            } catch (err) {
                console.error("Dashboard refresh error", err);
            }
        }

        function renderChart(labels, values) {
            const ctx = document.getElementById('psiChart').getContext('2d');
            
            if (psiChartInstance) {
                psiChartInstance.destroy();
            }
            
            psiChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Population Stability Index (PSI) per Feature',
                        data: values,
                        backgroundColor: values.map(v => v >= 0.25 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(139, 92, 246, 0.6)'),
                        borderColor: values.map(v => v >= 0.25 ? 'var(--danger)' : 'var(--primary)'),
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)'
                            },
                            ticks: {
                                color: '#9ca3af'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: '#9ca3af'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }

        // Playground predict form
        document.getElementById("playground-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            
            const payload = {
                samples: [{
                    age: parseFloat(document.getElementById("play-age").value),
                    income: parseFloat(document.getElementById("play-income").value),
                    credit_score: parseFloat(document.getElementById("play-credit").value),
                    debt_to_income: parseFloat(document.getElementById("play-dti").value),
                    employment_years: parseFloat(document.getElementById("play-employment").value),
                    prior_defaults: parseInt(document.getElementById("play-defaults").value)
                }]
            };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: getHeaders(),
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    const data = await response.json();
                    const prediction = data.predictions[0];
                    const prob = (prediction.probability * 100).toFixed(1);
                    
                    const box = document.getElementById("play-result-box");
                    const label = document.getElementById("play-prediction-label");
                    const probVal = document.getElementById("play-prob-value");
                    const probBar = document.getElementById("play-prob-bar");
                    
                    box.style.display = "block";
                    probVal.innerText = `${prob}%`;
                    probBar.style.width = `${prob}%`;
                    
                    if (prediction.prediction === 1) {
                        label.innerText = "REJECTED (High Default Risk)";
                        label.className = "prediction-label high-risk";
                        probBar.style.backgroundColor = "var(--danger)";
                        box.style.boxShadow = "0 0 15px rgba(239, 68, 68, 0.15)";
                    } else {
                        label.innerText = "APPROVED (Low Default Risk)";
                        label.className = "prediction-label low-risk";
                        probBar.style.backgroundColor = "var(--success)";
                        box.style.boxShadow = "0 0 15px rgba(16, 185, 129, 0.15)";
                    }
                    
                    refreshDashboardData();
                } else {
                    const err = await response.json();
                    showToast(err.detail || "Prediction request failed", true);
                }
            } catch (err) {
                showToast("Connection to backend server lost.", true);
            }
        });

        // Trigger manual retrain
        async function triggerManualRetrain() {
            try {
                const res = await fetch("/retrain", {
                    method: "POST",
                    headers: getHeaders()
                });
                
                if (res.ok) {
                    const data = await res.json();
                    showToast(data.message);
                    document.getElementById("retrain-spinner").classList.add("active");
                } else {
                    const err = await res.json();
                    showToast(err.detail || "Retrain request failed", true);
                }
            } catch (err) {
                showToast("Request failed", true);
            }
        }

        // Inject drifted feedback
        async function injectDriftedFeedback() {
            // Generate a drifted dataset samples
            const payload = {
                samples: []
            };
            const predictSamples = [];
            
            // Generate 20 low credit, high debt samples representing high default rate 
            // but label them mismatchingly to drop F1 accuracy of the current model
            for (let i = 0; i < 20; i++) {
                const sample = {
                    age: Math.random() * 20 + 25,
                    income: Math.random() * 10000 + 20000,
                    credit_score: Math.random() * 100 + 400, // Very low credit
                    debt_to_income: Math.random() * 30 + 55, // Very high debt
                    employment_years: Math.random() * 2,
                    prior_defaults: 1
                };
                predictSamples.push(sample);
                payload.samples.push({ ...sample, target: 0 }); // Mock label
            }

            try {
                // First simulate the live prediction telemetry to trigger drift logging
                await fetch("/predict", {
                    method: "POST",
                    headers: getHeaders(),
                    body: JSON.stringify({ samples: predictSamples })
                });

                // Then send the ground truth feedback to trigger performance evaluation & self-healing
                const res = await fetch("/feedback", {
                    method: "POST",
                    headers: getHeaders(),
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    const data = await res.json();
                    showToast(data.message);
                    if (data.retraining_triggered) {
                        document.getElementById("retrain-spinner").classList.add("active");
                    }
                    refreshDashboardData();
                } else {
                    const err = await res.json();
                    showToast(err.detail || "Feedback injection failed", true);
                }
            } catch (err) {
                showToast("Request failed", true);
            }
        }

        // Initialize auth check on load
        window.addEventListener("DOMContentLoaded", checkAuth);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
