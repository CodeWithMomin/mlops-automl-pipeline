import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple
from src import config

# Set optuna log level to warning to prevent clean logs pollution
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

class AutoMLTrainer:
    def __init__(self, n_trials: int = None):
        self.n_trials = n_trials or config.OPTUNA_TRIALS
        self.scaler = StandardScaler()
        self.continuous_cols = ["age", "income", "credit_score", "debt_to_income", "employment_years"]
        self.categorical_cols = ["prior_defaults"]

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses features: scales continuous columns and concatenates categorical ones.
        """
        df_clean = df.copy()
        
        # Split features and target
        X_cont = df_clean[self.continuous_cols].values
        X_cat = df_clean[self.categorical_cols].values
        
        if is_training:
            X_cont_scaled = self.scaler.fit_transform(X_cont)
        else:
            X_cont_scaled = self.scaler.transform(X_cont)
            
        X_processed = np.hstack([X_cont_scaled, X_cat])
        
        y = df_clean["target"].values if "target" in df_clean.columns else None
        return X_processed, y

    def _optimize_rf(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], float]:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": 42,
                "n_jobs": -1
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, zero_division=0)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value

    def _optimize_xgb(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], float]:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, zero_division=0)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value

    def _optimize_lgb(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], float]:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 10, 60),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return f1_score(y_val, preds, zero_division=0)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the AutoML tuning process, trains the best model, and returns details.
        """
        logger.info(f"Starting AutoML Training on {len(df)} samples...")
        
        # Train-validation-test split (70% train, 15% validation, 15% test)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["target"])
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["target"])
        
        # Preprocess features
        X_train, y_train = self.preprocess_data(train_df, is_training=True)
        X_val, y_val = self.preprocess_data(val_df, is_training=False)
        X_test, y_test = self.preprocess_data(test_df, is_training=False)
        
        # Hyperparameter optimization for all models
        rf_params, rf_score = self._optimize_rf(X_train, y_train, X_val, y_val)
        xgb_params, xgb_score = self._optimize_xgb(X_train, y_train, X_val, y_val)
        lgb_params, lgb_score = self._optimize_lgb(X_train, y_train, X_val, y_val)
        
        logger.info(f"Tuning F1 Scores - Random Forest: {rf_score:.4f}, XGBoost: {xgb_score:.4f}, LightGBM: {lgb_score:.4f}")
        
        # Select best model based on validation F1 score
        best_score = max(rf_score, xgb_score, lgb_score)
        if best_score == rf_score:
            best_model_name = "random_forest"
            best_params = rf_params
            model_class = RandomForestClassifier
        elif best_score == xgb_score:
            best_model_name = "xgboost"
            best_params = xgb_params
            model_class = xgb.XGBClassifier
        else:
            best_model_name = "lightgbm"
            best_params = lgb_params
            model_class = lgb.LGBMClassifier
            
        logger.info(f"Winner model: {best_model_name} (F1 Score: {best_score:.4f})")
        
        # Retrain best model on combined train + validation set for better generalization
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        best_model = model_class(**best_params)
        best_model.fit(X_train_val, y_train_val)
        
        # Evaluate on hold-out Test Set
        test_preds = best_model.predict(X_test)
        
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
        
        metrics = {
            "f1_score": float(test_f1),
            "accuracy": float(test_acc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        
        logger.info(f"Test Set Evaluation - F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        
        return {
            "model_name": best_model_name,
            "model_object": best_model,
            "scaler_object": self.scaler,
            "hyperparameters": best_params,
            "metrics": metrics,
            "training_samples": len(df)
        }
