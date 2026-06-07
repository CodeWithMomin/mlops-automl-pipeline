import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Security Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-in-production-123456")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Default admin credentials for API access
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")  # In production, use hashed passwords or real DB

# S3 / Storage Settings
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-bucket")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")  # Default to local MinIO

# Fallback Local Storage
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(BASE_DIR / "data_store"))

# Thresholds
DRIFT_THRESHOLD_PSI = float(os.getenv("DRIFT_THRESHOLD_PSI", "0.25"))
DRIFT_THRESHOLD_KS = float(os.getenv("DRIFT_THRESHOLD_KS", "0.05"))
PERFORMANCE_THRESHOLD_F1 = float(os.getenv("PERFORMANCE_THRESHOLD_F1", "0.80"))

# AutoML Settings
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "5"))
