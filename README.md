# 🌌 AutoML MLOps Pipeline with Self-Healing & Drift Monitoring

A production-grade, secure, and containerized MLOps platform featuring an **AutoML Engine**, **Continuous Data Drift Detection**, **Autonomous Self-Healing Retraining**, and a **Glassmorphic Dark-themed Web Dashboard UI**. 

This system dynamically evaluates multiple model architectures (Random Forest, XGBoost, LightGBM), optimizes their hyperparameters using **Optuna**, stores model artifacts in AWS S3 (or local MinIO), secure predictions with **JWT auth**, and dynamically updates the production model when drift or performance degradation is detected.

---

## 🌟 Key Features

1. **AutoML Engine**: Simulataneously evaluates Random Forest, XGBoost, and LightGBM models. Uses **Optuna** to optimize hyperparameters based on validation F1-score.
2. **Dynamic Data Drift Monitor**: Computes **Kolmogorov-Smirnov (KS-test)** p-values for continuous inputs and **Population Stability Index (PSI)** to catch covariate distribution shifts.
3. **Autonomous Self-Healing**: Automatically triggers background retraining when F1 score degrades below threshold (80.0%) or data drift is detected on feedback labels. Dynamic, zero-downtime hot-swapping loads the new model into memory.
4. **Interactive Dashboard**: A custom dark-mode web dashboard serving real-time model metrics, interactive prediction playground, live drift reporting, charts (Chart.js), and retraining trigger controls.
5. **Security & Data Integrity**: JWT-based authentication protects predictions, drift reports, and feedback API routes. Includes secure credential and secrets handling.
6. **Multi-Container Docker Orchestration**: Bundles the application inside an optimized Dockerfile and a multi-container `docker-compose.yml` including Local MinIO (S3 clone) and automated bucket creation.
7. **CI/CD Integration**: Configured with a GitHub Actions workflow (`deploy.yml`) containing steps to lint, run unit tests, build Docker images, and outline deployment steps to AWS EC2.

---

## 📂 Project Directory Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD Pipeline
├── src/
│   ├── api/
│   │   ├── auth.py             # JWT Creation and Dependency Checking
│   │   ├── main.py             # FastAPI App, Endpoints, Dashboard UI, Self-Healing loop
│   │   └── schemas.py          # Pydantic Schemas for Requests/Responses
│   ├── ml/
│   │   ├── automl.py           # Optuna AutoML Tuning & Evaluation
│   │   ├── dataset.py          # Synthetic Data Generator (Clean & Drifted)
│   │   └── drift.py            # KS-test & PSI Drift Calculators
│   ├── storage/
│   │   └── s3_manager.py       # S3 wrapper (boto3) with Local Folder Fallback
│   └── config.py               # Config loader for env variables & thresholds
├── tests/
│   └── test_mlops.py           # Pytest unit tests for APIs, ML, Drift & Auth
├── .env.example                # Template for environment settings
├── .gitignore                  # Git exclude configurations
├── Dockerfile                  # Production stage Docker setup
├── docker-compose.yml          # FastAPI + MinIO + Bucket Provisioner cluster
├── requirements.txt            # Python application library dependencies
└── README.md                   # Setup and execution guide (This file)
```

---

## 🚀 Getting Started

Follow these instructions to run the application on your local machine:

### Prerequisites
Make sure you have the following installed:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
* [Git](https://git-scm.com/)

---

### Step-by-Step Installation

#### 1. Copy or Clone the repository
Navigate into the root directory in your terminal:
```bash
cd "Untitled Folder"
```

#### 2. Create the configuration `.env` file
Copy the default environment variables:
```bash
# On Linux/macOS
cp .env.example .env

# On Windows (Command Prompt)
copy .env.example .env
```

#### 3. Start the Docker Stack
Build and spin up the multi-container stack:
```bash
docker compose up -d --build
```
*Note: This will download MinIO, compile the multi-stage FastAPI container, create internal Docker networks/volumes, and configure the `mlops-bucket` bucket automatically on startup.*

#### 4. Verify Services are Running
Check the status of your containers:
```bash
docker compose ps
```
You should see:
* `mlops_api` (Up and running on port `8000`)
* `mlops_minio` (Up and running on console port `9001` / API port `9000`)
* `mlops_minio_provisioner` (Completed - bucket creation successfully exited)

To view the server logs and witness baseline AutoML training:
```bash
docker compose logs -f api
```

---

## 🖥️ Using the Interface

### Web Dashboard
Open your web browser and go to:
👉 **[http://localhost:8000/](http://localhost:8000/)**

1. Enter the admin credentials:
   * **Username**: `admin`
   * **Password**: `admin123`
2. **Prediction Playground**: Submit values to generate loan default predictions (Approved vs Rejected) and inspect probability weights.
3. **Drift Dashboard**: After predicting or clicking the **Inject Drifted Feedback Batch** button, inspect the PSI levels and KS-test indicators.
4. **Self-Healing Retrainer**: Observe the background spinner overlay lock when retraining occurs. Once finished, the dashboard dynamically updates with the new model name and improved version metadata.

### API Swagger Documentation
Explore and call endpoints using the Swagger UI:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Running Local Unit Tests
If you want to run the python test suite locally (outside Docker), set up a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run pytest
PYTHONPATH=. pytest tests/
```

---

## 📊 MLOps Concepts Visualized

### Data Drift Metrics
* **Kolmogorov-Smirnov (KS-Test)**: Continuous features are evaluated; a p-value $< 0.05$ flags statistical distribution differences.
* **Population Stability Index (PSI)**:
  * $PSI < 0.1$: Distribution is stable.
  * $0.1 \le PSI < 0.25$: Minor shift occurred.
  * $PSI \ge 0.25$: Major drift detected.

### Self-Healing Retraining Loop
```
                                        +-------------------+
                                        |  Logged Telemetry |
                                        +---------+---------+
                                                  |
                                                  v
+------------------+                    +---------+---------+
|  Active Model    | <--[Hot-swapped]---+  AutoML Retraining| <--[Drift/Degradation Alarm]
+------------------+                    +---------+---------+
```
If users upload new labels through `/feedback` and the F1-score falls below `0.80` or features exhibit a severe PSI shift, the API triggers the retraining pipeline. It merges original data with feedback logs, runs Optuna-optimized training, uploads the fresh artifact to MinIO (S3), and swaps it in-memory.
