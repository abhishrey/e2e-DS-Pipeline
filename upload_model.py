import os
import joblib
import logging
import time
import requests
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import storage
from google.cloud import run_v2
from google.cloud import aiplatform
from google.auth import default
from google.auth.transport.requests import Request
from sklearn.metrics import f1_score
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Use environment variables for dynamic configuration
PROJECT_ID = 712745806180
REGION = "us-central1"
ENDPOINT_ID = 5712274068086980608
GCS_BUCKET = "anomaly-detection-knn"
VALIDATION_DATA_PATH = "training_data/combined_data_breaks.csv"
CLOUD_RUN_JOB_NAME = "predict-from-csv"  # Update with your Cloud Run Job name

MODEL_PATH = "/tmp/model.pkl"
GCS_MODEL_PATH_PREFIX = "models/"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow-tracking-server-712745806180.us-central1.run.app")  # Update this
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("AnomalyDetectionModel")

# Initialize Storage Client
storage_client = storage.Client()

# Step 1: Download the validation dataset once
validation_data_path = "/tmp/validation_data.csv"
bucket = storage_client.bucket(GCS_BUCKET)
blob = bucket.blob(VALIDATION_DATA_PATH)
blob.download_to_filename(validation_data_path)
logging.info("Validation dataset downloaded successfully.")

def get_latest_model_from_gcs():
    """Fetches the latest model file from GCS by modification timestamp."""
    logging.info("Fetching the latest model from GCS...")

    bucket = storage_client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix=GCS_MODEL_PATH_PREFIX))

    if not blobs:
        raise ValueError("No models found in GCS bucket.")

    latest_blob = max(blobs, key=lambda b: b.updated)
    logging.info(f"Latest model found: {latest_blob.name}")
    return latest_blob.name

def trigger_cloud_run_batch_prediction():
    """Triggers the Cloud Run Job for batch prediction and waits for it to finish."""
    logging.info(f"Triggering Cloud Run Job: {CLOUD_RUN_JOB_NAME}...")

    credentials, _ = default()
    credentials.refresh(Request())
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    # Cloud Run Jobs URL
    cloud_run_url = f"https://{REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{PROJECT_ID}/jobs/{CLOUD_RUN_JOB_NAME}:run"

    # Start the batch prediction job
    response = requests.post(cloud_run_url, headers=headers, json={})

    if response.status_code == 200:
        logging.info("Cloud Run Job started successfully.")
    else:
        logging.error(f"Failed to trigger Cloud Run Job: {response.text}")
        return None

    # Poll for job completion
    logging.info("Waiting for batch job to complete...")
    time.sleep(60) 

    return True 

def evaluate_new_model_on_same_data(model_path, validation_data_path):
    """Evaluates the new model on the same validation dataset."""
    model = joblib.load(model_path)
    validation_data = pd.read_csv(validation_data_path)

    X_val = validation_data.drop(columns=["BREAKS"])
    y_val = validation_data["BREAKS"]
    y_test_binary = (y_val != 0).astype(int)

    # Ensure feature order matches the model's expectations
    expected_feature_order = [
        'smd_0', 'smd_1', 'smd_2', 'smd_3', 'smd_4',
        'aoi_0', 'aoi_1', 'aoi_2', 'aoi_3', 'aoi_4',
        'ss_0', 'ss_1', 'ss_2', 'ss_3', 'ss_4',
        'cc_0', 'cc_1', 'Overall_processing_time', 'Tardiness'
    ]
    X_val = X_val[expected_feature_order]

    predictions = model.predict(X_val)
    predictions_binary = (predictions != 0).astype(int)

    return f1_score(y_test_binary, predictions_binary, average='binary')

def evaluate_deployed_model_on_same_data(validation_data_path):
    """Evaluates the deployed model using batch predictions on the same validation dataset."""
    predictions_blob_name = "predictions/predictions.csv"
    predictions_path = "/tmp/predictions.csv"

    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(predictions_blob_name)
    
    if not blob.exists():
        logging.error("Batch predictions file not found in GCS.")
        return None
    
    blob.download_to_filename(predictions_path)
    logging.info("Batch predictions downloaded successfully.")

    validation_data = pd.read_csv(validation_data_path)
    predictions = pd.read_csv(predictions_path)
    
    y_true = validation_data["BREAKS"]
    y_pred = predictions["predictions"]  # Ensure column name matches output format
    
    y_true = (y_true != 0).astype(int)
    y_pred = (y_pred != 0).astype(int)
    return f1_score(y_true, y_pred, average='binary')

# Step 2: Download the latest model from GCS
latest_model_path = get_latest_model_from_gcs()
blob = storage_client.bucket(GCS_BUCKET).blob(latest_model_path)
blob.download_to_filename(MODEL_PATH)
logging.info("Model downloaded successfully.")

# Step 3: Evaluate the new model
new_model_score = evaluate_new_model_on_same_data(MODEL_PATH, validation_data_path)
logging.info(f"New model evaluation score: {new_model_score}")

# Step 4: Trigger Cloud Run batch prediction job and evaluate the deployed model
if trigger_cloud_run_batch_prediction():
    deployed_model_score = evaluate_deployed_model_on_same_data(validation_data_path)
    logging.info(f"Deployed model evaluation score: {deployed_model_score}")
else:
    logging.error("Batch job did not complete successfully. Skipping deployment.")
    deployed_model_score = None

# Step 5: Compare performance and decide on deployment

# if deployed_model_score is None:
#     logging.error("Failed to evaluate the deployed model. Skipping deployment.")
# elif new_model_score > deployed_model_score:
#     logging.info("New model performs better. Proceeding with deployment...")

#     # Upload new model to GCS with timestamped name
#     timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
#     new_model_filename = f"models_versioning/model_{timestamp}_{new_model_score:.4f}.pkl"
#     new_blob = storage_client.bucket(GCS_BUCKET).blob(new_model_filename)
#     new_blob.upload_from_filename(MODEL_PATH)
#     logging.info(f"New model uploaded to GCS: {new_model_filename}")

#     # Update "latest-model.txt" with the new model filename
#     latest_blob = storage_client.bucket(GCS_BUCKET).blob("models_versioning/latest-model.txt")
#     try:
#         existing_history = latest_blob.download_as_text()
#     except Exception:
#         existing_history = ""  # If the file doesn't exist yet, start fresh

#     # Append the new model at the top
#     updated_history = f"{new_model_filename}\n{existing_history}"

#     # Upload updated history back to GCS
#     latest_blob.upload_from_string(updated_history)
#     logging.info(f"Updated latest model reference with history in latest-model.txt")

# else:
#     logging.info("New model does not perform better. Skipping deployment.")

with mlflow.start_run(run_name="model_comparison"):
    mlflow.log_metric("new_model_f1", new_model_score)
    if deployed_model_score is not None:
        mlflow.log_metric("deployed_model_f1", deployed_model_score)

    if new_model_score > deployed_model_score:
        
        logging.info("New model performs better. Proceeding with deployment...")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        versioned_model = f"models_versioning/model_{timestamp}_{new_model_score:.4f}.pkl"
        bucket.blob(versioned_model).upload_from_filename(MODEL_PATH)
        logging.info(f"New model uploaded to GCS: {versioned_model}")

        latest_blob = bucket.blob("models_versioning/latest-model.txt")
        try:
            history = latest_blob.download_as_text()
        except Exception:
            history = ""
        latest_blob.upload_from_string(f"{versioned_model}\n{history}")
        logging.info(f"Updated latest model reference with history in latest-model.txt")

        # Log artifact to MLflow
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.sklearn.log_model(
            sk_model=joblib.load(MODEL_PATH),
            artifact_path="sklearn-model",
            registered_model_name="AnomalyDetectionModel"
        )
        mlflow.set_tag("deployment_status", "deployed")
    else:
        logging.info("New model does not perform better. Skipping deployment.")
        mlflow.set_tag("deployment_status", "skipped")