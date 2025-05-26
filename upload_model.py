import os
import joblib
import logging
import time
import requests
import pandas as pd
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
ENDPOINT_ID = 5796490061704855552
GCS_BUCKET = "anomaly-detection-knn"
VALIDATION_DATA_PATH = "training_data/combined_data_breaks.csv"
CLOUD_RUN_JOB_NAME = "predict-from-csv"  # Update with your Cloud Run Job name

MODEL_PATH = "/tmp/model.pkl"
GCS_MODEL_PATH_PREFIX = "models/"

# Initialize Storage Client
storage_client = storage.Client()

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

def get_batch_predictions():
    """Downloads batch predictions from GCS and computes the F1-score."""
    logging.info("Fetching batch predictions from GCS...")

    predictions_blob_name = "predictions/predictions.csv"
    predictions_path = "/tmp/predictions.csv"

    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(predictions_blob_name)
    
    if not blob.exists():
        logging.error("Batch predictions file not found in GCS.")
        return None
    
    blob.download_to_filename(predictions_path)
    logging.info("Batch predictions downloaded successfully.")

    # Load predictions and ground truth
    validation_data = pd.read_csv("/tmp/validation_data.csv")
    predictions = pd.read_csv(predictions_path)
    
    y_true = validation_data["BREAKS"]
    y_pred = predictions["predictions"]  # Ensure column name matches output format
    
    y_true = (y_true != 0).astype(int)
    y_pred = (y_pred != 0).astype(int)
    return f1_score(y_true, y_pred, average='binary')

def evaluate_new_model(model_path):
    """Evaluates the new model on the validation dataset."""
    model = joblib.load(model_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(VALIDATION_DATA_PATH)
    blob.download_to_filename("/tmp/validation_data.csv")

    validation_data = pd.read_csv("/tmp/validation_data.csv")
    X_val = validation_data.drop(columns=["BREAKS"])
    y_val = validation_data["BREAKS"]
    y_test_binary  = (y_val != 0).astype(int)
    expected_feature_order = ['smd_0', 'smd_1', 'smd_2', 'smd_3', 'smd_4', 'aoi_0', 'aoi_1', 'aoi_2', 'aoi_3', 'aoi_4', 'ss_0', 'ss_1', 'ss_2', 'ss_3', 'ss_4', 'cc_0', 'cc_1', 'Overall_processing_time', 'Tardiness']
    X_val = X_val[expected_feature_order] 
    predictions = model.predict(X_val)
    predictions_binary = (predictions != 0).astype(int)

    return f1_score(y_test_binary , predictions_binary, average='binary')

# Step 1: Download the latest model from GCS
latest_model_path = get_latest_model_from_gcs()
blob = storage_client.bucket(GCS_BUCKET).blob(latest_model_path)
blob.download_to_filename(MODEL_PATH)
logging.info("Model downloaded successfully.")

# Step 2: Evaluate the new model
new_model_score = evaluate_new_model(MODEL_PATH)
logging.info(f"New model evaluation score: {new_model_score}")

# Step 3: Trigger Cloud Run batch prediction job
if trigger_cloud_run_batch_prediction():
    deployed_model_score = get_batch_predictions()
    logging.info(f"Deployed model evaluation score: {deployed_model_score}")
else:
    logging.error("Batch job did not complete successfully. Skipping deployment.")
    deployed_model_score = None

# Step 4: Compare performance and decide on deployment
if deployed_model_score is None:
    logging.error("Failed to evaluate the deployed model. Skipping deployment.")
elif new_model_score > deployed_model_score:
    logging.info("New model performs better. Proceeding with deployment...")

    # Upload new model to GCS with timestamped name
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    new_model_filename = f"models_versioning/model_{timestamp}_{new_model_score:.4f}.pkl"
    new_blob = storage_client.bucket(GCS_BUCKET).blob(new_model_filename)
    new_blob.upload_from_filename(MODEL_PATH)
    logging.info(f"New model uploaded to GCS: {new_model_filename}")

    # Update "latest-model.txt" with the new model filename
    latest_blob = storage_client.bucket(GCS_BUCKET).blob("models_versioning/latest-model.txt")
    # latest_blob.upload_from_string(new_model_filename)
    # logging.info(f"Updated latest model reference to: {new_model_filename}")
    try:
        existing_history = latest_blob.download_as_text()
    except Exception:
        existing_history = ""  # If the file doesn't exist yet, start fresh

    # Append the new model at the top
    updated_history = f"{new_model_filename}\n{existing_history}"

    # Upload updated history back to GCS
    latest_blob.upload_from_string(updated_history)
    logging.info(f"Updated latest model reference with history in latest-model.txt")

    # Register the new model in Vertex AI Model Registry
    # model = aiplatform.Model.upload(
    #     display_name=f"model_{new_model_score:.4f}",
    #     artifact_uri=f"gs://{GCS_BUCKET}/models_versioning",
    #     serving_container_image_uri="us-central1-docker.pkg.dev/abhishreya-sharma-ma/predictor-repo/predictor:latest"
    # )
    # logging.info(f"Model registered successfully in Vertex AI Model Registry.")

    # # Deploy to the existing endpoint
    # model.deploy(
    #     endpoint="5796490061704855552",
    #     machine_type="n1-standard-4",
    #     traffic_split={"0": 100}  # Full traffic to new model
    # )

    #logging.info(f"Model deployed successfully to Vertex AI endpoint.")
else:
    logging.info("New model does not perform better. Skipping deployment.")
