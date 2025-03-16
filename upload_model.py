# import os
# import joblib
# import pandas as pd
# from google.cloud import storage, aiplatform
# from google.cloud.storage import Blob
# from sklearn.metrics import f1_score

# # Configurations
# PROJECT_ID = os.getenv("PROJECT_ID")
# REGION = os.getenv("REGION", "us-central1")
# BUCKET_NAME = os.getenv("GCS_BUCKET")
# VALIDATION_DATA_PATH = os.getenv("VALIDATION_DATA_PATH", "training_data/combined_data_breaks.csv")

# # Initialize clients
# storage_client = storage.Client()
# aiplatform.init(project=PROJECT_ID, location=REGION)

# def get_latest_model_from_gcs(bucket_name, prefix="models/"):
#     """Finds the most recently uploaded model file in GCS."""
#     bucket = storage_client.bucket(bucket_name)
#     blobs = list(bucket.list_blobs(prefix=prefix))
    
#     if not blobs:
#         print("No model files found in the bucket.")
#         return None
    
#     # Sort files by updated timestamp (latest first)
#     latest_blob = sorted(blobs, key=lambda x: x.updated, reverse=True)[0]
    
#     return latest_blob.name  # Return the full GCS path of the latest model

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)

# def load_model(model_path):
#     return joblib.load(model_path)

# def get_current_model():
#     """Retrieves the currently deployed model."""
#     models = aiplatform.Model.list(order_by="update_time", filter="labels.deployed=true")
    
#     if models:
#         latest_model = models[0]  # Latest deployed model
#         current_model_path = latest_model.uri.replace("gs://", "").split("/", 1)
#         local_path = "/tmp/current_model.pkl"
#         download_blob(current_model_path[0], current_model_path[1], local_path)
#         return load_model(local_path), latest_model
#     return None, None

# def evaluate_model(model, X, y):
#     """Computes F1-score for the given model."""
#     predictions = model.predict(X)
#     return f1_score(y, predictions)

# def main():
#     # Get latest uploaded model from GCS dynamically
#     latest_model_filename = get_latest_model_from_gcs(BUCKET_NAME)
#     if not latest_model_filename:
#         print("No new model found. Exiting.")
#         return
    
#     new_model_local_path = "/tmp/new_model.pkl"
    
#     # Download new model and validation data
#     download_blob(BUCKET_NAME, latest_model_filename, new_model_local_path)
#     download_blob(BUCKET_NAME, VALIDATION_DATA_PATH, "/tmp/validation_data.csv")
    
#     # Load new model and validation data
#     new_model = load_model(new_model_local_path)
#     data = pd.read_csv("/tmp/validation_data.csv")
#     X = data.drop(columns=["BREAKS"])
#     y = data["BREAKS"]
    
#     # Evaluate new model
#     new_model_f1 = evaluate_model(new_model, X, y)
#     print(f"New model F1-score: {new_model_f1}")
    
#     # Get current model and evaluate
#     current_model, deployed_model = get_current_model()
#     if current_model:
#         current_model_f1 = evaluate_model(current_model, X, y)
#         print(f"Current model F1-score: {current_model_f1}")
        
#         if new_model_f1 <= current_model_f1:
#             print("New model does not outperform the current model. Skipping deployment.")
#             return
    
#     # Upload and deploy the new model
#     new_model_gcs_path = f"gs://{BUCKET_NAME}/{latest_model_filename}"
#     aiplatform.Model.upload(
#         display_name="anomaly-detector",
#         artifact_uri=new_model_gcs_path,
#         serving_container_image_uri="gcr.io/cloud-aiplatform/prediction"
#     )
#     print(f"New model '{latest_model_filename}' uploaded and deployed successfully.")

# if __name__ == "__main__":
#     main()
import os
import joblib
import logging
import requests
from google.cloud import storage
from google.cloud import aiplatform
from google.auth import default
from google.auth.transport.requests import Request
import pandas as pd
from sklearn.metrics import f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Use environment variables for dynamic configuration
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = "5796490061704855552"
GCS_BUCKET = os.getenv("GCS_BUCKET")
VALIDATION_DATA_PATH = os.getenv("VALIDATION_DATA_PATH")

# Model paths
MODEL_PATH = "/tmp/model.pkl"
GCS_MODEL_PATH = f"gs://{GCS_BUCKET}/models/"

def get_latest_model_from_gcs():
    """Fetch the latest model file from the GCS models/ directory."""
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(GCS_BUCKET, prefix="models/"))

    if not blobs:
        logging.error("No models found in GCS.")
        return None

    # Sort blobs by time created (latest first)
    latest_blob = sorted(blobs, key=lambda b: b.time_created, reverse=True)[0]
    latest_model_path = latest_blob.name
    logging.info(f"Latest model found: {latest_model_path}")
    return latest_model_path

def get_deployed_model_predictions(validation_data_path):
    """Get predictions from the currently deployed model on Vertex AI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(validation_data_path)
    blob.download_to_filename("/tmp/validation_data.csv")

    validation_data = pd.read_csv("/tmp/validation_data.csv")
    X_val = validation_data.drop(columns=["BREAKS"])  # Assuming BREAKS column is 'BREAKS'
    y_val = validation_data["BREAKS"]

    # Prepare payload for prediction
    instances = X_val.to_dict(orient="records")
    prediction_url = (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}:predict"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}

    # Get the access token
    credentials, _ = default()
    credentials.refresh(Request())
    headers["Authorization"] = f"Bearer {credentials.token}"

    # Make prediction request
    response = requests.post(prediction_url, json=payload, headers=headers)

    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        return f1_score(y_val, predictions)  # Assuming predictions are formatted correctly
    else:
        logging.error(f"Prediction failed: {response.text}")
        return None

def evaluate_new_model(model_path):
    """Evaluate the new model locally using validation data."""
    model = joblib.load(model_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(VALIDATION_DATA_PATH)
    blob.download_to_filename("/tmp/validation_data.csv")

    validation_data = pd.read_csv("/tmp/validation_data.csv")
    X_val = validation_data.drop(columns=["BREAKS"])
    y_val = validation_data["BREAKS"]

    predictions = model.predict(X_val)
    return f1_score(y_val, predictions)

# Get the latest model from GCS
latest_model_path = get_latest_model_from_gcs()
if latest_model_path:
    logging.info("Downloading latest model from GCS...")
    storage_client = storage.Client()
    blob = storage_client.bucket(GCS_BUCKET).blob(latest_model_path)
    blob.download_to_filename(MODEL_PATH)
    logging.info("Model downloaded successfully.")
else:
    logging.error("No model found in GCS. Exiting.")
    exit(1)

# Load and evaluate the new model
new_model_score = evaluate_new_model(MODEL_PATH)
logging.info(f"New model evaluation score: {new_model_score}")

# Get the current deployed model score
deployed_model_score = get_deployed_model_predictions(VALIDATION_DATA_PATH)
logging.info(f"Deployed model evaluation score: {deployed_model_score}")

# Only deploy the new model if it performs better
if deployed_model_score is None:
    logging.error("Failed to evaluate the deployed model. Skipping deployment.")
elif new_model_score > deployed_model_score:
    logging.info("New model performs better. Proceeding with deployment...")

    # Upload the new model to GCS with a unique name
    new_model_filename =  f"models/model_{new_model_score:.4f}.pkl"
    blob = storage_client.bucket(GCS_BUCKET).blob(new_model_filename)
    blob.upload_from_filename(MODEL_PATH)
    logging.info(f"New model uploaded to GCS: {new_model_filename}")

    # # Deploy the new model to Vertex AI
    # model = aiplatform.Model.upload(
    #     display_name=f"model_{new_model_score}",
    #     artifact_uri=f"gs://{GCS_BUCKET}/{new_model_filename}",
    #     serving_container_image_uri="gcr.io/cloud-aiplatform/training/tf2-cpu.2-7:latest",
    # )

    # Deploy to the endpoint
    model.deploy(endpoint=ENDPOINT_ID, machine_type="n1-standard-4", traffic_split={"0": 100})
    logging.info(f"Model deployed successfully to Vertex AI endpoint {ENDPOINT_ID}.")
else:
    logging.info("New model does not perform better. Skipping deployment.")

