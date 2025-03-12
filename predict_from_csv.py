import os
import pandas as pd
import json
import requests
from google.cloud import storage
from google.auth import default
from google.auth.transport.requests import Request

# Use environment variables for dynamic configuration
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")
BLOB_NAME = os.getenv("BLOB_NAME", "testing_data/test_data.csv")
PREDICTION_URL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}:predict"

# Set batch size for predictions
BATCH_SIZE = 100  # Adjust based on the size of individual records

def get_access_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

def main():
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(BLOB_NAME)
    
    # Download CSV data from GCS to local environment (temporary location)
    blob.download_to_filename('/tmp/test_data.csv')
    test_data = pd.read_csv('/tmp/test_data.csv')

    all_predictions = []

    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }
    
    # Split data into batches to avoid exceeding the 1.5 MB limit
    for start_idx in range(0, len(test_data), BATCH_SIZE):
        batch_data = test_data.iloc[start_idx:start_idx + BATCH_SIZE]
        instances = batch_data.to_dict(orient='records')
        payload = json.dumps({"instances": instances})
        
        response = requests.post(PREDICTION_URL, headers=headers, data=payload)
        
        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            all_predictions.extend(predictions)
            print(f"Processed batch {start_idx // BATCH_SIZE + 1}")
        else:
            print(f"Prediction failed for batch {start_idx // BATCH_SIZE + 1}: {response.text}")
            return

    # Combine predictions with original data and save to GCS
    test_data['predictions'] = all_predictions
    output_file = '/tmp/predictions.csv'
    test_data.to_csv(output_file, index=False)
    
    # Upload results back to GCS
    output_blob = bucket.blob("predictions/predictions.csv")
    output_blob.upload_from_filename(output_file)
    print("Predictions saved to GCS!")

if __name__ == '__main__':
    main()
