# Use official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY predict_from_csv.py .

# Set the entrypoint for the container to run the script
CMD ["python", "predict_from_csv.py"]