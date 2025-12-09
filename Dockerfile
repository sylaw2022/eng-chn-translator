# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# Copy application code
COPY app.py .
COPY custom_transformer.py .

# Copy model artifacts
# Assuming model files are in a local 'model' directory before building
COPY results/final_model.pt ./model/final_model.pt
COPY results/final_model_tokenizer ./model/final_model_tokenizer

# Expose port (Cloud Run uses 8080 by default)
ENV PORT=8080
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]

