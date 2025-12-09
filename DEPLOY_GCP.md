# Deploying to Google Cloud Run

This guide explains how to deploy your custom English-Chinese translation model to Google Cloud Run using the artifacts provided.

## Prerequisites

1.  **Google Cloud Project**: You need an active GCP project.
2.  **Google Cloud SDK**: Install the `gcloud` CLI tool.
3.  **Docker**: Installed locally (if you want to test the build locally).
4.  **Billing**: Enabled on your GCP project.

## 1. Setup

Ensure your project file structure looks like this (which it should if you are in the project root):

```
.
├── app.py
├── custom_transformer.py
├── Dockerfile
├── requirements_serving.txt
└── results/
    ├── final_model.pt
    └── final_model_tokenizer/
```

## 2. Build and Push to Google Container Registry (GCR)

Replace `PROJECT_ID` with your actual Google Cloud Project ID.

1.  **Authenticate**:
    ```bash
    gcloud auth login
    gcloud config set project PROJECT_ID
    ```

2.  **Enable Container Registry API**:
    ```bash
    gcloud services enable containerregistry.googleapis.com
    ```

3.  **Build the Image using Cloud Build** (easiest way, no local Docker needed):
    ```bash
    gcloud builds submit --tag gcr.io/PROJECT_ID/eng-chn-translator
    ```
    *This uploads your files to GCP and builds the Docker image there.*

## 3. Deploy to Cloud Run

Deploy the container image to a fully managed serverless environment.

```bash
gcloud run deploy eng-chn-translator \
  --image gcr.io/PROJECT_ID/eng-chn-translator \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated
```

*   `--memory 2Gi`: Allocates 2GB RAM. Since the model is ~500MB, this should be safe. Increase to 4Gi if you get OOM errors.
*   `--allow-unauthenticated`: Makes the API public. Remove this flag if you want to require authentication.

## 4. Test the API

Once deployed, Cloud Run will give you a URL (e.g., `https://eng-chn-translator-xyz-uc.a.run.app`).

### Using curl

```bash
curl -X POST "https://YOUR_CLOUD_RUN_URL/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you?"}'
```

### Using Python

```python
import requests

url = "https://YOUR_CLOUD_RUN_URL/translate"
data = {"text": "Hello, how are you?"}

response = requests.post(url, json=data)
print(response.json())
```

## 5. Local Testing (Optional)

You can build and run locally to test before deploying:

```bash
# Build
docker build -t translator-app .

# Run
docker run -p 8080:8080 translator-app
```

Then visit `http://localhost:8080/docs` to use the interactive Swagger UI.

