import os

GCS_PROJECT_ID = os.environ.get("GCS_PROJECT_ID", None)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", None)

CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

KFP_ENDPOINT = os.environ.get("KFP_ENDPOINT", None)
