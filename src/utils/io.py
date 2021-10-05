import os
from pathlib import Path

from google.cloud import storage
from google.oauth2.service_account import Credentials

from src.utils.environ import CREDENTIALS_PATH, GCS_BUCKET_NAME, GCS_PROJECT_ID


def get_upstream_dir() -> Path:
    path = Path(os.environ.get("UPSTREAM_DIR", "./Upstream/"))
    return path


def get_downstream_dir() -> Path:
    path = Path(os.environ.get("DOWNSTREAM_DIR", "./Downstream/"))
    return path


def upload_to_bucket(
    data: str, gcs_path: Path, bucket_name: str = GCS_BUCKET_NAME
) -> Path:
    credentials = Credentials.from_service_account_file(CREDENTIALS_PATH)
    client = storage.Client(credentials=credentials, project=GCS_PROJECT_ID)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_path.as_posix())
    blob.upload_from_string(data)
    return Path("gs://" + bucket_name).joinpath(gcs_path)

def download_from_bucket(file_path: Path, gcs_path: Path, bucket_name: str = GCS_BUCKET_NAME ):
    credentials = Credentials.from_service_account_file(CREDENTIALS_PATH)
    client = storage.Client(credentials=credentials, project=GCS_PROJECT_ID)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_path.as_posix())
    blob.download_to_filename(Path)
