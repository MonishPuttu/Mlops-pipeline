import boto3
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path
from config.utils import load_config

def _get_client():
    cfg = load_config()
    return boto3.client(
        "s3",
        endpoint_url=cfg["minio"]["endpoint"],
        aws_access_key_id=cfg["minio"]["access_key"],
        aws_secret_access_key=cfg["minio"]["secret_key"],
    )

def upload_csv(df: pd.DataFrame, bucket: str, key: str):
    client = _get_client()
    buf = StringIO()
    df.to_csv(buf, index=False)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
    )

def download_csv(bucket: str, key: str) -> pd.DataFrame:
    client = _get_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))

def upload_file(local_path: str, bucket: str, key: str):
    client = _get_client()
    with open(local_path, "rb") as f:
        client.put_object(Bucket=bucket, Key=key, Body=f.read())

def download_file(bucket: str, key: str, local_path: str):
    client = _get_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(obj["Body"].read())

def key_exists(bucket: str, key: str) -> bool:
    client = _get_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False
    except Exception:
        return False