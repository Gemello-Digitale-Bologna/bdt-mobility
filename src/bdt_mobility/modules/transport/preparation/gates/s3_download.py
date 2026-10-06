import os
import boto3
from pathlib import Path


def s3_download_file(bucket_name: str, file_path: str, output_path: Path) -> None:
    """
    Download a file from S3 using credentials from environment variables.

    Parameters:
        bucket_name: S3 bucket name
        file_path: Path to file in S3 bucket
        output_path: Local path where file will be saved

    Environment variables:
        - For standard buckets: Uses digitalhub CLI config
        - For 'protected' bucket:
            - PROTECTED_AWS_ENDPOINT_URL
            - PROTECTED_AWS_ACCESS_KEY_ID
            - PROTECTED_AWS_SECRET_ACCESS_KEY
    """
    if bucket_name == 'protected':
        # Protected bucket requires explicit credentials from environment
        endpoint_url = os.getenv("PROTECTED_AWS_ENDPOINT_URL")
        access_key_id = os.getenv("PROTECTED_AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("PROTECTED_AWS_SECRET_ACCESS_KEY")

        if not all([endpoint_url, access_key_id, secret_access_key]):
            raise ValueError(
                "Protected bucket requires environment variables: "
                "PROTECTED_AWS_ENDPOINT_URL, PROTECTED_AWS_ACCESS_KEY_ID, "
                "PROTECTED_AWS_SECRET_ACCESS_KEY"
            )

        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
    else:
        # Standard bucket: use digitalhub CLI configuration
        import digitalhub as dh
        s3_client = dh.get_s3_client()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket_name, file_path, str(output_path))