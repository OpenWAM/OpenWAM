"""Portable object-store helpers; credentials use the standard AWS provider chain."""

import os


def client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        config=Config(
            read_timeout=120, retries={"max_attempts": 4}, max_pool_connections=32
        ),
    )


def object_key(sha, prefix):
    return f"{prefix.strip('/')}/objects/{sha[:2]}/{sha}"
