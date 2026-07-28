import asyncio
from pathlib import Path

import boto3
from botocore.client import BaseClient

from capsule.config import Settings


class ObjectStorage:
    """Small S3-compatible adapter used by import and model-input stages."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.object_storage_bucket
        client_options = {
            "aws_access_key_id": settings.object_storage_access_key.get_secret_value(),
            "aws_secret_access_key": settings.object_storage_secret_key.get_secret_value(),
            "region_name": settings.object_storage_region,
        }
        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=settings.object_storage_endpoint,
            **client_options,
        )
        self._public_client: BaseClient = (
            boto3.client(
                "s3",
                endpoint_url=settings.object_storage_public_endpoint,
                **client_options,
            )
            if settings.object_storage_public_endpoint
            else self._client
        )

    async def ensure_bucket(self) -> None:
        def create_if_missing() -> None:
            try:
                self._client.head_bucket(Bucket=self._bucket)
            except self._client.exceptions.ClientError:
                self._client.create_bucket(Bucket=self._bucket)

        await asyncio.to_thread(create_if_missing)

    async def upload_file(self, source: Path, object_key: str) -> str:
        await asyncio.to_thread(
            self._client.upload_file,
            str(source),
            self._bucket,
            object_key,
        )
        return f"s3://{self._bucket}/{object_key}"

    async def upload_bytes(
        self,
        content: bytes,
        object_key: str,
        *,
        content_type: str,
    ) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=object_key,
            Body=content,
            ContentType=content_type,
        )
        return f"s3://{self._bucket}/{object_key}"

    async def presigned_get_url(self, object_key: str, *, expires_seconds: int = 3600) -> str:
        return await asyncio.to_thread(
            self._public_client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": object_key},
            ExpiresIn=expires_seconds,
        )
