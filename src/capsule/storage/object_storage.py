import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
from botocore.client import BaseClient

from capsule.config import Settings


@dataclass(frozen=True, slots=True)
class ObjectDownload:
    """One object-storage response, including HTTP range metadata."""

    content: bytes
    content_type: str | None
    content_range: str | None
    etag: str | None


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

    async def upload_file(
        self,
        source: Path,
        object_key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        extra_args = {"ContentType": content_type} if content_type else None

        def upload() -> None:
            kwargs = {"ExtraArgs": extra_args} if extra_args else {}
            self._client.upload_file(
                str(source),
                self._bucket,
                object_key,
                **kwargs,
            )

        await asyncio.to_thread(upload)
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

    async def presigned_get_uri(self, uri: str, *, expires_seconds: int = 3600) -> str:
        """Turn a stored ``s3://`` URI into a temporary model-readable URL."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self._bucket or not parsed.path.lstrip("/"):
            raise ValueError(f"object URI does not belong to configured bucket: {uri!r}")
        return await self.presigned_get_url(
            unquote(parsed.path.lstrip("/")),
            expires_seconds=expires_seconds,
        )

    async def download_uri(self, uri: str) -> bytes:
        return (await self.download_uri_response(uri)).content

    async def download_uri_response(
        self,
        uri: str,
        *,
        byte_range: str | None = None,
    ) -> ObjectDownload:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self._bucket or not parsed.path.lstrip("/"):
            raise ValueError(f"object URI does not belong to configured bucket: {uri!r}")

        object_key = unquote(parsed.path.lstrip("/"))

        def download() -> ObjectDownload:
            kwargs: dict[str, str] = {
                "Bucket": self._bucket,
                "Key": object_key,
            }
            if byte_range is not None:
                kwargs["Range"] = byte_range
            response = self._client.get_object(**kwargs)
            body = response["Body"]
            try:
                content = bytes(body.read())
            finally:
                body.close()
            return ObjectDownload(
                content=content,
                content_type=response.get("ContentType"),
                content_range=response.get("ContentRange"),
                etag=response.get("ETag"),
            )

        return await asyncio.to_thread(download)

    async def download_object(self, object_key: str) -> bytes:
        """Download one object by key without exposing a private storage URL."""
        if not object_key:
            raise ValueError("object key must not be empty")

        def download() -> bytes:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=object_key,
            )
            body = response["Body"]
            try:
                return bytes(body.read())
            finally:
                body.close()

        return await asyncio.to_thread(download)

    async def delete_uris(self, uris: Iterable[str]) -> int:
        """Delete known objects in this bucket and return the number requested."""
        keys: list[str] = []
        for uri in uris:
            parsed = urlparse(uri)
            if (
                parsed.scheme != "s3"
                or parsed.netloc != self._bucket
                or not parsed.path.lstrip("/")
            ):
                raise ValueError(f"object URI does not belong to configured bucket: {uri!r}")
            keys.append(unquote(parsed.path.lstrip("/")))
        if not keys:
            return 0

        def delete() -> int:
            for start in range(0, len(keys), 1_000):
                batch = keys[start : start + 1_000]
                response = self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
                )
                errors = response.get("Errors") or []
                if errors:
                    messages = [
                        str(item.get("Message") or item.get("Code") or "unknown")
                        for item in errors
                    ]
                    raise RuntimeError(
                        f"could not delete object storage records: {', '.join(messages)}"
                    )
            return len(keys)

        return await asyncio.to_thread(delete)

    async def delete_all_objects(self) -> int:
        """Delete every object in Capsule's configured, dedicated bucket."""

        def delete_all() -> int:
            deleted = 0
            while True:
                response = self._client.list_objects_v2(Bucket=self._bucket)
                keys = [
                    {"Key": str(item["Key"])}
                    for item in response.get("Contents") or []
                    if item.get("Key")
                ]
                if keys:
                    deletion = self._client.delete_objects(
                        Bucket=self._bucket,
                        Delete={"Objects": keys, "Quiet": True},
                    )
                    errors = deletion.get("Errors") or []
                    if errors:
                        messages = [
                            str(item.get("Message") or item.get("Code") or "unknown")
                            for item in errors
                        ]
                        raise RuntimeError(
                            "could not delete object storage records: "
                            + ", ".join(messages)
                        )
                    deleted += len(keys)
                if not keys:
                    return deleted

        return await asyncio.to_thread(delete_all)
