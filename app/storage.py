"""Object storage adapters backing session persistence: a local directory
(the default), or Azure Blob as an optional adapter (needs the `azure`
dependency group).

Kept synchronous, like the Azure Blob SDK's default client: the async
SessionStore layer offloads calls via asyncio.to_thread itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ObjectMeta:
    key: str
    last_modified: datetime | None
    size: int | None = None


class ObjectStore(Protocol):
    """A container-and-key blob store — Azure Blob or a local directory."""

    def get_bytes(self, key: str) -> bytes | None: ...

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> None: ...

    def list_keys(self, prefix: str = "") -> list[ObjectMeta]: ...


@lru_cache(maxsize=1)
def _azure_blob_service_client() -> Any | None:
    settings = get_settings()
    if not settings.azure_storage_configured:
        return None
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(
        account_url=settings.storage_account_url,
        credential=DefaultAzureCredential(),
    )


class AzureBlobObjectStore:
    """ObjectStore backed by one Azure Blob container."""

    def __init__(self, container: str) -> None:
        client = _azure_blob_service_client()
        if client is None:
            raise RuntimeError("storage account URL is not configured")
        self._client = client
        self._container = container

    def get_bytes(self, key: str) -> bytes | None:
        from azure.core.exceptions import ResourceNotFoundError

        blob = self._client.get_blob_client(container=self._container, blob=key)
        try:
            return blob.download_blob().readall()
        except ResourceNotFoundError:
            return None

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> None:
        from azure.storage.blob import ContentSettings

        blob = self._client.get_blob_client(container=self._container, blob=key)
        blob.upload_blob(
            data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )

    def list_keys(self, prefix: str = "") -> list[ObjectMeta]:
        container = self._client.get_container_client(self._container)
        return [
            ObjectMeta(key=b.name, last_modified=b.last_modified, size=b.size)
            for b in container.list_blobs(name_starts_with=prefix)
        ]


class LocalFileObjectStore:
    """One file per key under `root` — keys may contain "/" and nest into
    subdirectories, exactly like a blob name_starts_with prefix.

    Writes go through a temp file + replace() so a crash mid-write can't
    corrupt an existing session, mirroring how the Azure side never has a
    partial-upload state visible to a reader either.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"key escapes storage root: {key!r}")
        return candidate

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> None:
        _ = content_type  # no filesystem equivalent — kept for ObjectStore parity
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic on POSIX and Windows

    def list_keys(self, prefix: str = "") -> list[ObjectMeta]:
        # Walk only the directory the prefix names ("sessions/" -> sessions/),
        # not the whole root, where stored images can far outnumber sessions.
        directory, slash, _ = prefix.rpartition("/")
        base = self._path_for(directory) if slash else self._root
        if not base.is_dir():
            return []
        out: list[ObjectMeta] = []
        for path in base.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            key = path.relative_to(self._root).as_posix()
            if not key.startswith(prefix):
                continue
            stat = path.stat()
            out.append(
                ObjectMeta(
                    key=key,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    size=stat.st_size,
                )
            )
        return out


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore | None:
    """The ObjectStore session persistence (and status's introspection)
    uses, chosen by STORAGE_BACKEND.

    "local" (the default) writes JSON files under LOCAL_STORAGE_DIR.
    "azure" uses Azure Blob; "auto" uses Azure Blob when
    AZURE_STORAGE_ACCOUNT_URL is set, otherwise nothing. "memory" (and "auto"
    with nothing configured) returns None, so callers keep sessions in memory.
    """
    settings = get_settings()
    backend = settings.storage_backend
    if backend == "memory":
        return None
    if backend == "local":
        return LocalFileObjectStore(Path(settings.local_storage_dir) / settings.ai_logs_container)
    if backend == "azure" or (backend == "auto" and settings.azure_storage_configured):
        return AzureBlobObjectStore(settings.ai_logs_container)
    return None


def reset_object_store_for_tests() -> None:
    """Clear the process-wide store singletons (tests only)."""
    get_object_store.cache_clear()
    _azure_blob_service_client.cache_clear()
