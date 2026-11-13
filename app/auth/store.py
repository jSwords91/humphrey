"""The credential file for ChatGPT sign-in: local disk only, owner-only.

One JSON file under ``Settings.credentials_dir`` (default data/credentials,
which git and Docker both ignore). It is never part of session storage, so
tokens can't reach Azure Blob, and never reaches the browser. Writes go to a
private temporary file that replaces the real one, so a crash can't leave a
half-written file; the file is 0600 and its directory 0700.

Token fields are excluded from ``repr``, so logging a record (or an error
that contains one) can't print a token.

Some filesystems ignore chmod (a Windows drive mounted in WSL shows every
file as rwxrwxrwx). The store can't fix that, so it says so: ``is_private``
reports it, a warning is logged once, and the UI shows it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

FILE_NAME = "chatgpt.json"
FORMAT_VERSION = 1
Source = Literal["api_key", "chatgpt"]


@dataclass(slots=True)
class Tokens:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str = field(repr=False)
    expires_at: int  # unix seconds
    scope: tuple[str, ...] = ()


@dataclass(slots=True)
class Registration:
    """One ChatGPT account signed in through this app's registered client.

    ``client_id`` is the id OpenAI issued at first sign-in (``oaiapp_...``),
    bound to this account; it's reused to sign the same account in again.
    """

    client_id: str
    subject: str
    email: str | None = None
    tokens: Tokens | None = None
    # Whether the one-time "You're using your ChatGPT plan" notice was shown.
    plan_notice_seen: bool = False


@dataclass(slots=True)
class CredentialFile:
    # Identifies this Humphrey install to OpenAI (an opaque urn:uuid, not
    # user data). Created once and kept, as OpenAI's docs require.
    host_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    active_source: Source = "api_key"
    # A list so more accounts can be added later; the UI shows one for now.
    registrations: list[Registration] = field(default_factory=list)
    active_client_id: str | None = None

    def active_registration(self) -> Registration | None:
        return next(
            (r for r in self.registrations if r.client_id == self.active_client_id), None
        )

    def upsert(self, registration: Registration) -> None:
        """Add or replace the registration for its client id, and make it active."""
        self.registrations = [
            r for r in self.registrations if r.client_id != registration.client_id
        ]
        self.registrations.append(registration)
        self.active_client_id = registration.client_id


class CredentialStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.path = directory / FILE_NAME
        self._warned_not_private = False

    @property
    def lock_path(self) -> Path:
        return self._directory / f"{FILE_NAME}.lock"

    def ensure_directory(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self._directory, 0o700)

    def is_private(self) -> bool:
        """Whether only the owner can read the credentials. False where the
        filesystem ignores chmod: a Windows drive under WSL (/mnt/c) shows
        every file as rwxrwxrwx whatever Humphrey sets."""
        try:
            paths = [self._directory] + ([self.path] if self.path.exists() else [])
            return all(path.stat().st_mode & 0o077 == 0 for path in paths)
        except OSError:
            return False

    def load(self) -> CredentialFile:
        """The stored credentials, or a fresh file's worth if none (or if the
        file is unreadable: a bad file costs a sign-in, never a crash)."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return _from_json(raw)
        except FileNotFoundError:
            return CredentialFile()
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning("ChatGPT credential file is unreadable; starting signed out")
            return CredentialFile()

    def save(self, credentials: CredentialFile) -> None:
        self.ensure_directory()
        content = json.dumps(_to_json(credentials), indent=2) + "\n"
        handle, temporary = tempfile.mkstemp(dir=self._directory, prefix=f".{FILE_NAME}.")
        try:
            os.fchmod(handle, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        if not self.is_private() and not self._warned_not_private:
            self._warned_not_private = True
            logger.warning(
                "ChatGPT credentials at %s can't be made owner-only on this filesystem "
                "(chmod has no effect here, as on a Windows drive under WSL). Set "
                "HUMPHREY_CREDENTIALS_DIR to a folder on a Linux or macOS filesystem.",
                self._directory,
            )


def _to_json(credentials: CredentialFile) -> dict[str, Any]:
    return {"version": FORMAT_VERSION, **asdict(credentials)}


def _from_json(raw: Any) -> CredentialFile:
    if not isinstance(raw, dict) or raw.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported credential file")
    registrations = []
    for item in raw.get("registrations") or []:
        tokens = item.get("tokens")
        registrations.append(
            Registration(
                client_id=str(item["client_id"]),
                subject=str(item["subject"]),
                email=item.get("email"),
                tokens=Tokens(
                    access_token=str(tokens["access_token"]),
                    refresh_token=str(tokens["refresh_token"]),
                    id_token=str(tokens["id_token"]),
                    expires_at=int(tokens["expires_at"]),
                    scope=tuple(tokens.get("scope") or ()),
                )
                if tokens
                else None,
                plan_notice_seen=bool(item.get("plan_notice_seen")),
            )
        )
    source = raw.get("active_source")
    return CredentialFile(
        host_id=str(raw["host_id"]),
        active_source=source if source in ("api_key", "chatgpt") else "api_key",
        registrations=registrations,
        active_client_id=raw.get("active_client_id"),
    )
