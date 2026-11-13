"""Credentials must never reach git or a Docker image.

Humphrey keeps two kinds of secret on disk: an API key in .env, and ChatGPT
sign-in tokens under the credentials directory (app/auth). These checks fail
if either location stops being ignored, or if anything shaped like a real
credential is committed. Fakes in tests are built at runtime for the same
reason, so a real token pasted into a fixture fails here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Each matches the shape of a real credential, never a placeholder like "sk-...".
SECRET_PATTERNS = {
    "JWT (access or ID token)": re.compile(r"eyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}"),
    "OpenAI API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "issued ChatGPT client id": re.compile(r"\boaiapp_[A-Za-z0-9]{8,}"),
    "stored refresh token": re.compile(r'"refresh_token"\s*:\s*"[^"]{20,}"'),
}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


requires_git = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


@requires_git
@pytest.mark.parametrize("path", [".env", "data/credentials/chatgpt.json"])
def test_secret_locations_are_ignored_by_git(path: str) -> None:
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not git-ignored"


def test_the_default_credentials_dir_is_under_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import get_settings, reset_settings_for_tests

    monkeypatch.delenv("HUMPHREY_CREDENTIALS_DIR", raising=False)
    reset_settings_for_tests()
    assert Path(get_settings().credentials_dir).resolve() == (ROOT / "data" / "credentials").resolve()


def test_secret_locations_are_kept_out_of_docker_images() -> None:
    ignored = {line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()}
    assert {".env", "data"} <= ignored


@requires_git
def test_no_tracked_file_contains_a_credential() -> None:
    listing = _git("ls-files", "-z")
    tracked = [name for name in listing.stdout.split("\0") if name]
    assert listing.returncode == 0 and len(tracked) > 50, "git ls-files returned too little to check"
    found: list[str] = []
    for name in tracked:
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary files (images) and files deleted in the working tree
        for kind, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                found.append(f"{name}: {kind}")
    assert found == []


def test_the_patterns_catch_real_shapes() -> None:
    """Guard the guard: built at runtime, so this file never holds a match."""
    jwt = ".".join(["eyJ" + "a" * 12, "eyJ" + "b" * 12, "c" * 12])
    samples = {
        "JWT (access or ID token)": jwt,
        "OpenAI API key": "sk-" + "A1b2" * 6,
        "issued ChatGPT client id": "oaiapp_" + "Z9" * 6,
        "stored refresh token": '"refresh_token": "' + "r" * 30 + '"',
    }
    for kind, sample in samples.items():
        assert SECRET_PATTERNS[kind].search(sample), kind
    assert not SECRET_PATTERNS["OpenAI API key"].search('OPENAI_API_KEY="sk-..."')
