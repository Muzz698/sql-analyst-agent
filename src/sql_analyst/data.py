"""Download and verify the public Chinook sample database (SQLite edition).

Pinned to a release tag and verified with SHA-256 so every environment (local,
CI, Docker, HF Spaces, Streamlit Cloud) evaluates against identical data.
"""

import hashlib
import logging
import sqlite3
import urllib.request
from pathlib import Path

URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
SHA256 = "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"

log = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_chinook(dest: Path, force: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("Downloading Chinook database from %s", URL)
    urllib.request.urlretrieve(URL, tmp)
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        raise ChecksumError(f"Checksum mismatch: expected {SHA256}, got {digest}")
    tmp.replace(dest)
    with sqlite3.connect(f"file:{dest}?mode=ro", uri=True) as conn:
        n = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    log.info("Saved %s (%d tables)", dest, n)
    return dest


def ensure_database(db_path: Path) -> Path:
    """Download Chinook on first run if it's the configured DB and missing."""
    db_path = Path(db_path)
    if not db_path.exists() and db_path.name == "chinook.sqlite":
        download_chinook(db_path)
    return db_path
