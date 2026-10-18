"""Download the public Chinook sample database (SQLite edition).

Usage:  uv run python data/download_chinook.py [--force]

The file is pinned to a release tag and verified with SHA-256 so every
environment (local, CI, Docker, HF Spaces) evaluates against identical data.
"""

import argparse
import hashlib
import sqlite3
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
SHA256 = "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"
DEST = Path(__file__).resolve().parent / "chinook.sqlite"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest: Path = DEST, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"Already present: {dest}")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    digest = sha256(tmp)
    if SHA256 and digest != SHA256:
        tmp.unlink()
        sys.exit(f"Checksum mismatch: expected {SHA256}, got {digest}")
    tmp.replace(dest)
    with sqlite3.connect(f"file:{dest}?mode=ro", uri=True) as conn:
        n = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    print(f"Saved {dest} ({n} tables, sha256={digest})")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(force=parser.parse_args().force)
