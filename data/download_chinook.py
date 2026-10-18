"""Download the public Chinook sample database (SQLite edition).

Usage:  uv run python data/download_chinook.py [--force]

The app also downloads it automatically on first start (see sql_analyst.data).
"""

import argparse
import logging
from pathlib import Path

from sql_analyst.data import download_chinook

DEST = Path(__file__).resolve().parent / "chinook.sqlite"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = download_chinook(DEST, force=parser.parse_args().force)
    print(f"Chinook database ready: {path}")
