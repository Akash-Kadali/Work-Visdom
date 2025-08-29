# backend/drive_utils.py
from __future__ import annotations

"""
Local-mode Drive utils
----------------------

This module provides a *Drive-compatible* surface area, but all operations
are performed on the local filesystem. It lets the rest of the app call
things like `authenticate()` and `resolve_folder(...)` without caring
whether Google Drive is actually configured.

Highlights:
- `authenticate()` / `get_service()` are no-ops that return None.
- `resolve_folder(<ENV_KEY or path>)` maps env-style IDs/keys to local paths
  using backend.paths.folder(...).
- Helpers for listing/copying files between these mapped folders.
- CSV helpers that read/write via pandas.
- `upload_text_as_csv(...)` used by the app to "upload" (write) a CSV file.

All paths are returned as `Path` objects or stringified absolute paths.
"""

from pathlib import Path
from typing import List, Optional, Union

import os
import shutil
import pandas as pd

from backend.paths import folder, FOLDERS, IMAGES_DIR, CSV_DIR, OUTPUT_DIR, TEMP_DIR
from backend.local_store import list_images, read_csv, write_csv

Pathish = Union[str, Path]

__all__ = [
    "authenticate",
    "get_service",
    "resolve_folder",
    "list_images_in_folder_id",
    "upload_file_to_folder",
    "download_file_from_folder",
    "read_csv_local",
    "write_csv_local",
    "upload_text_as_csv",
]


# ---------------------------------------------------------------------------
# Drive compatibility shims
# ---------------------------------------------------------------------------

def authenticate():
    """
    Local mode: there is no remote service; return None.
    Code elsewhere can test `if service is None` to branch to local.
    """
    return None


def get_service():
    """Alias for authenticate(), kept for compatibility."""
    return authenticate()


# ---------------------------------------------------------------------------
# Folder resolution
# ---------------------------------------------------------------------------

def resolve_folder(folder_id_or_path: str) -> Path:
    """
    Accept either:
      - An env-style key (e.g. 'OUTPUT_LASER2_FOLDER_ID'), or
      - A literal filesystem path (absolute or relative).
    Returns a Path (may point to a non-existent location; callers can mkdir).
    """
    return folder(folder_id_or_path)


# ---------------------------------------------------------------------------
# Internal copy helper
# ---------------------------------------------------------------------------

def _copy_to(src_path: Pathish, dest_dir: Pathish) -> Path:
    """
    Copy `src_path` into the directory `dest_dir`.
    Returns the *destination file path*.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dst_dir = Path(dest_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# File & listing helpers
# ---------------------------------------------------------------------------

def list_images_in_folder_id(folder_id_or_path: str) -> List[Path]:
    """
    Return a list of image Paths within the resolved folder.
    """
    return list_images(resolve_folder(folder_id_or_path))


def upload_file_to_folder(local_path: Pathish, dest_folder_id_or_path: str) -> Path:
    """
    Copy `local_path` into the resolved destination folder.
    Returns the destination Path.
    """
    return _copy_to(local_path, resolve_folder(dest_folder_id_or_path))


def download_file_from_folder(
    src_folder_id_or_path: str,
    filename: str,
    dest_dir: Pathish = TEMP_DIR,
) -> Path:
    """
    Copy a file named `filename` from the source folder into `dest_dir`.
    Returns the destination Path.
    """
    src = resolve_folder(src_folder_id_or_path) / filename
    return _copy_to(src, dest_dir)


# ---------------------------------------------------------------------------
# CSV helpers (local, via pandas)
# ---------------------------------------------------------------------------

def _resolve_path_or_folder_key(p: Pathish) -> Path:
    """
    If `p` is a string that matches a known folder key, return that folder Path.
    Otherwise treat `p` as a literal Path.
    """
    if isinstance(p, str) and p in FOLDERS:
        return resolve_folder(p)
    return Path(p)


def read_csv_local(path_or_envkey: Pathish) -> pd.DataFrame:
    """
    Read a CSV from a literal path or from a folder key + filename.
    If a folder key is provided, the *caller* should pass a filename path,
    not just the folder key (use write_csv_local for folder+name cases).
    """
    p = _resolve_path_or_folder_key(path_or_envkey)
    return read_csv(p)


def write_csv_local(
    df: pd.DataFrame,
    path_or_envkey: Pathish,
    name_if_folder: Optional[str] = None,
) -> Path:
    """
    Write a CSV either to a literal path or into a resolved folder when the
    argument is a folder key. When using a folder key, `name_if_folder` is
    required to form the destination file name within that folder.
    Returns the written file Path.
    """
    if isinstance(path_or_envkey, str) and path_or_envkey in FOLDERS:
        if not name_if_folder:
            raise ValueError("write_csv_local: name_if_folder required when path is a folder key")
        dest = resolve_folder(path_or_envkey) / name_if_folder
    else:
        dest = Path(path_or_envkey)

    return write_csv(df, dest)


# ---------------------------------------------------------------------------
# Text/CSV upload compatibility
# ---------------------------------------------------------------------------

def upload_text_as_csv(
    text: str,
    filename: str,
    folder_id: str = "",
    replace_existing: bool = True,
) -> str:
    """
    Local replacement so app routes don't break when Drive is disabled.

    Writes `text` to a CSV file named `filename` under:
      - the resolved `folder_id` if provided (env key or literal path), else
      - CSV_DIR

    If `replace_existing` is False and the file already exists, the existing
    file is kept and its path is returned.

    Returns:
        Absolute file path as a string (acts as a 'file_id' for local mode).
    """
    # Choose output directory
    out_dir: Path
    if folder_id:
        out_dir = resolve_folder(folder_id)
    else:
        out_dir = CSV_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = "results.csv"

    path = out_dir / filename

    if path.exists() and not replace_existing:
        return str(path.resolve())

    # Ensure parent exists and write text
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text or "")

    return str(path.resolve())
