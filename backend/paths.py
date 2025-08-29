# backend/paths.py
from __future__ import annotations
from pathlib import Path
import os, sys
from typing import Dict

# --- Make relative paths resolve correctly in both dev and PyInstaller EXE ---
def _app_root() -> Path:
    # When frozen: use the folder that contains the executable (Visdom.exe)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Dev: repo root (two levels up from this file: backend/paths.py -> project)
    return Path(__file__).resolve().parents[1]

APP_ROOT: Path = _app_root()

# --- Load .env here defensively (non-overriding) so early imports are safe ---
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    # Only fill missing keys; system env still wins.
    load_dotenv(find_dotenv(), override=False)
except Exception:
    # If dotenv isn't present in the environment, just continue.
    pass

def _expand(v: str) -> str:
    # Expand %VAR% and $VAR plus ~user
    return os.path.expanduser(os.path.expandvars(v))

def _rel(var_name: str, default_rel: str) -> Path:
    val = _expand(os.getenv(var_name, default_rel))
    p = Path(val)
    if not p.is_absolute():
        p = APP_ROOT / p
    return p.resolve()

# --- Base dirs (can be overridden via .env) ---
IMAGES_DIR = _rel("IMAGES_DIR", "data/images")
CSV_DIR    = _rel("CSV_DIR",    "data/csv")
OUTPUT_DIR = _rel("OUTPUT_DIR", "data/output")
TEMP_DIR   = _rel("TEMP_DIR",   "data/tmp")

# Ensure they exist
for _d in (IMAGES_DIR, CSV_DIR, OUTPUT_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Map every *_FOLDER_ID env var you use to a real Path (dirs only) ---
FOLDERS: Dict[str, Path] = {
    "INPUT_IMAGE_FOLDER_ID":       _rel("INPUT_IMAGE_FOLDER_ID",       "data/images/input_full"),
    "INPUT_FOLDER_ID":             _rel("INPUT_FOLDER_ID",             "data/images/input_full"),  # alias
    "OUTPUT_LASER_FOLDER_ID":      _rel("OUTPUT_LASER_FOLDER_ID",      "data/images/crops/laser"),
    "OUTPUT_CODE_FOLDER_ID":       _rel("OUTPUT_CODE_FOLDER_ID",       "data/images/crops/code"),
    "OUTPUT_OCR_FOLDER_ID":        _rel("OUTPUT_OCR_FOLDER_ID",        "data/csv/ocr"),
    "OUTPUT_CLASSIFY_FOLDER_ID":   _rel("OUTPUT_CLASSIFY_FOLDER_ID",   "data/csv/classify"),
    "OUTPUT_BODY_FOLDER_ID":       _rel("OUTPUT_BODY_FOLDER_ID",       "data/images/regions/body"),
    "OUTPUT_LASER2_FOLDER_ID":     _rel("OUTPUT_LASER2_FOLDER_ID",     "data/images/regions/laser"),
    "OUTPUT_CRITICAL_FOLDER_ID":   _rel("OUTPUT_CRITICAL_FOLDER_ID",   "data/images/regions/critical"),
    "DEFECTIVE_LASER_FOLDER_ID":   _rel("DEFECTIVE_LASER_FOLDER_ID",   "data/images/defective/laser"),
    "DEFECTIVE_CRITICAL_FOLDER_ID":_rel("DEFECTIVE_CRITICAL_FOLDER_ID","data/images/defective/critical"),
    "DEFECTIVE_BODY_FOLDER_ID":    _rel("DEFECTIVE_BODY_FOLDER_ID",    "data/images/defective/body"),
    "OUTPUT_REPORT_FOLDER_ID":     _rel("OUTPUT_REPORT_FOLDER_ID",     "data/output/reports"),
}

def folder(path_or_envkey: str | Path) -> Path:
    """
    Accept either a key like 'OUTPUT_LASER2_FOLDER_ID' or a literal path string/Path.
    Always returns an existing directory Path. If a file path is given by mistake,
    we create/return its parent directory.
    """
    if isinstance(path_or_envkey, Path):
        p = path_or_envkey
    elif path_or_envkey in FOLDERS:
        p = FOLDERS[path_or_envkey]
    else:
        p = Path(_expand(str(path_or_envkey)))
        if not p.is_absolute():
            p = APP_ROOT / p

    # If it looks like a file path (has a suffix), use its parent as the directory.
    dir_path = p if p.suffix == "" else p.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path.resolve()
