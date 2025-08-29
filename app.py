# --- Standard library ---
import csv
import io
import os
import re
import shutil
import time
import threading
from collections import deque
from glob import iglob
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from PIL import Image, ImageOps
from flask import Flask, render_template, jsonify, request, send_file, url_for
try:
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload  # type: ignore
except Exception:  # library not installed in local mode
    MediaIoBaseDownload = MediaIoBaseUpload = None  # type: ignore
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from backend.paths import IMAGES_DIR, CSV_DIR, OUTPUT_DIR, TEMP_DIR, FOLDERS, folder

# --- Local modules ---
from backend.verifier import apply_user_corrections, get_verification_items  # local-first OCR verify
from backend.drive_utils import authenticate, resolve_folder  # keep Drive helpers
from backend.paths import IMAGES_DIR, CSV_DIR, OUTPUT_DIR, TEMP_DIR, FOLDERS, folder
from backend.local_store import pending_images_from_csv, read_csv, write_csv, list_images

# === Local inspectors (pure local; no Drive required) ===
from backend.inference_jobs.coat_body import run_inference as run_body_inference
from backend.inference_jobs.maxvit_laser import run_inference as run_laser_infer_maxvit
from backend.inference_jobs.coat_laser import run_inference as run_laser_infer_coat
from backend.inference_jobs.swinv2_laser import run_inference as run_laser_infer_swinv2
from backend.inference_jobs.maxvit_critical import run_inference as run_crit_infer_maxvit
from backend.inference_jobs.coat_critical import run_inference as run_crit_infer_coat
from backend.inference_jobs.swinv2_critical import run_inference as run_crit_infer_swinv2

from backend.inference_jobs.merge_laser_results import run_merge as merge_laser_compare
from backend.inference_jobs.merge_critical_results import run_merge as merge_critical_compare


# At the very top of app.py:
import sys, socket, webbrowser
from pathlib import Path

def _base_path():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

BASE = _base_path()
def resource_path(*parts) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return (base / Path(*parts)).resolve()

from flask import Flask
app = Flask(
    __name__,
    template_folder=str(resource_path("frontend", "templates")),
    static_folder=str(resource_path("frontend", "static")),
)

# Config-derived paths
MANUAL_REVIEW_CSV = CSV_DIR / os.getenv("MANUAL_VERIFY_CSV", "manually_review.csv")

MASTER_LOCK = threading.Lock()
MASTER_STATE = {
    "status": "idle",     # idle|running|done|error
    "phase": "idle",      # crop|split|ocr|critical|max|laser|max|body|idle
    "log": [],            # brief strings
    "started_at": None,
    "finished_at": None,
    "error": None,
    "message": "Idle.",
}

try:
    from backend.modal_request import (
        classify_laser_to_csv,          # accepts models: list[str]
        classify_critical_to_csv,       # accepts models: list[str]
    )
except Exception:
    classify_laser_to_csv = None
    classify_critical_to_csv = None
try:
    from backend.modal_request import (
        classify_laser_all_models_to_csv,
        classify_critical_all_models_to_csv,
    )
except Exception:
    classify_laser_all_models_to_csv = None
    classify_critical_all_models_to_csv = None
try:
    from backend.modal_request import classify_body_to_csv  # ideal
except Exception:
    classify_body_to_csv = None
try:
    from backend.modal_request import classify_body_coat_to_csv  # alt name
except Exception:
    classify_body_coat_to_csv = None

try:
    from backend.inference_jobs.coat_body import run_inference as run_body_inference
except Exception:
    run_body_inference = None
try:
    from backend.inference_jobs.coat_body import run_inference as coat_body_run
except Exception:
    coat_body_run = None

from backend.inference_jobs.merge_laser_results import run_merge as merge_laser_results
from backend.inference_jobs.merge_critical_results import run_merge as merge_critical_results
try:
    from backend.inference_jobs.yolo_crop import run_inference as run_crop_drive_job  # type: ignore
except Exception:
    run_crop_drive_job = None
try:
    from backend.automation import run_full_pipeline as run_ocr_pipeline_job  # type: ignore
except Exception:
    run_ocr_pipeline_job = None
# --- Local model runners (add these) ---
try:
    from backend.inference_jobs.swinv2_laser import run_inference as swinv2_laser_run
except Exception:
    swinv2_laser_run = None
try:
    from backend.inference_jobs.maxvit_laser import run_inference as maxvit_laser_run
except Exception:
    maxvit_laser_run = None
try:
    from backend.inference_jobs.coat_laser import run_inference as coat_laser_run
except Exception:
    coat_laser_run = None

try:
    from backend.inference_jobs.swinv2_critical import run_inference as swinv2_critical_run
except Exception:
    swinv2_critical_run = None
try:
    from backend.inference_jobs.maxvit_critical import run_inference as maxvit_critical_run
except Exception:
    maxvit_critical_run = None
try:
    from backend.inference_jobs.coat_critical import run_inference as coat_critical_run
except Exception:
    coat_critical_run = None

app.config["JSON_SORT_KEYS"] = False
INPUT_FOLDER_ID         = os.getenv("INPUT_FOLDER_ID", "").strip()
INPUT_IMAGE_FOLDER_ID   = os.getenv("INPUT_IMAGE_FOLDER_ID", "").strip()
LASER_20_FOLDER_ID = (
    os.getenv("INPUT_LASER_20_FOLDER_ID")
    or os.getenv("OUTPUT_LASER_FOLDER_ID")  # legacy
    or ""
).strip()
OUTPUT_CODE_FOLDER_ID   = os.getenv("OUTPUT_CODE_FOLDER_ID", "").strip()
BODY_FOLDER_ID          = os.getenv("OUTPUT_BODY_FOLDER_ID", "").strip()
CRITICAL_FOLDER_ID      = os.getenv("OUTPUT_CRITICAL_FOLDER_ID", "").strip()
LASER2_FOLDER_ID        = os.getenv("OUTPUT_LASER2_FOLDER_ID", "").strip()
OUTPUT_OCR_FOLDER_ID    = os.getenv("OUTPUT_OCR_FOLDER_ID", "").strip()
FINAL_CSV_NAME          = os.getenv("FINAL_CSV_NAME", "ocr_results.csv").strip()
OCR_SENTINEL_NAME       = os.getenv("OCR_SENTINEL_NAME", "OCR_DONE.txt").strip()
CROP_SENTINEL_NAME      = os.getenv("CROP_SENTINEL_NAME", "CROP20_DONE.txt").strip()
SPLIT_SENTINEL_NAME     = os.getenv("SPLIT_SENTINEL_NAME", "SPLIT_DONE.txt").strip()
OUTPUT_CLASSIFY_FOLDER_ID = os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", "").strip()
LASER_COMBINED_CSV_ENV = os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv").strip()
CRIT_COMBINED_CSV_ENV  = os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv").strip()
BODY_CSV_ENV = os.getenv("BODY_CSV", "body_coat_results.csv").strip()
PNG_COMPRESS_LEVEL      = int(os.getenv("PNG_COMPRESS_LEVEL", "0"))
PNG_OPTIMIZE            = bool(int(os.getenv("PNG_OPTIMIZE", "0")))
MAX_IMAGES              = int(os.getenv("MAX_IMAGES", "1000000"))
IMG_EXTS                = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_LOCK = threading.Lock()
CROP_LOCK  = threading.Lock()
OCR_LOCK   = threading.Lock()
INSPECT_LASER_LOCK     = threading.Lock()
INSPECT_CRITICAL_LOCK  = threading.Lock()
INSPECT_BODY_LOCK      = threading.Lock()  # 👈 NEW
INSPECT_STATE = {
    "laser":   {"status": "idle", "level": None, "selection": None, "summary": None, "started_at": None, "finished_at": None},
    "critical":{"status": "idle", "level": None, "selection": None, "summary": None, "started_at": None, "finished_at": None},
    "body":    {"status": "idle", "level": None, "selection": None, "summary": None, "started_at": None, "finished_at": None},  # 👈 NEW
}


LASER_CSV = os.getenv("LASER_CSV", "laser_all_models.csv")
CRITICAL_CSV = os.getenv("CRITICAL_CSV", "critical_all_models.csv")
BODY_CSV = os.getenv("BODY_CSV", "body_coat_results.csv")
FINAL_INSPECTION_CSV = os.getenv("FINAL_INSPECTION_CSV", "final_inspection_results.csv")
CONSENSUS_RULE = os.getenv("CONSENSUS_RULE", "majority").strip().lower()
OUTPUT_REPORT_FOLDER_ID = os.getenv("OUTPUT_REPORT_FOLDER_ID")

# ===== Defective Folder IDs from .env =====
DEFECTIVE_LASER_FOLDER_ID    = os.getenv("DEFECTIVE_LASER_FOLDER_ID", "").strip()
DEFECTIVE_CRITICAL_FOLDER_ID = os.getenv("DEFECTIVE_CRITICAL_FOLDER_ID", "").strip()
DEFECTIVE_BODY_FOLDER_ID     = os.getenv("DEFECTIVE_BODY_FOLDER_ID", "").strip()

MANUAL_USE_DEFECT_FOLDERS = True
REWRITE_CONFIRM_DISABLED = False
UNDO_STACK: deque = deque(maxlen=50)
HIDDEN_ROOTS: Set[str] = set()
_ROOT_RE = re.compile(
    r"(.*?_laser)(?:_die_\d{1,2})?_(laser|critical|body)\.(png|jpe?g)$",
    re.IGNORECASE
)
_DIE_RE = re.compile(r"_die_(\d{1,2})_", re.IGNORECASE)
_REGION_RE = re.compile(r"_(laser|critical|body)\.(png|jpe?g)$", re.IGNORECASE)

def _basename(p: str) -> str:
    return os.path.basename(str(p))

def _die_idx(name: str) -> int | None:
    m = _DIE_RE.search(name)
    return int(m.group(1)) if m else None

def _region_of(name: str) -> str | None:
    m = _REGION_RE.search(name)
    return m.group(1).lower() if m else None

def _base_key_of(name: str) -> str:
    # honor your existing rule that base ends at “…_laser”
    b = _basename(name)
    m = _ROOT_RE.search(b)
    if m:
        return m.group(1)
    # fallback: strip `_die_##_...`
    parts = re.split(r"_die_\d{1,2}_", b, flags=re.IGNORECASE, maxsplit=1)
    return parts[0] if parts else os.path.splitext(b)[0]

# --- Label normalization for defect decisions (Body/Laser/Critical) ---
def _norm_label(x: str) -> str:
    """
    Normalize a variety of label spellings/synonyms to: 'Good', 'Defective', or 'N/A'.
    Treat MIXED as Defective (since at least one model flagged a defect).
    """
    s = str(x or "").strip().upper()
    if s in ("", "N/A", "NA", "NONE", "NULL"):
        return "N/A"

    GOOD_SET = {"GOOD", "ALL_GOOD", "OK", "PASS", "PERFECT"}
    DEF_SET  = {"DEFECTIVE", "BAD", "NG", "FAIL", "ALL_DEFECTIVE", "ANY_DEFECT", "MIXED"}

    if s in GOOD_SET:
        return "Good"
    if s in DEF_SET:
        return "Defective"

    # Heuristics for unexpected tokens
    if "GOOD" in s:
        return "Good"
    if "DEFECT" in s:
        return "Defective"

    return "N/A"

import json, re, time
from datetime import datetime

INSPECT_STATE_PATH = Path(".manual_inspection_state.json")   # survives restarts
INSPECT_AUTOSAVE_EVERY = 1                                    # autosave cadence
INSPECT_RESULTS_CSV_NAME = "manual_inspection_results.csv"    # on Drive too

INSPECTION_RESULTS_FOLDER_ID = os.getenv("INSPECTION_RESULTS_FOLDER_ID")

from contextlib import contextmanager

@contextmanager
def _temp_env(**pairs):
    old = {}
    try:
        for k, v in pairs.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _load_manual_state():
    if INSPECT_STATE_PATH.exists():
        with open(INSPECT_STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # normalize types on load
        by_base = st.get("by_base_key") or {}
        comp = st.get("completed_keys") or []
        if isinstance(comp, list):
            comp = set(comp)
        return {
            "by_base_key": by_base,
            "completed_keys": comp,
            "action_log": st.get("action_log") or [],
            "version": st.get("version", 1),
        }
    return {
        "by_base_key": {},
        "completed_keys": set(),
        "action_log": [],
        "version": 1
    }


def _save_manual_state(state):
    # sets are not JSON-serializable
    st = dict(state)
    st["completed_keys"] = sorted(list(state.get("completed_keys", [])))
    with open(INSPECT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def _extract_base_key(name_or_link: str) -> str:
    # If a raw filename was passed, this suffices:
    m = re.search(r"([^/\\]+)", name_or_link)
    fname = m.group(1) if m else name_or_link

    m2 = re.match(r"^(.*?)_laser_die", fname)
    if m2:
        return m2.group(1)

    # Last fallback: strip extension and return head token
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", fname)
    return stem.split("_laser_die")[0]

STATE = _load_manual_state()
STATE["_since_last_autosave"] = 0

# normalize completed_keys to a set in-memory
if isinstance(STATE.get("completed_keys"), list):
    STATE["completed_keys"] = set(STATE["completed_keys"])

def root_of(name: str) -> str:
    """Strip die index and region to get the batch root (…_laser)."""
    m = _ROOT_RE.match(name or "")
    return m.group(1) if m else os.path.splitext(name)[0]


def region_of(name: str) -> str:
    m = _ROOT_RE.match(name or "")
    return (m.group(2).lower() if m else "").strip()

# ---------- CSV voting ----------
def _load_votes_from_drive() -> Dict[str, Dict[str, Dict[str, str]]]:
    try:
        return load_csv_results()
    except Exception:
        # Fallback to local files if needed
        return _load_all_model_csvs()

def _drive_copy_file(service, src_id: str, dst_folder_id: str, new_name: Optional[str] = None) -> Optional[str]:
    try:
        body = {"parents": [dst_folder_id]}
        if new_name:
            body["name"] = new_name
        created = service.files().copy(fileId=src_id, body=body, fields="id").execute()
        return created.get("id")
    except Exception:
        return None

def _pick(row, *keys, default="GOOD"):
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return str(row[k]).strip().upper()
    return default

@app.post("/manual/save_csv")
def manual_save_csv():
    from datetime import datetime
    import io as _io
    import csv as _csv
    import hashlib

    # Normalize completed_keys type before save
    if isinstance(STATE.get("completed_keys"), list):
        STATE["completed_keys"] = set(STATE["completed_keys"])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buf = _io.StringIO()
    buf.write(f"# Manual Inspection Report — generated {now}\n")
    buf.write("# Tool: Visdom\n")
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(["base_key", "total", "good_count", "defective"])
    for base_key, row in sorted((STATE.get("by_base_key") or {}).items()):
        writer.writerow([
            base_key,
            int(row.get("total", 0) or 0),
            int(row.get("good_count", 0) or 0),
            int(row.get("defective", 0) or 0),
        ])
    csv_text = buf.getvalue()
    checksum = hashlib.md5(csv_text.encode("utf-8")).hexdigest()

    # Always write a local copy
    local_path = Path(INSPECT_RESULTS_CSV_NAME)
    with open(local_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    # Upload to Drive if configured
    drive_ok, file_id, drive_error = False, "", None
    if INSPECTION_RESULTS_FOLDER_ID:
        try:
            from backend.drive_utils import upload_text_as_csv
            file_id = upload_text_as_csv(
                text=csv_text,
                filename=INSPECT_RESULTS_CSV_NAME,
                folder_id=INSPECTION_RESULTS_FOLDER_ID,
                replace_existing=True,
            )
            drive_ok = True
        except Exception as e:
            drive_error = str(e)

    # Persist state
    _save_manual_state(STATE)

    return jsonify({
        "ok": True,
        "local_path": str(local_path.resolve()),
        "checksum_md5": checksum,
        "drive_uploaded": drive_ok,
        "drive_file_id": file_id,
        "drive_error": drive_error,
    }), 200

PREPARE_DEFECTS_LOCK = threading.Lock()
PREPARE_DEFECTS_STATE = {
    "status": "idle",
    "counts": {},
    "started_at": None,
    "finished_at": None,
}

def _ensure_folder_ids_present() -> Optional[str]:
    missing = []
    if not DEFECTIVE_LASER_FOLDER_ID:    missing.append("DEFECTIVE_LASER_FOLDER_ID")
    if not DEFECTIVE_CRITICAL_FOLDER_ID: missing.append("DEFECTIVE_CRITICAL_FOLDER_ID")
    if not DEFECTIVE_BODY_FOLDER_ID:     missing.append("DEFECTIVE_BODY_FOLDER_ID")
    return ", ".join(missing) if missing else None

@app.route("/prepare_defects", methods=["POST"])
def prepare_defects():
    svc = authenticate()
    drive_mode = _is_drive(svc)

    # In Drive mode, make sure all destination IDs are present
    if drive_mode:
        miss = _ensure_folder_ids_present()
        if miss:
            return jsonify({"message": f"❌ Missing .env keys: {miss}"}), 400

    if not PREPARE_DEFECTS_LOCK.acquire(blocking=False):
        return jsonify({"message": "⏳ Already preparing defects."}), 429

    def _worker():
        try:
            PREPARE_DEFECTS_STATE.update({
                "status": "running",
                "started_at": time.time(),
                "finished_at": None,
                "counts": {},
            })

            votes_all = _load_votes_from_drive()  # falls back to local CSVs if Drive missing
            if not votes_all:
                PREPARE_DEFECTS_STATE.update({
                    "status": "done",
                    "counts": {"new_candidates": 0, "laser": 0, "critical": 0, "body": 0},
                    "finished_at": time.time(),
                })
                return

            # Build source maps and existing destination name-sets
            if drive_mode:
                src_by_region = {
                    "laser":    {f["name"]: f for f in _drive_list_images(svc, LASER2_FOLDER_ID)},
                    "critical": {f["name"]: f for f in _drive_list_images(svc, CRITICAL_FOLDER_ID)},
                    "body":     {f["name"]: f for f in _drive_list_images(svc, BODY_FOLDER_ID)},
                }
                dst_existing = {
                    "laser":    {f["name"] for f in _drive_list_images(svc, DEFECTIVE_LASER_FOLDER_ID)},
                    "critical": {f["name"] for f in _drive_list_images(svc, DEFECTIVE_CRITICAL_FOLDER_ID)},
                    "body":     {f["name"] for f in _drive_list_images(svc, DEFECTIVE_BODY_FOLDER_ID)},
                }
            else:
                def _map_dir(d: Path) -> Dict[str, Dict]:
                    return {p.name: {"path": p} for p in list_images(d)}
                src_by_region = {
                    "laser":    _map_dir(_REGION_DIRS["laser"]),
                    "critical": _map_dir(_REGION_DIRS["critical"]),
                    "body":     _map_dir(_REGION_DIRS["body"]),
                }
                dst_existing = {
                    "laser":    {p.name for p in list_images(folder(DEFECTIVE_LASER_FOLDER_ID))},
                    "critical": {p.name for p in list_images(folder(DEFECTIVE_CRITICAL_FOLDER_ID))},
                    "body":     {p.name for p in list_images(folder(DEFECTIVE_BODY_FOLDER_ID))},
                }

            # Compute new candidates per region
            to_copy = {"laser": set(), "critical": set(), "body": set()}
            for img_name, regions in votes_all.items():
                for reg in ("laser", "critical", "body"):
                    votes = (regions or {}).get(reg) or {}
                    if not votes:
                        continue
                    is_def = any((str(v).lower() == "defective") for v in votes.values() if v and v != "N/A")
                    if not is_def:
                        continue
                    if img_name not in dst_existing.get(reg, set()):
                        to_copy[reg].add(img_name)

            copied = {"laser": 0, "critical": 0, "body": 0}
            missing_src = {"laser": 0, "critical": 0, "body": 0}

            # Perform copies
            for reg in ("laser", "critical", "body"):
                for img_name in sorted(to_copy[reg]):
                    src = src_by_region.get(reg, {}).get(img_name)
                    if not src:
                        missing_src[reg] += 1
                        continue

                    if drive_mode:
                        dst_folder = (
                            DEFECTIVE_LASER_FOLDER_ID    if reg == "laser" else
                            DEFECTIVE_CRITICAL_FOLDER_ID if reg == "critical" else
                            DEFECTIVE_BODY_FOLDER_ID
                        )
                        if _drive_copy_file(svc, src["id"], dst_folder, new_name=img_name):
                            copied[reg] += 1
                    else:
                        dst_dir = folder(
                            DEFECTIVE_LASER_FOLDER_ID    if reg == "laser" else
                            DEFECTIVE_CRITICAL_FOLDER_ID if reg == "critical" else
                            DEFECTIVE_BODY_FOLDER_ID
                        )
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(src["path"], dst_dir / img_name)
                            copied[reg] += 1
                        except Exception:
                            pass

            PREPARE_DEFECTS_STATE.update({
                "status": "done",
                "counts": {
                    "new_candidates": sum(len(s) for s in to_copy.values()),
                    **copied,
                    "missing_sources": missing_src,
                },
                "finished_at": time.time(),
            })
        except Exception as e:
            PREPARE_DEFECTS_STATE.update({
                "status": "error",
                "counts": {"error": str(e)},
                "finished_at": time.time(),
            })
        finally:
            if PREPARE_DEFECTS_LOCK.locked():
                PREPARE_DEFECTS_LOCK.release()

    # Kick the worker
    threading.Thread(target=_worker, daemon=True).start()

    # Quick preview (non-blocking)
    try:
        votes_all = _load_votes_from_drive()
        if drive_mode:
            dst_existing = {
                "laser":    {f["name"] for f in _drive_list_images(svc, DEFECTIVE_LASER_FOLDER_ID)},
                "critical": {f["name"] for f in _drive_list_images(svc, DEFECTIVE_CRITICAL_FOLDER_ID)},
                "body":     {f["name"] for f in _drive_list_images(svc, DEFECTIVE_BODY_FOLDER_ID)},
            }
        else:
            dst_existing = {
                "laser":    {p.name for p in list_images(folder(DEFECTIVE_LASER_FOLDER_ID))},
                "critical": {p.name for p in list_images(folder(DEFECTIVE_CRITICAL_FOLDER_ID))},
                "body":     {p.name for p in list_images(folder(DEFECTIVE_BODY_FOLDER_ID))},
            }

        preview_new = 0
        for img_name, regions in votes_all.items():
            for reg in ("laser", "critical", "body"):
                votes = (regions or {}).get(reg) or {}
                if not votes:
                    continue
                if any((str(v).lower() == "defective") for v in votes.values() if v and v != "N/A"):
                    if img_name not in dst_existing[reg]:
                        preview_new += 1
                        if preview_new > 20:
                            break
            if preview_new > 20:
                break

        if preview_new == 0:
            return jsonify({"message": "No new images to prepare.", "new_count": 0}), 200
        return jsonify({"message": "🚀 Preparing defect folders...", "new_count": preview_new}), 200
    except Exception:
        return jsonify({"message": "🚀 Preparing defect folders..."}), 200

@app.post("/export/die_reports")
def export_die_reports():
    import csv, re, os

    # Inputs (respect .env) — these already exist in app config
    laser_csv    = (folder(CSV_DIR) / LASER_CSV).resolve()
    critical_csv = (folder(CSV_DIR) / CRITICAL_CSV).resolve()
    body_csv     = (folder(CSV_DIR) / BODY_CSV).resolve()

    # === NEW: OCR helper (local-first) =======================================
    CORE_SUFFIX_RE = re.compile(r"_(laser|critical|body|code)$", re.IGNORECASE)

    def _stem(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    def _core_base(stem: str) -> str:
        # strip trailing _laser/_critical/_body/_code to get the common base
        return CORE_SUFFIX_RE.sub("", stem)

    def _booly(x) -> bool:
        s = str(x or "").strip().lower()
        return s in ("1", "true", "yes", "y")

    def load_ocr_map() -> dict[str, str]:
        ocr_dir = folder(OUTPUT_OCR_FOLDER_ID)
        ocr_path = (ocr_dir / FINAL_CSV_NAME).resolve()
        out: dict[str, str] = {}
        if not ocr_path.exists():
            return out
        with open(ocr_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                img = (row.get("image") or "").strip()
                if not img:
                    continue
                core = _core_base(_stem(img))
                agree = _booly(row.get("agree"))
                pred_crnn = (row.get("pred_crnn") or "").strip()
                actual = (row.get("actual_code") or "").strip()
                if agree and pred_crnn:
                    code = pred_crnn
                elif (not agree) and actual:
                    code = actual
                else:
                    code = "CHECK"
                out[core] = code
        return out

    def read_map(p: Path, region: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not p.exists():
            return out
        with open(p, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                img = row.get("image") or row.get("img") or row.get("filename") or row.get("file") or ""
                if not img:
                    continue
                bname = _basename(img)
                base  = _base_key_of(bname)       # e.g. "..._laser"
                die   = _die_idx(bname)
                if die is None:
                    continue

                d = out.setdefault((base, die), {"base": base, "die": die})
                if region in ("laser", "critical"):
                    d[f"{region}_maxvit"]  = _norm_label(row.get("MaxViT") or row.get("maxvit") or row.get("MaxViT_pred"))
                    d[f"{region}_coat"]    = _norm_label(row.get("CoaT")  or row.get("coat")   or row.get("CoaT_pred")  or row.get("classification"))
                    d[f"{region}_swinv2"]  = _norm_label(row.get("SwinV2")or row.get("swinv2") or row.get("SwinV2_pred"))
                    d[f"{region}_cons"]    = _norm_label(row.get("consensus") or row.get("Consensus") or row.get("final") or row.get("label"))
                else:  # body often has only one model
                    d["body_coat"] = _norm_label(row.get("CoaT") or row.get("coat") or row.get("classification") or row.get("label"))
                    d["body_cons"] = d.get("body_coat", "N/A")
        return out

    laser_map    = read_map(Path(laser_csv),   "laser")
    critical_map = read_map(Path(critical_csv),"critical")
    body_map     = read_map(Path(body_csv),    "body")
    ocr_map = load_ocr_map()

    def code_for_base(base_with_region: str) -> str:
        core = _core_base(base_with_region)
        return ocr_map.get(core, "CHECK")
    keys = set(laser_map.keys()) | set(critical_map.keys()) | set(body_map.keys())
    die_rows: list[dict] = []
    basecodes: dict[str, list[str]] = {}

    for key in sorted(keys, key=lambda t: (t[0], t[1])):  # (base, die)
        base, die = key
        lm = laser_map.get(key, {})
        cm = critical_map.get(key, {})
        bm = body_map.get(key, {})

        # collect votes (defaults)
        row = {
            "base_name": base,                   # e.g. "..._laser"
            "die_index": die,
            "die_image": f"{base}_die_{die}",    # matches your naming
            "laser_maxvit":   lm.get("laser_maxvit","N/A"),
            "laser_coat":     lm.get("laser_coat","N/A"),
            "laser_swinv2":   lm.get("laser_swinv2","N/A"),
            "laser_cons":     lm.get("laser_cons","N/A"),
            "critical_maxvit": cm.get("critical_maxvit","N/A"),
            "critical_coat":   cm.get("critical_coat","N/A"),
            "critical_swinv2": cm.get("critical_swinv2","N/A"),
            "critical_cons":   cm.get("critical_cons","N/A"),
            "body_coat":       bm.get("body_coat","N/A"),
            "body_cons":       bm.get("body_cons","N/A"),
            # === NEW: put the resolved code on every die row =================
            "code":            code_for_base(base),
        }

        # overall per-die decision
        g_l = (row["laser_cons"]    == "Good") or (row["laser_cons"]    == "N/A")  # treat missing as neutral
        g_c = (row["critical_cons"] == "Good") or (row["critical_cons"] == "N/A")
        g_b = (row["body_cons"]     == "Good") or (row["body_cons"]     == "N/A")

        overall = "Perfect" if (g_l and g_c and g_b) else "Defective"
        row["overall"] = overall
        die_rows.append(row)

        # defect code string per base (e.g., b5 c4 l3)
        codes = basecodes.setdefault(base, [])
        if not g_b and row["body_cons"] != "N/A":      codes.append(f"b{die}")
        if not g_c and row["critical_cons"] != "N/A":  codes.append(f"c{die}")
        if not g_l and row["laser_cons"] != "N/A":     codes.append(f"l{die}")

    # Write CSV #1 — all models by die (now includes "code")
    die_csv_path = folder(CSV_DIR) / "die_all_models.csv"
    with open(die_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "base_name","die_index","die_image",
            "laser_maxvit","laser_coat","laser_swinv2","laser_cons",
            "critical_maxvit","critical_coat","critical_swinv2","critical_cons",
            "body_coat","body_cons",
            "code",                # <=== NEW
            "overall",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in die_rows:
            w.writerow(r)

    # Write CSV #2 — base summary GOOD/DEFECTIVE + codes + Code
    base_csv_path = folder(CSV_DIR) / "base_summary.csv"
    with open(base_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "status", "defect_codes", "code"])  # <=== NEW header
        all_bases = sorted(basecodes.keys() | {r["base_name"] for r in die_rows})
        for base in all_bases:
            defect_marks = "".join(sorted(basecodes.get(base, []), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0)))
            status = "GOOD" if not defect_marks else "DEFECTIVE"
            w.writerow([base, status, defect_marks, code_for_base(base)])  # <=== include Code

    return jsonify({
        "ok": True,
        "message": "✅ Built die_all_models.csv and base_summary.csv (with Code).",
        "outputs": {
            "die_all_models": str(die_csv_path.resolve()),
            "base_summary": str(base_csv_path.resolve()),
        }
    }), 200

# Frontend (main.js) calls this alias:
@app.route("/prepare_defect_folders", methods=["POST"])
def prepare_defect_folders():
    return prepare_defects()

# ---------------- Proxy Helper ----------------
def make_proxy(image_path: str) -> str:
    if image_path.startswith("http"):
        # For Google Drive or any remote URL
        return f"/proxy?url={image_path}"
    else:
        # For locally saved files in static/uploads
        return url_for("static", filename=f"uploads/{os.path.basename(image_path)}")

@app.route("/manual/confirm_toggle", methods=["POST"])
def manual_confirm_toggle():
    global REWRITE_CONFIRM_DISABLED
    payload = request.get_json(silent=True) or {}
    REWRITE_CONFIRM_DISABLED = bool(payload.get("disable"))
    return jsonify({"ok": True, "disabled": REWRITE_CONFIRM_DISABLED})

STATE_LOCK = threading.Lock()
def _push_undo(entry):
    with STATE_LOCK:
        UNDO_STACK.append(entry)
        if len(UNDO_STACK) > 50:
            UNDO_STACK.popleft()

@app.route("/manual/undo", methods=["POST"])
def manual_undo():
    if not UNDO_STACK:
        return jsonify({"ok": False, "message": "Nothing to undo"}), 200

    action = UNDO_STACK.pop()
    if action["type"] == "decision":
        base_key = action["base_key"]
        rows = _load_csv(MANUAL_VERIFY_CSV)
        prev = action["prev"]
        if prev is None:
            rows.pop(base_key, None)
        else:
            rows[base_key] = prev
        _save_csv(MANUAL_VERIFY_CSV, rows)
        if action.get("removed_from_queue"):
            HIDDEN_ROOTS.discard(base_key)
    return jsonify({"ok": True})

@app.route("/manual/decision", methods=["POST"])
def manual_decision():
    data = request.get_json(force=True)
    base_key = root_of(data["base_key"])
    per_image_decision = (data.get("decision") or "").upper()  # GOOD | DEFECTIVE
    region = (data.get("region") or "")
    last_image = data.get("last_image") or ""

    # Track per-base progress
    prog = _progress_load()
    p = prog.get(base_key) or {"base_key": base_key, "total": 0, "good_count": 0, "defective": "0"}

    # Count total siblings by scanning current defect folders for this base
    svc = authenticate()
    sibs = 0
    for reg, fid in (("laser", DEFECTIVE_LASER_FOLDER_ID),
                     ("critical", DEFECTIVE_CRITICAL_FOLDER_ID),
                     ("body", DEFECTIVE_BODY_FOLDER_ID)):
        for f in _drive_list_images(svc, fid):
            if root_of(f["name"]) == base_key:
                sibs += 1
    if sibs == 0:
        sibs = max(1, int(p.get("total") or 0))
    if not p.get("total"):
        p["total"] = str(sibs)

    # Decisions CSV
    rows = _load_csv(MANUAL_VERIFY_CSV)
    prev = rows.get(base_key)

    removed = False
    if per_image_decision == "DEFECTIVE":
        rows[base_key] = {
            "base_key": base_key,
            "last_image": last_image,
            "region": region,
            "decision": "DEFECTIVE",
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_csv(MANUAL_VERIFY_CSV, rows)
        HIDDEN_ROOTS.add(base_key)
        removed = True
        p["defective"] = "1"
    else:
        p["good_count"] = str(int(p.get("good_count") or 0) + 1)
        if int(p["good_count"]) >= int(p["total"]) and p.get("defective") != "1":
            rows[base_key] = {
                "base_key": base_key,
                "last_image": last_image,
                "region": region,
                "decision": "GOOD",
                "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_csv(MANUAL_VERIFY_CSV, rows)
            HIDDEN_ROOTS.add(base_key)
            removed = True

    prog[base_key] = p
    _progress_save(prog)
    _push_undo({"type": "decision", "base_key": base_key, "prev": prev, "removed_from_queue": removed})
    return jsonify({"ok": True})


@app.route("/image/<file_id>")
def proxy_image(file_id: str):
    try:
        svc = authenticate()
        if not _is_drive(svc):
            return jsonify({"error": "Drive is not available in LOCAL mode."}), 400

        meta = svc.files().get(fileId=file_id, fields="name,mimeType").execute()
        mime = meta.get("mimeType") or "application/octet-stream"
        data = _drive_download_bytes(svc, file_id)
        return send_file(
            io.BytesIO(data),
            mimetype=mime,
            as_attachment=False,
            download_name=meta.get("name", "image"),
        )
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {e}"}), 404



def parse_votes_from_row(row: Dict[str,str], region: str) -> Dict[str,str]:
    if region in ("laser", "critical"):
        return {
            "maxvit": _pick(row, "maxvit_label", "MaxViT"),
            "coat":   _pick(row, "coat_label",   "CoaT"),
            "swinv2": _pick(row, "swinv2_label", "SwinV2"),
        }
    elif region == "body":
        return {"maxvit": "N/A", "coat": _pick(row, "classification", "CoaT"), "swinv2": "N/A"}
    return {}

def _load_all_model_csvs() -> Dict[str, Dict[str, Dict[str, str]]]:
    rows: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _norm_label(v: Optional[str]) -> str:
        s = (v or "").strip().upper()
        if s in ("",):
            return "GOOD"
        # Common synonyms / legacy values
        if s in ("BAD", "DEF", "DEFECT", "NG", "ALL_DEFECTIVE", "ANY_DEFECT"):
            return "DEFECTIVE"
        if s in ("OK", "ALL_GOOD"):
            return "GOOD"
        # Expected: GOOD or DEFECTIVE
        return s

    def _basename_any(p: str) -> str:
        # Works for both POSIX and Windows-style paths
        p = (p or "").strip().replace("\\", "/")
        return p.rsplit("/", 1)[-1]

    def load_one(path: str, region: str):
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                raw = (row.get("image") or row.get("filename") or "").strip()
                if not raw:
                    continue
                img = _basename_any(raw)  # <<< key by basename so it matches filesystem/Drive names

                if img not in rows:
                    rows[img] = {"laser": {}, "critical": {}, "body": {}}

                # case-insensitive column access
                lower = {str(k).lower(): (v if v is not None else "") for k, v in row.items()}

                if region in ("laser", "critical"):
                    maxvit = row.get("MaxViT") or lower.get("maxvit") or lower.get("maxvit_label")
                    coat   = row.get("CoaT")   or lower.get("coat")   or lower.get("coat_label")
                    swinv2 = row.get("SwinV2") or lower.get("swinv2") or lower.get("swinv2_label")

                    rows[img][region] = {
                        "maxvit": _norm_label(maxvit) or "GOOD",
                        "coat":   _norm_label(coat)   or "GOOD",
                        "swinv2": _norm_label(swinv2) or "GOOD",
                    }
                else:
                    # Body CSVs are often single-model (CoaT) or may use a generic 'classification' column
                    coat = row.get("CoaT") or lower.get("coat") or lower.get("classification")
                    rows[img][region] = {"coat": _norm_label(coat) or "GOOD"}

    load_one((CSV_DIR / LASER_COMBINED_CSV_ENV).as_posix(), "laser")
    load_one((CSV_DIR / CRIT_COMBINED_CSV_ENV ).as_posix(), "critical")
    load_one((CSV_DIR / BODY_CSV_ENV         ).as_posix(), "body")
    return rows

def load_csv_results() -> Dict[str, Dict[str, Dict[str, str]]]:
    svc = authenticate()
    if not _is_drive(svc):
        return _load_all_model_csvs()

    def _norm(s: Optional[str]) -> str:
        v = (s or "").strip().upper()
        if v in ("",): return "GOOD"
        if v in ("BAD","DEF","DEFECT","NG","ALL_DEFECTIVE","ANY_DEFECT"): return "DEFECTIVE"
        if v in ("OK","ALL_GOOD"): return "GOOD"
        return v

    results: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _basename_any(p: str) -> str:
        p = (p or "").strip().replace("\\", "/")
        return p.rsplit("/", 1)[-1]

    def load_from_drive(filename: str, kind: str):
        files = _drive_find_by_name(svc, OUTPUT_CLASSIFY_FOLDER_ID, filename) if OUTPUT_CLASSIFY_FOLDER_ID else []
        if not files: return
        try:
            data = _drive_download_bytes(svc, files[0]["id"]).decode("utf-8", "ignore")
            rdr = csv.DictReader(io.StringIO(data))
            for row in rdr:
                raw = (row.get("image") or row.get("filename") or "").strip()
                if not raw: continue
                img = _basename_any(raw)
                results.setdefault(img, {"laser": {}, "critical": {}, "body": {}})
                if kind in ("laser","critical"):
                    results[img][kind] = {
                        "maxvit": _norm(row.get("MaxViT") or row.get("maxvit_label")),
                        "coat":   _norm(row.get("CoaT")   or row.get("coat_label")),
                        "swinv2": _norm(row.get("SwinV2") or row.get("swinv2_label")),
                    }
                elif kind == "body":
                    results[img]["body"] = {"coat": _norm(row.get("CoaT") or row.get("classification"))}
        except Exception:
            pass

    load_from_drive(LASER_COMBINED_CSV_ENV, "laser")
    load_from_drive(CRIT_COMBINED_CSV_ENV,  "critical")
    load_from_drive(BODY_CSV_ENV,           "body")
    return results

def severity_count(votes: Dict[str, str]) -> int:
    return sum(1 for v in votes.values() if v.lower() == "defective")


def overall_status(votes: Dict[str, str]) -> str:
    return "Defective" if any(v.lower() == "defective" for v in votes.values()) else "Good"

from pathlib import Path
from backend.paths import folder, IMAGES_DIR
from backend.local_store import list_images

def _drive_list_images(service, folder_id_or_path: str) -> List[Dict]:
    if service is None or not hasattr(service, "files"):
        base = folder(str(folder_id_or_path))
        results: List[Dict] = []
        for p in list_images(base):
            try:
                rel = p.relative_to(IMAGES_DIR).as_posix()
            except Exception:
                rel = p.as_posix()
            results.append({"name": p.name, "id": "", "path": rel})
        results.sort(key=lambda d: d["name"])
        return results

    # ---------- Drive mode (unchanged behaviour) ----------
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    results = []
    page_token = None
    q = f"'{folder_id_or_path}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType, size)"
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=page_token).execute()
        for f in resp.get("files", []):
            name = f.get("name", "")
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                results.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token or len(results) >= MAX_IMAGES:
            break
    return results[:MAX_IMAGES]

def _drive_list_names(service, folder_id: str) -> Set[str]:
    names: Set[str] = set()
    if not folder_id:
        return names

    # Local-mode fallback
    if service is None or not hasattr(service, "files"):
        base = folder(str(folder_id))  # maps env path/ID to a local Path
        try:
            # include images
            for p in list_images(base):
                names.add(p.name)
            # include any non-image files too (e.g., sentinels like *_DONE.txt)
            for p in base.glob("*"):
                if p.is_file():
                    names.add(p.name)
        except Exception:
            pass
        return names

    # Drive mode
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(name)"
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=page_token).execute()
        for f in resp.get("files", []):
            n = f.get("name")
            if n:
                names.add(n)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return names

def _drive_file_exists(service, folder_id: str, exact_name: str) -> bool:
    if not folder_id or not exact_name:
        return False
    # Local-mode
    if service is None or not hasattr(service, "files"):
        base = folder(str(folder_id))
        return (base / exact_name).exists()
    # Drive mode
    safe_name = exact_name.replace("'", "\\'")
    q = f"'{folder_id}' in parents and trashed=false and name='{safe_name}'"
    resp = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
    return len(resp.get("files", [])) > 0

def _drive_find_by_name(service, folder_id: str, exact_name: str) -> List[Dict]:
    if not folder_id or not exact_name:
        return []
    # Local-mode
    if service is None or not hasattr(service, "files"):
        base = folder(str(folder_id))
        p = base / exact_name
        return [{"id": "", "name": exact_name, "path": p.as_posix()}] if p.exists() else []
    # Drive mode
    safe = exact_name.replace("'", "\\'")
    q = f"'{folder_id}' in parents and trashed=false and name='{safe}'"
    resp = service.files().list(q=q, fields="files(id,name)", pageSize=100).execute()
    return resp.get("files", []) or []

def _drive_delete_by_name(service, folder_id: str, exact_name: str) -> int:
    files = _drive_find_by_name(service, folder_id, exact_name)
    deleted = 0
    for f in files:
        try:
            service.files().delete(fileId=f["id"]).execute()
            deleted += 1
        except Exception:
            pass
    return deleted
def _drive_upload_text(service, folder_id: str, name: str, text: str) -> str:
    bio = io.BytesIO(text.encode("utf-8"))
    media = MediaIoBaseUpload(bio, mimetype="text/plain", resumable=False)
    meta = {"name": name, "parents": [folder_id], "mimeType": "text/plain"}
    created = service.files().create(body=meta, media_body=media, fields="id").execute()
    return created.get("id", "")
def _drive_download_bytes(service, file_id: str) -> bytes:
    buf = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request, chunksize=1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()
def _drive_download_text(service, file_id: str) -> str:
    return _drive_download_bytes(service, file_id).decode("utf-8", errors="ignore")
def _drive_upload_png(service, folder_id: str, name: str, data: bytes) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="image/png", resumable=True)
    file_meta = {"name": name, "parents": [folder_id], "mimeType": "image/png"}
    created = service.files().create(body=file_meta, media_body=media, fields="id").execute()
    return created.get("id")
def _bases_from_inputs(service, folder_id: str) -> Set[str]:
    if not folder_id:
        return set()
    files = _drive_list_images(service, folder_id)
    return {os.path.splitext(f["name"])[0] for f in files}
def _bases_with_code(service, code_folder_id: str) -> Set[str]:
    if not code_folder_id:
        return set()
    names = _drive_list_names(service, code_folder_id)
    out: Set[str] = set()
    for n in names:
        ln = n.lower()
        if ln.endswith("_code.png") or ln.endswith("_code.jpg") or ln.endswith("_code.jpeg"):
            stem, _ = os.path.splitext(n)
            if stem.lower().endswith("_code"):
                out.add(stem[:-5])
    return out

def _drive_list_code_names(service, folder_id: str) -> Set[str]:
    names: Set[str] = set()
    suff = {".png", ".jpg", ".jpeg"}

    # ----- Local mode ---------------------------------------------------------
    if not _is_drive(service):
        local = (OUTPUT_CODE_FOLDER_ID or "").strip()
        if not local:
            return names
        base = folder(local)
        if base and base.exists():
            for p in base.iterdir():
                if p.is_file():
                    n = p.name
                    ln = n.lower()
                    if any(ln.endswith(s) for s in suff) and ln.endswith(("_code.png", "_code.jpg", "_code.jpeg")):
                        names.add(n.strip())
        return names

    # ----- Drive mode ---------------------------------------------------------
    if not folder_id:
        return names
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(name)"
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=page_token).execute()
        for f in (resp.get("files") or []):
            n = (f.get("name") or "").strip()
            ln = n.lower()
            if any(ln.endswith(s) for s in suff) and ln.endswith(("_code.png", "_code.jpg", "_code.jpeg")):
                names.add(n)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return names

def _final_csv_images(service) -> Set[str]:
    # Local-mode: look for FINAL_CSV_NAME under CSV_DIR
    if service is None or not hasattr(service, "files"):
        path = (CSV_DIR / FINAL_CSV_NAME) if FINAL_CSV_NAME else None
        if not path or not path.exists():
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {
                    (row.get("image") or "").strip()
                    for row in csv.DictReader(f)
                    if (row.get("image") or "").strip()
                }
        except Exception:
            return set()

    # Drive mode (existing behavior)
    if not OUTPUT_OCR_FOLDER_ID or not FINAL_CSV_NAME:
        return set()
    files = _drive_find_by_name(service, OUTPUT_OCR_FOLDER_ID, FINAL_CSV_NAME)
    if not files:
        return set()
    try:
        data = _drive_download_bytes(service, files[0]["id"]).decode("utf-8", errors="ignore")
        imgs: Set[str] = set()
        for row in csv.DictReader(io.StringIO(data)):
            img = (row.get("image") or "").strip()
            if img:
                imgs.add(img)
        return imgs
    except Exception:
        return set()

def _to_png_bytes(im: Image.Image) -> bytes:
    out = io.BytesIO()
    kwargs = {"compress_level": PNG_COMPRESS_LEVEL, "optimize": PNG_OPTIMIZE}
    icc = im.info.get("icc_profile")
    if icc:
        kwargs["icc_profile"] = icc
    im.save(out, format="PNG", **kwargs)
    return out.getvalue()

def _make_crops(src: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
    w, h = src.width, src.height
    x30 = int(round(0.30 * w))
    y20 = int(round(0.20 * h))
    body = src.crop((x30, 0, w, h))
    critical = src.crop((0, 0, x30, y20))
    laser2 = src.crop((0, y20, x30, h))
    return body, critical, laser2

# === REPLACE START: _process_one (robust split for odd/CMYK/corrupt inputs) ==
def _process_one(service, file_dict: Dict) -> Dict:
    name = file_dict["name"]
    base, _ = os.path.splitext(name)

    # ---------- Local mode ----------
    if not _is_drive(service):
        rel = (file_dict.get("path") or name)
        src_path = (IMAGES_DIR / rel)

        # guard against zero-byte / corrupt inputs early
        if (not src_path.exists()) or (src_path.stat().st_size < 64):
            raise ValueError("empty/too-small image")

        with Image.open(src_path) as im:
            # fix orientation and normalize mode
            im = ImageOps.exif_transpose(im)
            src = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            body, critical, laser2 = _make_crops(src)

        # Write PNGs to local output folders
        body_dir     = folder(BODY_FOLDER_ID)
        critical_dir = folder(CRITICAL_FOLDER_ID)
        laser2_dir   = folder(LASER2_FOLDER_ID)
        body_dir.mkdir(parents=True, exist_ok=True)
        critical_dir.mkdir(parents=True, exist_ok=True)
        laser2_dir.mkdir(parents=True, exist_ok=True)

        body_fp     = body_dir     / f"{base}_body.png"
        critical_fp = critical_dir / f"{base}_critical.png"
        laser2_fp   = laser2_dir   / f"{base}_laser.png"
        body.save(str(body_fp), format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=PNG_OPTIMIZE)
        critical.save(str(critical_fp), format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=PNG_OPTIMIZE)
        laser2.save(str(laser2_fp), format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=PNG_OPTIMIZE)

        return {"image": name, "body_path": str(body_fp), "critical_path": str(critical_fp), "laser2_path": str(laser2_fp)}

    # ---------- Drive mode ----------
    fid = file_dict["id"]
    raw = _drive_download_bytes(service, fid)
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im)
        src = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        body, critical, laser2 = _make_crops(src)
        body_bytes     = _to_png_bytes(body)
        critical_bytes = _to_png_bytes(critical)
        laser2_bytes   = _to_png_bytes(laser2)

    body_name     = f"{base}_body.png"
    critical_name = f"{base}_critical.png"
    laser2_name   = f"{base}_laser.png"
    body_id     = _drive_upload_png(service, BODY_FOLDER_ID, body_name, body_bytes)
    critical_id = _drive_upload_png(service, CRITICAL_FOLDER_ID, critical_name, critical_bytes)
    laser2_id   = _drive_upload_png(service, LASER2_FOLDER_ID, laser2_name, laser2_bytes)
    return {"image": name, "body_id": body_id, "critical_id": critical_id, "laser2_id": laser2_id}
# === REPLACE END: _process_one ================================================

def _split_todo_list(service) -> Tuple[List[Dict], Dict[str, int]]:
    inputs = _drive_list_images(service, LASER_20_FOLDER_ID)
    body_names     = _drive_list_names(service, BODY_FOLDER_ID)
    critical_names = _drive_list_names(service, CRITICAL_FOLDER_ID)
    laser2_names   = _drive_list_names(service, LASER2_FOLDER_ID)
    todo: List[Dict] = []
    for f in inputs:
        base, _ = os.path.splitext(f["name"])
        b = f"{base}_body.png"
        c = f"{base}_critical.png"
        l = f"{base}_laser.png"
        if not (b in body_names and c in critical_names and l in laser2_names):
            todo.append(f)
    dbg = {"inputs": len(inputs), "body": len(body_names), "critical": len(critical_names), "laser2": len(laser2_names), "todo": len(todo)}
    return todo, dbg

def _split_worker(files_to_process: List[Dict], service, total_expected: int):
    try:
        start = time.time()
        ok, errs = 0, []
        for f in files_to_process:
            try:
                _ = _process_one(service, f)
                ok += 1
                if ok % 25 == 0:
                    print(f"✅ Split: {ok}/{len(files_to_process)}")
            except Exception as e:
                errs.append({"file": f.get("name"), "error": str(e)})
                print(f"❌ Split error on {f.get('name')}: {e}")
        dur = time.time() - start
        print(f"🏁 Split finished: {ok}/{len(files_to_process)} ok, {len(errs)} errors in {dur:.1f}s (inputs={total_expected})")

        # Sentinel
        try:
            if LASER2_FOLDER_ID and SPLIT_SENTINEL_NAME:
                if _is_drive(service):
                    _drive_upload_text(
                        service,
                        LASER2_FOLDER_ID,
                        SPLIT_SENTINEL_NAME,
                        f"split_count={ok}, errors={len(errs)}, finished_at={time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    # local file sentinel
                    out_dir = folder(LASER2_FOLDER_ID)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / SPLIT_SENTINEL_NAME).write_text(
                        f"split_count={ok}, errors={len(errs)}, finished_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
                        encoding="utf-8"
                    )
        except Exception:
            pass
    finally:
        SPLIT_LOCK.release()


def _crop_worker():
    try:
        if not run_crop_drive_job:
            print("❌ YOLO crop job is not wired (backend.inference_jobs.yolo_crop missing).")
            return
        run_crop_drive_job()
        try:
            if OUTPUT_CODE_FOLDER_ID and CROP_SENTINEL_NAME:
                svc = authenticate()
                _drive_upload_text(svc, OUTPUT_CODE_FOLDER_ID, CROP_SENTINEL_NAME, time.strftime("crop_done_at=%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
    finally:
        CROP_LOCK.release()

def _ocr_worker():
    try:
        if not run_ocr_pipeline_job:
            print("❌ OCR pipeline is not wired (backend.automation missing).")
            return
        run_ocr_pipeline_job()
        try:
            if OUTPUT_OCR_FOLDER_ID and OCR_SENTINEL_NAME:
                svc = authenticate()
                _drive_upload_text(svc, OUTPUT_OCR_FOLDER_ID, OCR_SENTINEL_NAME, time.strftime("ocr_done_at=%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
    finally:
        OCR_LOCK.release()
_ALLOWED_MODELS = ("swinv2", "coat", "maxvit")

def _normalize_models(level: str, selection: Optional[Dict]) -> List[str]:
    level = (level or "max").strip().lower()
    selection = selection or {}

    if level == "mini":
        chosen = str(selection.get("mini", "") or "").strip().lower()
        models = [chosen] if chosen in _ALLOWED_MODELS else ["maxvit"]
    elif level == "moderate":
        arr = selection.get("moderate") or []
        arr = [str(x).strip().lower() for x in arr if str(x).strip()]
        pref_order = {"maxvit": 0, "coat": 1, "swinv2": 2}
        arr = [m for m in arr if m in _ALLOWED_MODELS]
        arr = sorted(dict.fromkeys(arr), key=lambda m: pref_order[m])
        if len(arr) != 2:
            models = ["coat", "maxvit"]
        else:
            models = arr
    else:
        models = ["swinv2", "coat", "maxvit"]

    order = {"maxvit": 0, "coat": 1, "swinv2": 2}
    models = sorted(models, key=lambda m: order[m])
    return models

def _classify_region(region: str, models: List[str]) -> Dict:
    region = (region or "").strip().lower()
    if region not in ("laser", "critical", "body"):
        raise ValueError(f"Unknown region: {region}")

    # Combined CSV filenames (defaults)
    LASER_COMBINED = os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv")
    CRIT_COMBINED  = os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv")
    BODY_CSV_NAME  = os.getenv("BODY_CSV",           "body_coat_results.csv")

    # Local runners
    laser_runners = {
        "maxvit":  run_laser_infer_maxvit,
        "coat":    run_laser_infer_coat,
        "swinv2":  run_laser_infer_swinv2,
    }
    critical_runners = {
        "maxvit":  run_crit_infer_maxvit,
        "coat":    run_crit_infer_coat,
        "swinv2":  run_crit_infer_swinv2,
    }
    body_runner = run_body_inference  # single model (CoaT)

    per_model_csvs: Dict[str, str] = {}
    in_folder = _region_input_folder(region)  # Drive folder ID or local path/env alias

    if region == "laser":
        for m in models:
            fn = laser_runners.get(m)
            if callable(fn):
                try:
                    # make sure runners see INPUT_FOLDER_ID
                    with _temp_env(INPUT_FOLDER_ID=in_folder):
                        res = fn()
                    cp = (res or {}).get("csv_path") if isinstance(res, dict) else (res if isinstance(res, str) else "")
                    if cp:
                        per_model_csvs[m] = cp
                except Exception as e:
                    per_model_csvs[m] = f"ERROR: {e}"

        merged = merge_laser_compare(selected_models=models)
        combined_csv = (merged or {}).get("combined_csv") or str(CSV_DIR / LASER_COMBINED)
        compare_csv  = (merged or {}).get("compare_csv")
        return {
            "region": "laser",
            "selected_models": models,
            "csv_paths": per_model_csvs,
            "combined_csv": combined_csv,
            "compare_csv": compare_csv,
        }

    if region == "critical":
        for m in models:
            fn = critical_runners.get(m)
            if callable(fn):
                try:
                    with _temp_env(INPUT_FOLDER_ID=in_folder):
                        res = fn()
                    cp = (res or {}).get("csv_path") if isinstance(res, dict) else (res if isinstance(res, str) else "")
                    if cp:
                        per_model_csvs[m] = cp
                except Exception as e:
                    per_model_csvs[m] = f"ERROR: {e}"

        merged = merge_critical_compare(selected_models=models)
        combined_csv = (merged or {}).get("combined_csv") or str(CSV_DIR / CRIT_COMBINED)
        compare_csv  = (merged or {}).get("compare_csv")
        return {
            "region": "critical",
            "selected_models": models,
            "csv_paths": per_model_csvs,
            "combined_csv": combined_csv,
            "compare_csv": compare_csv,
        }

    # Body (single CoaT)
    if not callable(body_runner):
        raise RuntimeError("Body classifier is not wired.")

    with _temp_env(INPUT_FOLDER_ID=in_folder):
        res = body_runner()
    cp = (res or {}).get("csv_path") if isinstance(res, dict) else (res if isinstance(res, str) else "")

    return {
        "region": "body",
        "selected_models": ["coat"],
        "csv_paths": {"coat": cp} if cp else {},
        "combined_csv": cp or str(CSV_DIR / BODY_CSV_NAME),
        "compare_csv": None,
    }

def _region_input_folder(region: str) -> str:
    if region == "laser":
        return LASER2_FOLDER_ID
    if region == "critical":
        return CRITICAL_FOLDER_ID
    if region == "body":
        return BODY_FOLDER_ID
    return ""

def _region_combined_csv_name(region: str) -> str:
    if region == "laser":
        return LASER_COMBINED_CSV_ENV
    if region == "critical":
        return CRIT_COMBINED_CSV_ENV
    if region == "body":
        return BODY_CSV_ENV
    return "unknown.csv"

def _drive_csv_images(service, folder_id: str, filename: str) -> Set[str]:
    files = _drive_find_by_name(service, folder_id, filename)
    if not files:
        return set()
    try:
        data = _drive_download_bytes(service, files[0]["id"]).decode("utf-8", "ignore")
        return {
            (row.get("image") or "").strip()
            for row in csv.DictReader(io.StringIO(data))
            if (row.get("image") or "").strip()
        }
    except Exception:
        return set()

from backend.paths import folder, CSV_DIR
from backend.local_store import read_csv

# Map the three regions to local directories (used when service is None)
_REGION_DIRS = {
    "laser":    folder(os.getenv("OUTPUT_LASER2_FOLDER_ID",   "data/images/regions/laser")),
    "critical": folder(os.getenv("OUTPUT_CRITICAL_FOLDER_ID", "data/images/regions/critical")),
    "body":     folder(os.getenv("OUTPUT_BODY_FOLDER_ID",     "data/images/regions/body")),
}

# === PATCH ADD: safe relative helper for /local/ proxy ===
def _rel_under_images(p: Path) -> Optional[str]:
    """
    Return p relative to IMAGES_DIR if (and only if) it's inside IMAGES_DIR.
    Otherwise return None so callers can skip it (prevents /local/ 403s).
    """
    try:
        root = IMAGES_DIR.resolve()
        return p.resolve().relative_to(root).as_posix()
    except Exception:
        return None

# === REPLACE START: _region_new_images_exist =================================
def _region_new_images_exist(region: str) -> Tuple[bool, Dict]:
    """
    Are there region images that are NOT yet present in the region's combined CSV?
    Works in LOCAL and DRIVE modes and always compares by basename.
    """
    region = (region or "").strip().lower()
    svc = authenticate()
    drive_mode = _is_drive(svc)

    folder_id_or_path = _region_input_folder(region)
    if not folder_id_or_path:
        return False, {"region": region, "reason": "missing folder config"}

    csv_name = _region_combined_csv_name(region)

    # --- collect current image names (basenames) ---
    current: Set[str] = set()
    if drive_mode:
        try:
            for f in _drive_list_images(svc, folder_id_or_path):
                n = (f.get("name") or "").strip()
                if n:
                    current.add(n)
        except Exception as e:
            return False, {"region": region, "error": f"drive list failed: {e}"}
    else:
        base = folder(folder_id_or_path)
        try:
            for p in list_images(base):
                current.add(p.name)
        except Exception:
            # fallback: filter name listing
            current = {
                n for n in _drive_list_names(svc, str(base))
                if os.path.splitext(n)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
            }

    # --- collect 'seen' names from the COMBINED CSV (basenames) ---
    seen: Set[str] = set()
    try:
        if drive_mode:
            if OUTPUT_CLASSIFY_FOLDER_ID and csv_name:
                files = _drive_find_by_name(svc, OUTPUT_CLASSIFY_FOLDER_ID, csv_name)
                if files:
                    data = _drive_download_bytes(svc, files[0]["id"]).decode("utf-8", "ignore")
                    for row in csv.DictReader(io.StringIO(data)):
                        raw = (row.get("image") or row.get("filename") or "").strip()
                        if raw:
                            seen.add(os.path.basename(raw))
        else:
            df = read_csv(CSV_DIR / csv_name)
            if not df.empty:
                cols = {c.lower(): c for c in df.columns}
                col = cols.get("image") or cols.get("filename")
                if col:
                    for _, r in df.iterrows():
                        raw = str(r.get(col, "")).strip()
                        if raw:
                            seen.add(os.path.basename(raw))
    except Exception:
        seen = set()  # treat as "no CSV yet"

    new_names = sorted({os.path.basename(n) for n in current} - seen)
    dbg = {
        "region": region,
        "mode": "drive" if drive_mode else "local",
        "folder": folder_id_or_path,
        "current": len(current),
        "in_csv": len(seen),
        "new_count": len(new_names),
        "preview_new": new_names[:20],
        "csv_name": csv_name,
        "classify_folder": OUTPUT_CLASSIFY_FOLDER_ID if drive_mode else str(CSV_DIR),
    }
    return (len(new_names) > 0), dbg

def _inspect_worker(region: str, level: str, selection: Dict):
    lock = INSPECT_LASER_LOCK if region == "laser" else INSPECT_CRITICAL_LOCK
    state = INSPECT_STATE[region]
    try:
        ok, dbg_can = _can_run_classifier(region)
        if not ok:
            state.update({
                "status": "done",
                "summary": {"region": region, "skipped": True, "reason": "Classifier not available in current mode.", "debug": dbg_can},
                "finished_at": time.time(),
            })
            if lock.locked():
                lock.release()
            return
        state.update({"status": "running", "level": level, "selection": selection or None,
                      "started_at": time.time(), "finished_at": None})
        t0 = time.time()

        models = _normalize_models(level, selection)
        # This call runs models and performs the merge (laser/critical) exactly once:
        cls_summary = _classify_region(region, models)

        dur = round(time.time() - t0, 2)
        state.update({
            "status": "done",
            "summary": {"region": region, "level": level, "selected_models": models,
                        "classify": cls_summary, "merge": {"done": True}, "duration_sec": dur},
            "finished_at": time.time(),
        })
    except Exception as e:
        state.update({"status": "error", "summary": {"error": str(e)}, "finished_at": time.time()})
    finally:
        if lock.locked():
            lock.release()

def _inspect_body_worker():
    lock = INSPECT_BODY_LOCK
    state = INSPECT_STATE["body"]
    try:
        state.update({"status": "running", "level": None, "selection": None,
                      "started_at": time.time(), "finished_at": None})
        t0 = time.time()

        # Choose the local body runner
        runner = None
        if callable(run_body_inference):
            runner = run_body_inference
        elif callable(coat_body_run):
            runner = coat_body_run

        if not runner:
            state.update({"status": "done",
                          "summary": {"region": "body", "skipped": True,
                                      "reason": "No local body runner available."},
                          "finished_at": time.time()})
            return

        in_folder = _region_input_folder("body")
        with _temp_env(INPUT_FOLDER_ID=in_folder):
            cls_summary = runner()

        dur = round(time.time() - t0, 2)
        state.update({"status": "done",
                      "summary": {"region": "body", "classify": cls_summary,
                                  "merge": None, "duration_sec": dur},
                      "finished_at": time.time()})
        print(f"✅ Inspection (body) finished in {dur}s")
    except Exception as e:
        state.update({"status": "error", "summary": {"error": str(e)}, "finished_at": time.time()})
        print(f"❌ Inspection (body) failed: {e}")
    finally:
        if lock.locked():
            lock.release()

@app.route("/", endpoint="index")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "VISDOM – Pixels to Reports"

@app.route("/crop", endpoint="crop_page")
def crop():
    try:
        return render_template("crop.html")
    except Exception:
        return "Crop page"

# --- Inspection (ML) page ---
@app.route("/inspection", methods=["GET"])
def inspection():
    return render_template("inspection.html")

@app.route("/ocr")
def ocr():
    try:
        return render_template("ocr.html")
    except Exception:
        return "OCR page"

@app.route("/verify_init")
def verify_init():
    try:
        return render_template("verify.html")
    except Exception:
        return "Verify page"

# --- Manual Inspection (new page) ---
@app.route("/manual_inspection")
def manual_inspection():
    return render_template("manual_inspection.html")

# === PATCH ADD: dashboard page (near other page routes) ===
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/classify/body", methods=["POST"])
def classify_body():
    try:
        if not callable(run_body_inference):
            return jsonify({"error": "Body inference module not available."}), 501
        res = run_body_inference()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
ALLOWED_HOSTS = {"drive.google.com", "lh3.googleusercontent.com", "content.googleapis.com"}

def _host_ok(u: str) -> bool:
    from urllib.parse import urlparse
    h = urlparse(u).hostname or ""
    return h in ALLOWED_HOSTS

@app.route("/proxy")
def proxy():
    import urllib.request
    target = (request.args.get("url") or "").strip()
    if not target.startswith(("http://", "https://")) or not _host_ok(target):
        return jsonify({"error": "blocked url"}), 400
    try:
        with urllib.request.urlopen(target, timeout=7) as resp:
            data = resp.read(2_000_000)  # 2MB cap
            mime = resp.headers.get_content_type() or "application/octet-stream"
            return send_file(io.BytesIO(data), mimetype=mime)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/verify_data")
def verify_data():
    only_dis = (request.args.get("only_disagreements", "1") != "0")
    svc = authenticate()

    # ---- LOCAL MODE ----
    if not _is_drive(svc):
        try:
            items = get_verification_items(only_disagreements=only_dis)
            # Normalize: ensure proxy_url present for any img_path
            norm = []
            for it in items:
                p = it.get("proxy_url") or ""
                if (not p) and it.get("img_path"):
                    it["proxy_url"] = f"/local/{it['img_path']}"
                norm.append(it)
            summary = {
                "total_rows": len(norm),
                "review_count": len(norm),
                "all_ok": len(norm) == 0,
            }
            return jsonify({"summary": summary, "items": norm}), 200
        except Exception as e:
            return jsonify({"error": f"Local verification failed: {e}"}), 500

    # ---- DRIVE MODE ----
    if not OUTPUT_OCR_FOLDER_ID or not FINAL_CSV_NAME:
        return jsonify({"error": "OCR output folder or CSV name not configured."}), 400

    csv_files = _drive_find_by_name(svc, OUTPUT_OCR_FOLDER_ID, FINAL_CSV_NAME)
    if not csv_files:
        return jsonify({"error": f"Final CSV '{FINAL_CSV_NAME}' not found in OUTPUT_OCR_FOLDER_ID."}), 404

    csv_text = _drive_download_text(svc, csv_files[0]["id"])
    rows = list(csv.DictReader(io.StringIO(csv_text))) if csv_text else []

    items = []
    for r in rows:
        img_name = (r.get("image") or "").strip()

        raw_agree = r.get("agree", "")
        if isinstance(raw_agree, bool):
            agree_bool = raw_agree
        else:
            s = str(raw_agree).strip().upper()
            agree_bool = s in ("YES", "TRUE", "1")

        # Fallbacks so mismatches still show
        if not agree_bool:
            py = (r.get("pred_yolo", "") or "").strip()
            pc = (r.get("pred_crnn", "") or "").strip()
            if py and pc and py == pc:
                agree_bool = True
        if not agree_bool and (r.get("actual_code") or "").strip():
            agree_bool = True

        if only_dis and agree_bool:
            continue

        img_id = ""
        proxy_url = ""
        direct_url = ""
        if img_name and OUTPUT_CODE_FOLDER_ID:
            m = _drive_find_by_name(svc, OUTPUT_CODE_FOLDER_ID, img_name)
            if m:
                img_id = m[0]["id"]
                proxy_url = f"/image/{img_id}"
                direct_url = f"https://drive.google.com/uc?id={img_id}"

        items.append({
            "image": img_name,
            "pred_yolo": r.get("pred_yolo", ""),
            "pred_crnn": r.get("pred_crnn", ""),
            "agree": "YES" if agree_bool else "NO",
            "img_id": img_id,
            "img_url": direct_url,
            "proxy_url": proxy_url,
        })

    summary = {"total_rows": len(rows), "review_count": len(items), "all_ok": len(items) == 0 and len(rows) > 0}
    return jsonify({"summary": summary, "items": items}), 200

def _extract_corrections(payload: Dict) -> Dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("actual_code"), dict):
        raw = payload["actual_code"]
    elif isinstance(payload.get("corrections"), dict):
        raw = payload["corrections"]
    else:
        raw = {}
        for k, v in payload.items():
            if not isinstance(k, str):
                continue
            if k.startswith("actual_code[") and k.endswith("]"):
                img = k[len("actual_code["):-1]
                raw[img] = v

    out = {}
    for k, v in (raw or {}).items():
        img = (k or "").strip()
        code = (v or "").strip()
        if img and code:
            out[img] = code
    return out

@app.route("/save_verifications", methods=["POST"])
def save_verifications():
    try:
        payload = request.get_json(silent=True) or {}
        corrections = _extract_corrections(payload)
        if not corrections:
            return jsonify({"message": "No corrections provided.", "received": 0}), 200

        summary = apply_user_corrections(corrections)  # updates FINAL_CSV_NAME + verified_codes.csv on Drive
        msg = f"Saved: {summary.get('applied',0)} (invalid: {summary.get('invalid',0)})"
        return jsonify({"message": msg, "summary": summary}), 200
    except Exception as e:
        return jsonify({"message": f"❌ Failed to save corrections: {e}"}), 500
@app.route("/apply_corrections", methods=["POST"])

def apply_corrections_alias():
    return save_verifications()
# --- Helper: report "done" once on transition from running -------------------
_JOB_STATUS_MEM: dict[str, dict] = {}

def _done_once(job: str, computed_status: str) -> str:
    mem = _JOB_STATUS_MEM.get(job, {"last": "idle"})
    prev = mem.get("last", "idle")

    # Transition running -> (idle|stale): emit 'done' once
    if prev == "running" and computed_status in ("idle", "stale"):
        mem["last"] = "done"
        _JOB_STATUS_MEM[job] = mem
        return "done"

    # If we emitted 'done' last time, flip to the new steady state now
    if prev == "done" and computed_status != "running":
        mem["last"] = computed_status
        _JOB_STATUS_MEM[job] = mem
        return computed_status

    # Normal pass-through + record
    mem["last"] = computed_status
    _JOB_STATUS_MEM[job] = mem
    return computed_status


# --- Corrected job_status ----------------------------------------------------
@app.route("/job_status")
def job_status():
    job = (request.args.get("job") or "").lower()
    svc = authenticate()
    out = {"job": job}

    try:
        if job == "crop":
            in_bases  = _bases_from_inputs(svc, INPUT_IMAGE_FOLDER_ID or INPUT_FOLDER_ID)
            out_bases = _bases_with_code(svc, OUTPUT_CODE_FOLDER_ID)
            pending   = sorted(in_bases - out_bases)

            # Treat sentinel as a hard "done" signal when lock isn't held
            sentinel_done = False
            try:
                if OUTPUT_CODE_FOLDER_ID and CROP_SENTINEL_NAME:
                    sentinel_done = _drive_file_exists(svc, OUTPUT_CODE_FOLDER_ID, CROP_SENTINEL_NAME)
            except Exception:
                sentinel_done = False

            if CROP_LOCK.locked():
                status = "running"
            elif sentinel_done and len(in_bases) > 0:
                status = "idle"
                pending = []
            elif pending:
                status = "stale"
            else:
                status = "idle"

            status = _done_once(job, status)

            out.update({
                "status": status,
                "pending_count": len(pending),
                "pending_preview": pending[:20],
                "counts": {"inputs": len(in_bases), "codes": len(out_bases)},
                "up_to_date": (status in ("idle", "done") and (len(in_bases) > 0)),
                "sentinel_done": bool(sentinel_done),
            })

        elif job == "split":
            todo, dbg = _split_todo_list(svc)

            if SPLIT_LOCK.locked():
                status = "running"
            elif todo:
                status = "stale"
            else:
                status = "idle"

            status = _done_once(job, status)

            out.update({
                "status": status,
                "pending_count": len(todo),
                "pending_preview": [t["name"] for t in todo[:20]],
                "counts": dbg,
                "up_to_date": (status in ("idle", "done") and int(dbg.get("inputs", 0)) > 0),
            })

        elif job == "ocr":
            code_names = _drive_list_code_names(svc, OUTPUT_CODE_FOLDER_ID)
            csv_images = _final_csv_images(svc)
            new_images = sorted(code_names - csv_images)

            if OCR_LOCK.locked():
                status = "running"
            elif new_images:
                status = "stale"
            else:
                status = "idle"

            status = _done_once(job, status)

            out.update({
                "status": status,
                "pending_count": len(new_images),
                "pending_preview": new_images[:20],
                "counts": {"codes": len(code_names), "csv_rows": len(csv_images)},
                "up_to_date": (status in ("idle", "done") and (len(code_names) > 0 or len(csv_images) > 0)),
            })

        elif job in ("inspect_laser", "inspect_critical", "inspect_body"):
            if job == "inspect_laser":
                region = "laser"
                locked = INSPECT_LASER_LOCK.locked()
            elif job == "inspect_critical":
                region = "critical"
                locked = INSPECT_CRITICAL_LOCK.locked()
            else:
                region = "body"
                locked = INSPECT_BODY_LOCK.locked()

            state = INSPECT_STATE[region]
            can_run_bool, can_dbg = _can_run_classifier(region)

            # base status from lock/state
            status = "running" if locked else (state.get("status") or "idle")

            should_run, dbg = _region_new_images_exist(region)

            # Surface config errors explicitly
            if (not locked) and (not should_run) and dbg.get("reason") == "missing folder config":
                status = "error"

            # Allow 'stale' if we're idle OR just finished ('done') and there is new work
            if (not locked) and can_run_bool and status in ("idle", "running", "done") and should_run and dbg.get("new_count", 0) > 0:
                status = "stale"

            status = _done_once(job, status)

            out.update({
                "status": status,
                "region": region,
                "level": state.get("level"),
                "selection": state.get("selection"),
                "summary": state.get("summary"),
                "started_at": state.get("started_at"),
                "finished_at": state.get("finished_at"),
                "debug": {**(dbg or {}), **(can_dbg or {})},
                "enabled": bool(can_run_bool),
            })


        elif job == "prepare_defects":
            st = PREPARE_DEFECTS_STATE.get("status","idle")
            st = _done_once(job, st if st in ("running","idle","stale","done","error") else "idle")
            out.update({
                "status": st,
                "counts": PREPARE_DEFECTS_STATE.get("counts",{}),
                "started_at": PREPARE_DEFECTS_STATE.get("started_at"),
                "finished_at": PREPARE_DEFECTS_STATE.get("finished_at"),
            })
            return jsonify(out), 200

        elif job == "master":
            st = MASTER_STATE.get("status", "idle")
            out.update({
                "status": st,
                "phase": MASTER_STATE.get("phase", "idle"),
                "log": MASTER_STATE.get("log", []),
                "started_at": MASTER_STATE.get("started_at"),
                "finished_at": MASTER_STATE.get("finished_at"),
            })
            return jsonify(out), 200

        else:
            out["error"] = ("unknown job (use job=crop|split|ocr|inspect_laser|"
                            "inspect_critical|inspect_body|prepare_defects|master)")
            return jsonify(out), 400

        return jsonify(out), 200

    except Exception as e:
        out.update({"status": "error", "error": str(e)})
        return jsonify(out), 500
    
@app.route("/prepare_defects/status")
def prepare_defects_status():
    return jsonify(PREPARE_DEFECTS_STATE), 200

@app.get("/__debug/config")
def _debug_config():
    snap = {
        "cwd": str(Path.cwd()),
        "_MEIPASS": getattr(sys, "_MEIPASS", None),
        "env": {k: os.getenv(k) for k in [
            "IMAGES_DIR","CSV_DIR","OUTPUT_DIR","TEMP_DIR",
            "INPUT_IMAGE_FOLDER_ID","OUTPUT_LASER2_FOLDER_ID",
            "OUTPUT_CRITICAL_FOLDER_ID","OUTPUT_BODY_FOLDER_ID"
        ]},
        "resolved": {
            "IMAGES_DIR": str(IMAGES_DIR),
            "CSV_DIR": str(CSV_DIR),
            "OUTPUT_DIR": str(OUTPUT_DIR),
            "TEMP_DIR": str(TEMP_DIR),
        }
    }
    return jsonify(snap)

@app.route("/crop_drive", methods=["POST"])
def crop_drive():
    if not run_crop_drive_job:
        return jsonify({"message": "❌ Crop job module not configured on server."}), 501

    if not CROP_LOCK.acquire(blocking=False):
        return jsonify({"message": "⏳ Crop is already running. Please wait."}), 429

    try:
        svc = authenticate()
        input_folder = INPUT_IMAGE_FOLDER_ID or INPUT_FOLDER_ID

        if input_folder and OUTPUT_CODE_FOLDER_ID:
            in_bases  = _bases_from_inputs(svc, input_folder)
            out_bases = _bases_with_code(svc, OUTPUT_CODE_FOLDER_ID)
            todo_bases = sorted(in_bases - out_bases)

            if not todo_bases:
                CROP_LOCK.release()
                return jsonify({
                    "message": "✅ Crop outputs already exist for all inputs. Nothing to do.",
                    "debug": {
                        "input_folder": input_folder,
                        "code_folder": OUTPUT_CODE_FOLDER_ID,
                        "input_count": len(in_bases),
                        "code_count": len(out_bases),
                        "sample_inputs": sorted(list(in_bases))[:10],
                        "sample_outputs": sorted(list(out_bases))[:10],
                    }
                }), 200

            threading.Thread(target=_crop_worker, daemon=True).start()
            return jsonify({
                "message": f"🚀 Crop 20 + Code started for {len(todo_bases)} new image(s).",
                "new_bases": todo_bases[:20],
            }), 200

        svc = authenticate()
        if OUTPUT_CODE_FOLDER_ID and CROP_SENTINEL_NAME and _drive_file_exists(svc, OUTPUT_CODE_FOLDER_ID, CROP_SENTINEL_NAME):
            CROP_LOCK.release()
            return jsonify({"message": f"✅ Crop 20 + Code already completed ({CROP_SENTINEL_NAME} found)."}), 200

        threading.Thread(target=_crop_worker, daemon=True).start()
        return jsonify({"message": "🚀 Crop 20 + Code started. Check Drive outputs."}), 200

    except Exception as e:
        if CROP_LOCK.locked():
            CROP_LOCK.release()
        return jsonify({"message": f"❌ Failed to start crop job: {e}"}), 500

@app.route("/run_master", methods=["POST"])
def run_master():
    if not MASTER_LOCK.acquire(blocking=False):
        return jsonify({"message": "⏳ Master already running."}), 429

    def _run():
        try:
            MASTER_STATE.update({
                "status": "running",
                "phase": "crop",
                "log": [],
                "error": None,
                "started_at": time.time(),
                "finished_at": None
            })
            _master_log("Master started.")

            # 1) Crop
            if CROP_LOCK.acquire(blocking=False):
                _master_log("Crop step…")
                _crop_worker()  # worker releases CROP_LOCK when done
                time.sleep(0.2)

            # 2) Split 20->3 (idempotent)
            svc = authenticate()
            todo, dbg = _split_todo_list(svc)
            if todo:
                _master_log(f"Split step… ({len(todo)} new)")
                SPLIT_LOCK.acquire(blocking=True)
                threading.Thread(
                    target=_split_worker,
                    args=(todo, svc, dbg.get("inputs", 0)),
                    daemon=True
                ).start()
                while SPLIT_LOCK.locked():
                    time.sleep(0.5)
            else:
                _master_log("Split: no new images.")

            # 3) OCR
            if OCR_LOCK.acquire(blocking=False):
                _master_log("OCR step…")
                _ocr_worker()  # worker releases OCR_LOCK when done
                time.sleep(0.2)

            # 4) Critical (Max) — only if runnable
            should, dbg = _region_new_images_exist("critical")
            if should and _can_run_classifier("critical"):
                _master_log(f"Critical step… ({dbg.get('new_count', 0)} new)")
                INSPECT_CRITICAL_LOCK.acquire(blocking=True)
                threading.Thread(
                    target=_inspect_worker,
                    args=("critical", "max", {}),
                    daemon=True
                ).start()
                while INSPECT_CRITICAL_LOCK.locked():
                    time.sleep(0.5)
            elif should:
                _master_log("Critical: SKIPPED — classifier unavailable in this mode.")
            else:
                _master_log("Critical: up-to-date.")

            # 5) Laser (Max) — only if runnable
            should, dbg = _region_new_images_exist("laser")
            if should and _can_run_classifier("laser"):
                _master_log(f"Laser step… ({dbg.get('new_count', 0)} new)")
                INSPECT_LASER_LOCK.acquire(blocking=True)
                threading.Thread(
                    target=_inspect_worker,
                    args=("laser", "max", {}),
                    daemon=True
                ).start()
                while INSPECT_LASER_LOCK.locked():
                    time.sleep(0.5)
            elif should:
                _master_log("Laser: SKIPPED — classifier unavailable in this mode.")
            else:
                _master_log("Laser: up-to-date.")

            # 6) Body (CoAt) — only if runnable
            should, dbg = _region_new_images_exist("body")
            if should and _can_run_classifier("body"):
                _master_log(f"Body step… ({dbg.get('new_count', 0)} new)")
                INSPECT_BODY_LOCK.acquire(blocking=True)
                threading.Thread(target=_inspect_body_worker, daemon=True).start()
                while INSPECT_BODY_LOCK.locked():
                    time.sleep(0.5)
            elif should:
                _master_log("Body: SKIPPED — classifier unavailable in this mode.")
            else:
                _master_log("Body: up-to-date.")

            # 7) Prepare Defects (idempotent) — no lock here
            _master_log("Prepare Defects step…")
            status, data = _invoke_route(prepare_defects, path="/prepare_defects", method="POST")
            _master_log("Prepare defects started", {"status": status, "resp": data})

            # Wait until the background worker updates PREPARE_DEFECTS_STATE
            def _prep_done():
                return PREPARE_DEFECTS_STATE.get("status") in ("done", "error")
            _wait_until(_prep_done, timeout_s=3600, poll_s=1.5)

            MASTER_STATE.update({
                "status": "done",
                "phase": "idle",
                "finished_at": time.time(),
                "message": "Master completed."
            })
            _master_log("Master completed.")
        except Exception as e:
            MASTER_STATE.update({
                "status": "error",
                "phase": "idle",
                "error": str(e),
                "finished_at": time.time(),
                "message": f"Master error: {e}"
            })
        finally:
            if MASTER_LOCK.locked():
                MASTER_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": "🚀 Master started."}), 200

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    if not run_ocr_pipeline_job:
        return jsonify({"message": "❌ OCR pipeline module not configured on server."}), 501

    svc = authenticate()
    drive_mode = _is_drive(svc)

    # In Drive mode the output folder must be configured
    if drive_mode and not OUTPUT_OCR_FOLDER_ID:
        return jsonify({"message": "❌ Missing .env key: OUTPUT_OCR_FOLDER_ID"}), 400

    if not OCR_LOCK.acquire(blocking=False):
        return jsonify({"message": "⏳ OCR pipeline already running. Please wait."}), 429

    try:
        code_names = _drive_list_code_names(svc, OUTPUT_CODE_FOLDER_ID)
        csv_images = _final_csv_images(svc)
        new_images = sorted(code_names - csv_images)

        if drive_mode:
            final_exists = bool(FINAL_CSV_NAME and _drive_file_exists(svc, OUTPUT_OCR_FOLDER_ID, FINAL_CSV_NAME))
            sent_exists  = bool(OCR_SENTINEL_NAME and _drive_file_exists(svc, OUTPUT_OCR_FOLDER_ID, OCR_SENTINEL_NAME))

            if not new_images and (final_exists or sent_exists):
                OCR_LOCK.release()
                return jsonify({
                    "message": "✅ OCR is already up-to-date. Nothing to do.",
                    "debug": {
                        "codes": len(code_names),
                        "csv_rows": len(csv_images),
                        "final_exists": final_exists,
                        "sentinel_exists": sent_exists,
                    }
                }), 200

            # if we do have new work, remove stale final/sentinel so the UI reflects fresh state
            if new_images and final_exists:
                _drive_delete_by_name(svc, OUTPUT_OCR_FOLDER_ID, FINAL_CSV_NAME)
            if new_images and sent_exists:
                _drive_delete_by_name(svc, OUTPUT_OCR_FOLDER_ID, OCR_SENTINEL_NAME)

        # Start worker
        threading.Thread(target=_ocr_worker, daemon=True).start()
        msg = "🚀 OCR pipeline started."
        if new_images:
            msg += f" Found {len(new_images)} new code crop(s)."
        return jsonify({
            "message": msg,
            "preview_new": new_images[:20],
            "debug": {"codes": len(code_names), "csv_rows": len(csv_images)}
        }), 200

    except Exception as e:
        if OCR_LOCK.locked():
            OCR_LOCK.release()
        return jsonify({"message": f"❌ Failed to start OCR pipeline: {e}"}), 500

@app.route("/split_drive", methods=["POST"])
def split_drive():
    missing = []
    if not LASER_20_FOLDER_ID:   missing.append("OUTPUT_LASER_FOLDER_ID")
    if not BODY_FOLDER_ID:       missing.append("OUTPUT_BODY_FOLDER_ID")
    if not CRITICAL_FOLDER_ID:   missing.append("OUTPUT_CRITICAL_FOLDER_ID")
    if not LASER2_FOLDER_ID:     missing.append("OUTPUT_LASER2_FOLDER_ID")
    if missing:
        return jsonify({"message": f"❌ Missing .env keys: {', '.join(missing)}"}), 400

    if not SPLIT_LOCK.acquire(blocking=False):
        return jsonify({"message": "⏳ Split is already running. Please wait."}), 429

    service = authenticate()

    todo, dbg = _split_todo_list(service)

    if not todo:
        SPLIT_LOCK.release()
        return jsonify({
            "message": "✅ Split outputs already exist for all Laser-20 images. Nothing to do.",
            "debug": dbg
        }), 200

    try:
        if SPLIT_SENTINEL_NAME and _drive_file_exists(service, LASER2_FOLDER_ID, SPLIT_SENTINEL_NAME):
            _drive_delete_by_name(service, LASER2_FOLDER_ID, SPLIT_SENTINEL_NAME)
    except Exception:
        pass

    threading.Thread(target=_split_worker, args=(todo, service, dbg.get("inputs", 0)), daemon=True).start()

    return jsonify({
        "message": f"🚀 Split started on {len(todo)} Laser-20 image(s). Existing outputs were skipped.",
        "input_folder": LASER_20_FOLDER_ID,
        "output_folders": {"body": BODY_FOLDER_ID, "critical": CRITICAL_FOLDER_ID, "laser2": LASER2_FOLDER_ID},
        "debug": dbg
    }), 200

def _parse_inspection_payload():
    data = request.get_json(silent=True) or {}
    level = (data.get("level") or "max").strip().lower()
    if level not in ("mini", "moderate", "max"):
        level = "max"
    sel = data.get("selection") or None
    return level, sel

def _crop_has_new_work():
    svc = authenticate()
    input_folder = INPUT_IMAGE_FOLDER_ID or INPUT_FOLDER_ID
    if not input_folder or not OUTPUT_CODE_FOLDER_ID:
        return False, {"reason": "missing folder config"}
    in_bases  = _bases_from_inputs(svc, input_folder)
    out_bases = _bases_with_code(svc, OUTPUT_CODE_FOLDER_ID)
    todo_bases = sorted(in_bases - out_bases)
    return (len(todo_bases) > 0), {
        "input_count": len(in_bases),
        "code_count": len(out_bases),
        "new_count": len(todo_bases),
        "preview_new": todo_bases[:20],
    }

def _split_has_new_work():
    svc = authenticate()
    todo, dbg = _split_todo_list(svc)
    return (len(todo) > 0), {
        "todo": len(todo),
        "preview_new": [t["name"] for t in todo[:20]],
        "dbg": dbg,
    }

def _ocr_has_new_work():
    svc = authenticate()
    code_names = _drive_list_code_names(svc, OUTPUT_CODE_FOLDER_ID)
    csv_images = _final_csv_images(svc)
    new_images = sorted(code_names - csv_images)
    return (len(new_images) > 0), {
        "codes": len(code_names),
        "csv_rows": len(csv_images),
        "new_count": len(new_images),
        "preview_new": new_images[:20],
    }

def _preflight_has_new_work():
    """Aggregate preflight for the master run."""
    crop_new,  crop_dbg  = _crop_has_new_work()
    split_new, split_dbg = _split_has_new_work()
    ocr_new,   ocr_dbg   = _ocr_has_new_work()
    laser_new, laser_dbg = _region_new_images_exist("laser")
    crit_new,  crit_dbg  = _region_new_images_exist("critical")
    body_new,  body_dbg  = _region_new_images_exist("body")
    any_new = any([crop_new, split_new, ocr_new, laser_new, crit_new, body_new])
    return any_new, {
        "crop": crop_dbg, "split": split_dbg, "ocr": ocr_dbg,
        "laser": laser_dbg, "critical": crit_dbg, "body": body_dbg
    }
@app.post("/automation/start")
def automation_start():
    # Prevent duplicate runs
    with MASTER_LOCK:
        if MASTER_STATE.get("status") == "running":
            return jsonify({
                "message": "⏳ Automation already running.",
                "phase": MASTER_STATE.get("phase", "starting"),
            }), 429

    # Preflight: run only if something is new
    has_new, dbg = _preflight_has_new_work()
    if not has_new:
        return jsonify({
            "message": "✅ Everything is up-to-date. No new images to process.",
            "preflight": dbg
        }), 200

    # initialize/reset state and start worker
    with MASTER_LOCK:
        MASTER_STATE.update({
            "status": "running",
            "phase": "starting",
            "message": "Starting…",
            "error": None,
            "log": [],
            "started_at": time.time(),
            "finished_at": None,
        })
    threading.Thread(target=_automation_worker, daemon=True).start()
    return jsonify({"message": "🚀 Automation started", "preflight": dbg}), 200

@app.route("/inspect_laser", methods=["POST"])
def inspect_laser():
    level, selection = _parse_inspection_payload()
    ok, _dbg_can = _can_run_classifier("laser")
    if not ok:
        return jsonify({"started": False, "message": "❌ Laser classifier not available in this mode."}), 501

    if not INSPECT_LASER_LOCK.acquire(blocking=False):
        return jsonify({"started": False, "message": "Inspection is already in progress."}), 429
    try:
        should, debug = _region_new_images_exist("laser")
        if not should:
            if INSPECT_LASER_LOCK.locked(): INSPECT_LASER_LOCK.release()
            return jsonify({"started": False, "message": "✅ No new Laser images to inspect.", "debug": debug}), 200
        threading.Thread(target=_inspect_worker, args=("laser", level, selection), daemon=True).start()
        return jsonify({"started": True, "level": level, "selection": selection, "debug": debug}), 200
    except Exception as e:
        if INSPECT_LASER_LOCK.locked(): INSPECT_LASER_LOCK.release()
        return jsonify({"started": False, "message": f"❌ Failed to start Laser inspection: {e}"}), 500

@app.route("/inspect_critical", methods=["POST"])
def inspect_critical():
    level, selection = _parse_inspection_payload()
    ok, _dbg_can = _can_run_classifier("critical")
    if not ok:
        return jsonify({"started": False, "message": "❌ Critical classifier not available in this mode."}), 501

    if not INSPECT_CRITICAL_LOCK.acquire(blocking=False):
        return jsonify({"started": False, "message": "Inspection is already in progress."}), 429
    try:
        should, debug = _region_new_images_exist("critical")
        if not should:
            if INSPECT_CRITICAL_LOCK.locked(): INSPECT_CRITICAL_LOCK.release()
            return jsonify({"started": False, "message": "✅ No new Critical images to inspect.", "debug": debug}), 200

        threading.Thread(target=_inspect_worker, args=("critical", level, selection), daemon=True).start()
        return jsonify({"started": True, "level": level, "selection": selection, "debug": debug}), 200
    except Exception as e:
        if INSPECT_CRITICAL_LOCK.locked():
            INSPECT_CRITICAL_LOCK.release()
        return jsonify({"started": False, "message": f"❌ Failed to start Critical inspection: {e}"}), 500


@app.route("/inspect_body", methods=["POST"])
def inspect_body():
    ok, _dbg = _can_run_classifier("body")
    if not ok:
        return jsonify({"message":"❌ Body inspection unavailable: no classifier wired for this mode."}), 501

    if not INSPECT_BODY_LOCK.acquire(blocking=False):
        return jsonify({"message":"⏳ Body inspection already running."}), 429

    should, dbg = _region_new_images_exist("body")
    if not should:
        if INSPECT_BODY_LOCK.locked(): INSPECT_BODY_LOCK.release()
        return jsonify({"message":"No new body images.", "debug":dbg, "new_count":0}), 200

    threading.Thread(target=_inspect_body_worker, daemon=True).start()
    return jsonify({"message":"🚀 Body inspection started.", "new_count": dbg.get("new_count")}), 200

def severity_to_color(sev: int) -> str:
    return "red" if sev >= 3 else "orange" if sev == 2 else "yellow" if sev == 1 else "none"

MANUAL_VERIFY_CSV = os.getenv("MANUAL_VERIFY_CSV", "manually_review.csv").strip()
MANUAL_VERIFY_PATH = (CSV_DIR / MANUAL_VERIFY_CSV).resolve()
MANUAL_PROGRESS_CSV = "manual_progress.csv"  # side-car to count per-base

def _load_csv(path: str | Path) -> Dict[str, Dict]:
    rows = {}
    if not Path(path).exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row.get("base_key","")] = row
    return rows

def _save_csv(path: str | Path, rows: Dict[str, Dict]):
    if not rows:
        # write empty with header
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["base_key","decision","last_image","region","reviewed_at"])
            w.writeheader()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["base_key","decision","last_image","region","reviewed_at"])
        w.writeheader()
        w.writerows(rows.values())

def _progress_load() -> Dict[str, Dict]:
    if not os.path.exists(MANUAL_PROGRESS_CSV):
        return {}
    with open(MANUAL_PROGRESS_CSV, newline="", encoding="utf-8") as f:
        return {r["base_key"]: r for r in csv.DictReader(f)}

def _progress_save(d: Dict[str, Dict]):
    with open(MANUAL_PROGRESS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["base_key","total","good_count","defective"])
        w.writeheader()
        w.writerows(d.values())

@app.route("/manual_items")
def manual_items():
    show_all = request.args.get("show_all", "0") == "1"

    # Respect query flag or .env to decide prepared vs raw folders
    use_prepared_qs  = (request.args.get("use_prepared_folders", "0") == "1")
    use_prepared_env = (os.getenv("MANUAL_USE_DEFECT_FOLDERS", "0") == "1")
    USE_PREPARED = use_prepared_qs or use_prepared_env

    # Paths
    LASER2_DIR       = folder(os.getenv("OUTPUT_LASER2_FOLDER_ID",   "data/images/regions/laser"))
    CRITICAL_DIR     = folder(os.getenv("OUTPUT_CRITICAL_FOLDER_ID", "data/images/regions/critical"))
    BODY_DIR         = folder(os.getenv("OUTPUT_BODY_FOLDER_ID",     "data/images/regions/body"))
    DEF_LASER_DIR    = folder(os.getenv("DEFECTIVE_LASER_FOLDER_ID",    "data/images/defective/laser"))
    DEF_CRITICAL_DIR = folder(os.getenv("DEFECTIVE_CRITICAL_FOLDER_ID", "data/images/defective/critical"))
    DEF_BODY_DIR     = folder(os.getenv("DEFECTIVE_BODY_FOLDER_ID",     "data/images/defective/body"))

    LASER_COMBINED_CSV = CSV_DIR / os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv")
    CRIT_COMBINED_CSV  = CSV_DIR / os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv")
    BODY_CSV           = CSV_DIR / os.getenv("BODY_CSV",           "body_coat_results.csv")

    HIDDEN_SESSION = globals().get("HIDDEN_ROOTS", set()) or set()

    def _root_of_local(name: str) -> str:
        stem = Path(name).stem
        for tag in ("_laser", "_body", "_critical"):
            if stem.endswith(tag):
                return stem[: -len(tag)]
        return stem

    def _sev_color(sev: int) -> str:
        return "red" if sev >= 3 else "orange" if sev == 2 else "yellow" if sev == 1 else "none"

    # Unified: read completed from CSV_DIR / MANUAL_VERIFY_CSV
    def _completed_bases_from_csv() -> Set[str]:
        completed: Set[str] = set()
        path = (CSV_DIR / os.getenv("MANUAL_VERIFY_CSV", "manually_review.csv")).resolve()
        if not path.exists():
            return completed
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dec = (row.get("decision") or "").strip().upper()
                    bk  = (row.get("base_key")  or "").strip()
                    if bk and dec in ("GOOD", "DEFECTIVE"):
                        completed.add(bk)
        except Exception:
            pass
        return completed

    def _load_votes_local() -> dict:
        votes: dict = {}
        def _ingest(df: pd.DataFrame, region: str):
            if df.empty: return
            cols = {c.lower(): c for c in df.columns}
            img_col = cols.get("image") or cols.get("filename")
            if not img_col: return
            for _, row in df.iterrows():
                raw = str(row[img_col]).strip()
                if not raw: continue
                img = os.path.basename(raw)
                regd = votes.setdefault(img, {}).setdefault(region, {})
                for m in ("maxvit", "coat", "swinv2"):
                    if m in cols: regd[m] = str(row[cols[m]]).strip()
                if region == "body" and "classification" in cols and "coat" not in regd:
                    regd["coat"] = str(row[cols["classification"]]).strip()
        if LASER_COMBINED_CSV.exists(): _ingest(read_csv(LASER_COMBINED_CSV), "laser")
        if CRIT_COMBINED_CSV.exists():  _ingest(read_csv(CRIT_COMBINED_CSV),  "critical")
        if BODY_CSV.exists():           _ingest(read_csv(BODY_CSV),           "body")
        return votes

    def _list_local_images(dir_path: Path) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in list_images(dir_path):
            rel = _rel_under_images(p)
            if not rel: continue
            out[p.name] = {"path": p, "rel": rel}
        return out

    votes_all = _load_votes_local()
    completed_bases = _completed_bases_from_csv()

    src = (
        {"laser": _list_local_images(DEF_LASER_DIR),
         "critical": _list_local_images(DEF_CRITICAL_DIR),
         "body": _list_local_images(DEF_BODY_DIR)}
        if USE_PREPARED else
        {"laser": _list_local_images(LASER2_DIR),
         "critical": _list_local_images(CRITICAL_DIR),
         "body": _list_local_images(BODY_DIR)}
    )

    items = []
    for reg, mapping in src.items():
        for name, meta in mapping.items():
            rt = _root_of_local(name)
            if not show_all and (rt in HIDDEN_SESSION or rt in completed_bases):
                continue

            per_model = (votes_all.get(name, {}) or {}).get(reg, {}) or {}
            display = {
                "MaxViT": (per_model.get("maxvit", "N/A") or "N/A").title(),
                "CoaT":   (per_model.get("coat",   "N/A") or "N/A").title(),
                "SwinV2": (per_model.get("swinv2", "N/A") or "N/A").title(),
            }
            sev = sum(1 for v in display.values() if v.lower() == "defective")
            any_bad = sev > 0

            rel = meta["rel"]
            items.append({
                "image": name,
                "region": reg,
                "root": rt,
                "img_id": "",
                "img_path": rel,
                "image_url": rel,
                "proxy_url": f"/local/{rel}",
                "votes": display,
                "severity": sev,
                "severity_color": _sev_color(sev),
                "status": "Defective" if any_bad else "Good",
            })

    return jsonify({"count": len(items), "items": items}), 200

@app.route("/manual_data")
def manual_data():
    fresh = request.args.get("fresh") == "1"
    only_bad = (request.args.get("only_bad") == "1")

    # Respect query flag or .env to decide prepared vs raw folders
    use_prepared_qs  = (request.args.get("use_prepared_folders", "0") == "1")
    use_prepared_env = (os.getenv("MANUAL_USE_DEFECT_FOLDERS", "0") == "1")
    USE_PREPARED = use_prepared_qs or use_prepared_env

    LASER2_DIR       = folder(os.getenv("OUTPUT_LASER2_FOLDER_ID",   "data/images/regions/laser"))
    CRITICAL_DIR     = folder(os.getenv("OUTPUT_CRITICAL_FOLDER_ID", "data/images/regions/critical"))
    BODY_DIR         = folder(os.getenv("OUTPUT_BODY_FOLDER_ID",     "data/images/regions/body"))
    DEF_LASER_DIR    = folder(os.getenv("DEFECTIVE_LASER_FOLDER_ID",    "data/images/defective/laser"))
    DEF_CRITICAL_DIR = folder(os.getenv("DEFECTIVE_CRITICAL_FOLDER_ID", "data/images/defective/critical"))
    DEF_BODY_DIR     = folder(os.getenv("DEFECTIVE_BODY_FOLDER_ID",     "data/images/defective/body"))

    LASER_COMBINED_CSV = CSV_DIR / os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv")
    CRIT_COMBINED_CSV  = CSV_DIR / os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv")
    BODY_CSV           = CSV_DIR / os.getenv("BODY_CSV",           "body_coat_results.csv")

    HIDDEN_SESSION = set() if fresh else (globals().get("HIDDEN_ROOTS", set()) or set())

    def _root_of_local(name: str) -> str:
        stem = Path(name).stem
        for tag in ("_laser", "_body", "_critical"):
            if stem.endswith(tag):
                return stem[: -len(tag)]
        return stem

    def _load_votes_local() -> dict:
        votes: dict = {}
        def _ingest(df: pd.DataFrame, region: str):
            if df.empty: return
            cols = {c.lower(): c for c in df.columns}
            img_col = cols.get("image") or cols.get("filename")
            if not img_col: return
            for _, row in df.iterrows():
                raw = str(row[img_col]).strip()
                if not raw: continue
                img = os.path.basename(raw)
                regd = votes.setdefault(img, {}).setdefault(region, {})
                for m in ("maxvit", "coat", "swinv2"):
                    if m in cols: regd[m] = str(row[cols[m]]).strip()
                if region == "body" and "classification" in cols and "coat" not in regd:
                    regd["coat"] = str(row[cols["classification"]]).strip()
        if LASER_COMBINED_CSV.exists(): _ingest(read_csv(LASER_COMBINED_CSV), "laser")
        if CRIT_COMBINED_CSV.exists():  _ingest(read_csv(CRIT_COMBINED_CSV),  "critical")
        if BODY_CSV.exists():           _ingest(read_csv(BODY_CSV),           "body")
        return votes

    # Unified: read completed from CSV_DIR / MANUAL_VERIFY_CSV
    def _completed_bases_from_csv() -> Set[str]:
        completed: Set[str] = set()
        path = (CSV_DIR / os.getenv("MANUAL_VERIFY_CSV", "manually_review.csv")).resolve()
        if not path.exists():
            return completed
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dec = (row.get("decision") or "").strip().upper()
                    bk  = (row.get("base_key")  or "").strip()
                    if bk and dec in ("GOOD", "DEFECTIVE"):
                        completed.add(bk)
        except Exception:
            pass
        return completed

    def _list_candidates_from_dir(reg: str, dir_path: Path, votes_all: dict) -> list[tuple[str, dict]]:
        out = []
        for p in list_images(dir_path):
            rel = _rel_under_images(p)
            if not rel: continue
            img_name = p.name
            base_key = _root_of_local(img_name)
            models = (votes_all.get(img_name, {}) or {}).get(reg, {}) or {}
            out.append((rel, {"region": reg, "base_key": base_key, "img_name": img_name, "models": models}))
        return out

    votes_all = _load_votes_local()
    completed_bases = _completed_bases_from_csv()

    candidates: list[tuple[str, dict]] = []
    if USE_PREPARED:
        candidates += _list_candidates_from_dir("laser",    DEF_LASER_DIR,    votes_all)
        candidates += _list_candidates_from_dir("critical", DEF_CRITICAL_DIR, votes_all)
        candidates += _list_candidates_from_dir("body",     DEF_BODY_DIR,     votes_all)
    else:
        candidates += _list_candidates_from_dir("laser",    LASER2_DIR,    votes_all)
        candidates += _list_candidates_from_dir("critical", CRITICAL_DIR,  votes_all)
        candidates += _list_candidates_from_dir("body",     BODY_DIR,      votes_all)

    items: list[dict] = []
    for image_rel, meta in candidates:
        base_key = meta["base_key"]
        if base_key in HIDDEN_SESSION or base_key in completed_bases:
            continue

        models = meta.get("models", {}) or {}
        votes = sum(1 for v in models.values() if str(v).lower() == "defective")
        if only_bad and votes == 0:
            continue

        level = "red" if votes >= 3 else "orange" if votes == 2 else "yellow" if votes == 1 else "none"

        items.append({
            "image": meta["img_name"],
            "image_name": meta["img_name"],
            "img_path": image_rel,
            "image_url": image_rel,
            "proxy_url": f"/local/{image_rel}",
            "models": {
                "maxvit": (models.get("maxvit") or "N/A"),
                "coat":   (models.get("coat")   or "N/A"),
                "swinv2": (models.get("swinv2") or "N/A"),
            },
            "region": meta["region"],
            "base_key": base_key,
            "severity": votes,
            "level": level,
        })

    return jsonify({"items": items}), 200

@app.route("/rewrite_label", methods=["POST"])
def rewrite_label():
    global REWRITE_CONFIRM_DISABLED
    p = request.get_json(force=True)
    image  = (p.get("image") or "").strip()
    region = (p.get("region") or "").strip().lower()
    model  = (p.get("model") or "").strip()
    target = (p.get("to") or "").strip().title()
    confirm = bool(p.get("confirm", False))
    daa = bool(p.get("dont_ask_again", False))

    if not image or region not in ("laser","critical","body") or model not in ("MaxViT","CoaT","SwinV2"):
        return jsonify({"error": "invalid payload"}), 400

    if not REWRITE_CONFIRM_DISABLED and not confirm:
        return jsonify({"needs_confirm": True}), 412

    if daa:
        REWRITE_CONFIRM_DISABLED = True

    votes_all = _load_votes_from_drive()
    prev = votes_all.get(image, {}).get(region, {})
    prev_val = prev.get(model.lower(), "Good")

    action = {"type": "rewrite", "image": image, "region": region, "model": model, "from": prev_val, "to": target}
    UNDO_STACK.append(action)

    display = {
        "MaxViT": (prev.get("maxvit","N/A") or "N/A").title(),
        "CoaT":   (prev.get("coat","N/A")   or "N/A").title(),
        "SwinV2": (prev.get("swinv2","N/A") or "N/A").title(),
    }
    display[model] = target
    sev = sum(1 for v in display.values() if v.lower() == "defective")

    return jsonify({
        "ok": True,
        "override_applied": action,
        "votes": display,
        "severity": sev,
        "severity_color": "red" if sev == 3 else "orange" if sev == 2 else "yellow" if sev == 1 else "none",
        "dont_ask_again": REWRITE_CONFIRM_DISABLED
    }), 200

def _ensure_row(base_key: str):
    if base_key not in STATE["by_base_key"]:
        STATE["by_base_key"][base_key] = {"total": 0, "good_count": 0, "defective": 0}

@app.post("/manual/mark_good")
def mark_good():
    data = request.get_json(force=True)
    base_key = _extract_base_key(data.get("name") or data.get("image") or "")
    _ensure_row(base_key)
    row = STATE["by_base_key"][base_key]
    row["total"] += 1
    row["good_count"] += 1
    STATE["action_log"].append({"op": "good", "key": base_key})
    # mark completed if any decision exists
    if row["good_count"] + row["defective"] >= 1:
        STATE.setdefault("completed_keys", set()).add(base_key)
    # persist now (not just batched)
    _save_manual_state(STATE)
    STATE["_since_last_autosave"] = 0
    return jsonify(ok=True, base_key=base_key, row=row)

@app.post("/manual/mark_defective")
def mark_defective():
    data = request.get_json(force=True)
    base_key = _extract_base_key(data.get("name") or data.get("image") or "")
    _ensure_row(base_key)
    row = STATE["by_base_key"][base_key]
    row["total"] += 1
    row["defective"] += 1
    STATE["action_log"].append({"op": "defective", "key": base_key})
    STATE.setdefault("completed_keys", set()).add(base_key)
    _save_manual_state(STATE)
    STATE["_since_last_autosave"] = 0
    return jsonify(ok=True, base_key=base_key, row=row)

@app.route("/manual/undo_decision", methods=["POST"])
def manual_undo_decision():
    if not STATE["action_log"]:
        return jsonify(ok=False, error="Nothing to undo")
    last = STATE["action_log"].pop()
    base_key = last["key"]
    row = STATE["by_base_key"].get(base_key)
    if row:
        row["total"] = max(0, row["total"] - 1)
        if last["op"] == "good":
            row["good_count"] = max(0, row["good_count"] - 1)
        elif last["op"] == "defective":
            row["defective"] = max(0, row["defective"] - 1)
        if row["good_count"] + row["defective"] == 0:
            STATE.setdefault("completed_keys", set()).discard(base_key)
    _save_manual_state(STATE)
    STATE["_since_last_autosave"] = 0
    return jsonify(ok=True, base_key=base_key, row=row)

@app.route("/undo", methods=["POST"])
def undo():
    with STATE_LOCK:
        if not UNDO_STACK:
            return jsonify({"message": "Nothing to undo."}), 200
        last = UNDO_STACK.pop()
    return jsonify({"message": "Undone.", "undone": last, "remaining": len(UNDO_STACK)}), 200

@app.route("/mark_defective_and_hide_root", methods=["POST"])
def mark_defective_and_hide_root():
    p = request.get_json(force=True)
    image = (p.get("image") or "").strip()
    if not image:
        return jsonify({"error": "image required"}), 400
    rt = root_of(image)
    if not rt:
        return jsonify({"error": "invalid filename format"}), 400
    UNDO_STACK.append({"type":"hide_root","root":rt})
    HIDDEN_ROOTS.add(rt)
    return jsonify({"ok": True, "hidden_root": rt}), 200

@app.route("/manual/reload", methods=["POST"])
def manual_reload():
    return jsonify({"ok": True})

def _remaining_items(all_items_names):
    completed = set(STATE.get("completed_keys", set()))
    # NEW: union with CSV-driven completed set
    completed |= _completed_bases_from_csv()
    rem = []
    for nm in all_items_names:
        bk = _extract_base_key(nm)
        if bk not in completed:
            rem.append(nm)
    return rem

@app.post("/manual/next")
def manual_next():
    data = request.get_json(force=True) or {}
    all_items_names = data.get("all_items_names", [])  # FRONTEND must pass the *names*
    rem = _remaining_items(all_items_names)
    if not rem:
        return jsonify(done=True, message="All images classified.")
    # basic: return first remaining
    return jsonify(done=False, name=rem[0])

_PHASE_COLOR = {
    "idle": "red", "error": "red",
    "starting": "orange", "crop": "orange", "split": "orange",
    "yellow": "yellow", "ocr": "yellow",
    "inspect_critical": "yellow", "inspect_laser": "yellow", "inspect_body": "yellow",
    "prepare_defects": "yellow",
    "done": "green", "green": "green",
}


def _set_master(status=None, phase=None, msg=None, err=None, done=False):
    with MASTER_LOCK:
        if status is not None:
            MASTER_STATE["status"] = status
        if phase is not None:
            MASTER_STATE["phase"] = phase
        if msg is not None:
            MASTER_STATE["message"] = msg
        if err is not None:
            MASTER_STATE["error"] = err
        if status == "running" and not MASTER_STATE.get("started_at"):
            MASTER_STATE["started_at"] = time.time()
        if done:
            MASTER_STATE["finished_at"] = time.time()

def _master_log(*parts):
    msg = " ".join(str(p) for p in parts if p is not None)
    MASTER_STATE["log"].append(msg)
    MASTER_STATE["message"] = msg
    print("[MASTER]", msg)


def _invoke_route(func, *, method="POST", path="/", json_body=None):
    """
    Safely call a Flask route function (that returns a Response) without an active HTTP request.
    """
    with app.test_request_context(path, method=method, json=json_body):
        resp = func()
    status = getattr(resp, "status_code", 200)
    try:
        data = resp.get_json(silent=True)
    except Exception:
        data = None
    return status, (data or {})

def _wait_until(predicate, timeout_s=3600, poll_s=1.5):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False

def _job_status(job: str) -> dict:
    with app.test_request_context(f"/job_status?job={job}", method="GET"):
        resp = job_status()
    try:
        return resp.get_json(silent=True) or {}
    except Exception:
        return {}

def _automation_worker():
    try:
        _set_master(status="running", phase="starting", msg="Starting automation…", err=None)
        MASTER_STATE["log"] = []  # reset log each run

        # ---- 1) Crop ---------------------------------------------------------
        _set_master(phase="crop", msg="Cropping full images → code/regions")
        st, data = _invoke_route(crop_drive, path="/crop_drive", method="POST")
        _master_log("Crop started", {"status": st, "resp": data})
        _wait_until(lambda: (not CROP_LOCK.locked()) and (_job_status("crop").get("status") in ("idle", "error", None)))

        # ---- 2) Split --------------------------------------------------------
        _set_master(phase="split", msg="Splitting Laser-20 → Body/Critical/Laser")
        st, data = _invoke_route(split_drive, path="/split_drive", method="POST")
        _master_log("Split started", {"status": st, "resp": data})
        _wait_until(lambda: (not SPLIT_LOCK.locked()) and (_job_status("split").get("status") in ("idle", "error", None)))

        # Signal "yellow" region in UI after crop/split
        _set_master(phase="yellow", msg="Crop & Split done — preparing OCR")

        # ---- 3) OCR ----------------------------------------------------------
        _set_master(phase="ocr", msg="Running OCR pipeline")
        st, data = _invoke_route(run_pipeline, path="/run_pipeline", method="POST")
        _master_log("OCR started", {"status": st, "resp": data})
        _wait_until(lambda: (not OCR_LOCK.locked()) and (_job_status("ocr").get("status") in ("idle", "error", None)))

        # ---- 4) Critical (Max) ----------------------------------------------
        _set_master(phase="inspect_critical", msg="Inspection: Critical (Max)")
        if _can_run_classifier("critical") and INSPECT_CRITICAL_LOCK.acquire(blocking=False):
            try:
                threading.Thread(
                    target=_inspect_worker, args=("critical", "max", {}), daemon=True
                ).start()
            except Exception:
                if INSPECT_CRITICAL_LOCK.locked():
                    INSPECT_CRITICAL_LOCK.release()
                raise
        _wait_until(lambda: not INSPECT_CRITICAL_LOCK.locked())

        # ---- 5) Laser (Max) --------------------------------------------------
        _set_master(phase="inspect_laser", msg="Inspection: Laser (Max)")
        if _can_run_classifier("laser") and INSPECT_LASER_LOCK.acquire(blocking=False):
            try:
                threading.Thread(
                    target=_inspect_worker, args=("laser", "max", {}), daemon=True
                ).start()
            except Exception:
                if INSPECT_LASER_LOCK.locked():
                    INSPECT_LASER_LOCK.release()
                raise
        _wait_until(lambda: not INSPECT_LASER_LOCK.locked())

        # ---- 6) Body (CoaT) --------------------------------------------------
        _set_master(phase="inspect_body", msg="Inspection: Body (CoaT)")
        if _can_run_classifier("body") and INSPECT_BODY_LOCK.acquire(blocking=False):
            try:
                threading.Thread(target=_inspect_body_worker, daemon=True).start()
            except Exception:
                if INSPECT_BODY_LOCK.locked():
                    INSPECT_BODY_LOCK.release()
                raise
        _wait_until(lambda: not INSPECT_BODY_LOCK.locked())

        # ---- 7) Prepare Defects (Drive) -------------------------------------
        _set_master(phase="prepare_defects", msg="Collecting defective images for Manual Review")
        st, data = _invoke_route(prepare_defects, path="/prepare_defects", method="POST")
        _master_log("Prepare defects started", {"status": st, "resp": data})

        # Poll PREPARE_DEFECTS_STATE until done (or error)
        def _prep_done():
            return PREPARE_DEFECTS_STATE.get("status") in ("done", "error")
        _wait_until(_prep_done, timeout_s=3600, poll_s=1.5)

        _set_master(status="done", phase="done", msg="Automation completed.", done=True)
        _master_log("All steps completed")
    except Exception as e:
        app.logger.exception("Automation failed")
        _set_master(status="error", phase="error", msg="Automation failed.", err=f"{type(e).__name__}: {e}", done=True)
        _master_log("Automation error", {"error": f"{type(e).__name__}: {e}"})

@app.get("/automation/status")
def automation_status():
    phase = (MASTER_STATE.get("phase") or "idle").lower()
    color = _PHASE_COLOR.get(phase, "red")
    return jsonify({
        "status": MASTER_STATE.get("status", "idle"),
        "phase": phase,
        "color": color,
        "message": MASTER_STATE.get("message"),
        "log": MASTER_STATE.get("log", []),
        "started_at": MASTER_STATE.get("started_at"),
        "finished_at": MASTER_STATE.get("finished_at"),
        "error": MASTER_STATE.get("error"),
    }), 200

@app.get("/master/status")
def master_status_alias():
    # Alias so frontend polling to /master/status works
    return automation_status()

@app.post("/master/start")
def master_start_alias():
    # Alias so "Run Master" hitting /master/start works
    return automation_start()
@app.post("/manual/reset_hidden")
def manual_reset_hidden():
    HIDDEN_ROOTS.clear()
    return jsonify(ok=True, cleared="hidden_roots")

@app.post("/manual/reset_all_state")
def manual_reset_all_state():
    # Clear in-memory hides
    HIDDEN_ROOTS.clear()
    # Reset persistent manual state
    STATE["by_base_key"] = {}
    STATE["completed_keys"] = set()
    STATE["action_log"] = []
    try:
        if INSPECT_STATE_PATH.exists():
            INSPECT_STATE_PATH.unlink()
    except Exception:
        pass
    return jsonify(ok=True, cleared=["hidden_roots", "manual_state_json"])
@app.post("/manual/reset_local_csvs")
def manual_reset_local_csvs():
    removed = []
    for p in [MANUAL_VERIFY_CSV, MANUAL_PROGRESS_CSV]:
        try:
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
        except Exception:
            pass
    return jsonify(ok=True, removed=removed)
@app.post("/manual/clear_hidden")
def manual_clear_hidden():
    HIDDEN_ROOTS.clear()
    return jsonify(ok=True, cleared=True)

@app.post("/manual/forget_completed")
def manual_forget_completed():
    # keep per-base counts but make them eligible again
    STATE["completed_keys"] = set()
    _save_manual_state(STATE)
    return jsonify(ok=True)

@app.post("/manual/reset_all")
def manual_reset_all():
    # full reset: hide list + progress + decisions (optional file deletes)
    HIDDEN_ROOTS.clear()
    STATE["by_base_key"] = {}
    STATE["completed_keys"] = set()
    STATE["action_log"] = []
    try:
        if INSPECT_STATE_PATH.exists():
            INSPECT_STATE_PATH.unlink()  # forget persisted state
    except Exception:
        pass
    try:
        if os.path.exists(MANUAL_PROGRESS_CSV):
            os.remove(MANUAL_PROGRESS_CSV)
        if os.path.exists(MANUAL_VERIFY_CSV):
            os.remove(MANUAL_VERIFY_CSV)
    except Exception:
        pass
    _save_manual_state(STATE)
    return jsonify(ok=True, reset=True)

def _completed_bases_from_csv() -> Set[str]:
    completed: Set[str] = set()
    path = (CSV_DIR / MANUAL_VERIFY_CSV).resolve()
    if not path.exists():
        return completed

    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dec = (row.get("decision") or "").strip().upper()
                bk  = (row.get("base_key") or "").strip()
                if bk and dec in ("GOOD", "DEFECTIVE"):
                    completed.add(bk)
    except Exception:
        pass
    return completed


def _can_run_classifier(region: str) -> Tuple[bool, Dict]:
    r = (region or "").strip().lower()
    dbg: Dict[str, object] = {"region": r, "mode": "local"}

    if r == "laser":
        availability = {
            "maxvit": callable(run_laser_infer_maxvit),
            "coat": callable(run_laser_infer_coat),
            "swinv2": callable(run_laser_infer_swinv2),
        }
        ok = any(availability.values())
        dbg["available_models"] = [m for m, is_ok in availability.items() if is_ok]
        if not ok:
            dbg["reason"] = "No local Laser runners are wired (maxvit/coat/swinv2)."
        return ok, dbg

    if r == "critical":
        availability = {
            "maxvit": callable(run_crit_infer_maxvit),
            "coat": callable(run_crit_infer_coat),
            "swinv2": callable(run_crit_infer_swinv2),
        }
        ok = any(availability.values())
        dbg["available_models"] = [m for m, is_ok in availability.items() if is_ok]
        if not ok:
            dbg["reason"] = "No local Critical runners are wired (maxvit/coat/swinv2)."
        return ok, dbg

    if r == "body":
        # Body is single-model (CoaT) but may be exposed under two import names.
        available = callable(run_body_inference) or callable(coat_body_run)
        ok = bool(available)
        dbg["available_models"] = ["coat"] if ok else []
        if not ok:
            dbg["reason"] = "No local Body runner is wired (CoaT)."
        return ok, dbg

    return False, {"region": r, "reason": "Unknown region"}

from mimetypes import guess_type

def _is_drive(service) -> bool:
    return (service is not None) and hasattr(service, "files")

# Serve files that live under IMAGES_DIR via /local/<relpath> URLs safely.
@app.route("/local/<path:rel>")
def serve_local_image(rel: str):
    import mimetypes
    try:
        root = IMAGES_DIR.resolve()
        target = (IMAGES_DIR / rel).resolve()
        _ = target.relative_to(root)  # raises if outside
    except Exception:
        return jsonify({"error": "forbidden"}), 403

    if not target.exists() or not target.is_file():
        return jsonify({"error": "not found"}), 404

    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return send_file(str(target), mimetype=mime)

import csv
from collections import defaultdict

def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        try:
            return [r for r in csv.DictReader(f)]
        except Exception:
            return []

def _is_def_label(s: str) -> bool:
    s = (s or "").strip().upper()
    return s in ("DEFECTIVE", "BAD", "ANY_DEFECTIVE", "ALL_DEFECTIVE", "MIXED")

@app.get("/api/csv/list")
def api_csv_list():
    """List all CSVs under CSV_DIR."""
    out = []
    for p in sorted(CSV_DIR.rglob("*.csv")):
        try:
            out.append({
                "name": p.name,
                "rel": str(p.relative_to(CSV_DIR)),
                "size": p.stat().st_size,
                "mtime": int(p.stat().st_mtime),
            })
        except Exception:
            pass
    return jsonify({"root": str(CSV_DIR), "files": out})

@app.get("/api/csv/get")
def api_csv_get():
    """Return a CSV as JSON rows. Query: ?rel=<path under CSV_DIR>"""
    rel = (request.args.get("rel") or "").strip()
    if not rel:
        return jsonify({"error": "missing rel"}), 400
    try:
        path = (CSV_DIR / rel).resolve()
        if not str(path).startswith(str(CSV_DIR.resolve())):
            return jsonify({"error": "forbidden"}), 403
        rows = _read_csv_rows(path)
        return jsonify({"columns": list(rows[0].keys()) if rows else [], "rows": rows})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.get("/api/stats/defects")
def api_stats_defects():
    """Compute Good vs Defective for laser/critical/body across combined CSVs."""
    # File names can be overridden from .env (they already are elsewhere)
    LASER_COMBINED = CSV_DIR / os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv")
    CRIT_COMBINED  = CSV_DIR / os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv")
    BODY_CSV       = CSV_DIR / os.getenv("BODY_CSV",           "body_coat_results.csv")

    def count_from_combined(rows: list[dict], consensus_key="consensus") -> tuple[int,int]:
        good = bad = 0
        for r in rows:
            cons = (r.get(consensus_key) or "").strip().upper()
            if cons in ("ALL_GOOD", "GOOD", "PASS"):
                good += 1
            elif _is_def_label(cons):
                bad += 1
            else:
                # fallback from model columns
                labs = [ (r.get("MaxViT") or "").upper(),
                         (r.get("CoaT") or "").upper(),
                         (r.get("SwinV2") or "").upper() ]
                if any(_is_def_label(x) for x in labs):
                    bad += 1
                elif any(x for x in labs):
                    good += 1
        return good, bad

    def count_from_simple(rows: list[dict], key="classification") -> tuple[int,int]:
        good = bad = 0
        for r in rows:
            lab = (r.get(key) or "").strip().upper()
            if lab in ("GOOD", "OK", "PASS"):
                good += 1
            elif _is_def_label(lab):
                bad += 1
        return good, bad

    laser_good, laser_bad = count_from_combined(_read_csv_rows(LASER_COMBINED))
    crit_good,  crit_bad  = count_from_combined(_read_csv_rows(CRIT_COMBINED))
    body_good,  body_bad  = count_from_simple(_read_csv_rows(BODY_CSV), key="classification")

    def pct(bad, good):
        tot = bad + good
        return 0.0 if tot == 0 else round(100.0 * bad / tot, 4)

    return jsonify({
        "laser":    {"good": laser_good, "bad": laser_bad, "pct_defective": pct(laser_bad, laser_good)},
        "critical": {"good": crit_good,  "bad": crit_bad,  "pct_defective": pct(crit_bad,  crit_good)},
        "body":     {"good": body_good,  "bad": body_bad,  "pct_defective": pct(body_bad,  body_good)},
    })

@app.get("/api/defective/list")
def api_defective_list():
    """
    Aggregate all 'defective' images across laser/critical/body, showing region, image and reason.
    """
    LASER_COMBINED = CSV_DIR / os.getenv("LASER_COMBINED_CSV", "laser_all_models.csv")
    CRIT_COMBINED  = CSV_DIR / os.getenv("CRIT_COMBINED_CSV",  "critical_all_models.csv")
    BODY_CSV       = CSV_DIR / os.getenv("BODY_CSV",           "body_coat_results.csv")

    out = []

    # laser/critical combined CSVs
    for region, path in (("laser", LASER_COMBINED), ("critical", CRIT_COMBINED)):
        for r in _read_csv_rows(path):
            cons = (r.get("consensus") or "").upper()
            labs = { "MaxViT": (r.get("MaxViT") or "").upper(),
                     "CoaT":   (r.get("CoaT")   or "").upper(),
                     "SwinV2": (r.get("SwinV2") or "").upper() }
            if cons in ("ALL_DEFECTIVE", "ANY_DEFECTIVE", "MIXED") or any(_is_def_label(v) for v in labs.values()):
                out.append({
                    "region": region,
                    "image": (r.get("image") or "").strip(),
                    "consensus": cons,
                    "labels": labs,
                })

    # body coat single-model CSV
    for r in _read_csv_rows(BODY_CSV):
        lab = (r.get("classification") or "").upper()
        if _is_def_label(lab):
            out.append({
                "region": "body",
                "image": (r.get("image") or r.get("filename") or "").strip(),
                "consensus": lab,
                "labels": {"CoaT": lab},
            })

    # sort stable by region then image
    out.sort(key=lambda x: (x["region"], x["image"]))
    return jsonify({"count": len(out), "items": out})
def _rel_under_images(p: Path) -> str | None:
    """Return path to p relative to IMAGES_DIR if and only if p is inside IMAGES_DIR."""
    root = IMAGES_DIR.resolve()
    try:
        # fast path
        return p.resolve().relative_to(root).as_posix()
    except Exception:
        # not a descendant → skip (return None)
        return None
def _port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _open_browser_when_ready(url: str, host="127.0.0.1", port=5000, tries=60, wait=1.0):
    # Wait until the server is actually listening, then open default browser
    for _ in range(tries):
        if _port_open(host, port):
            try:
                webbrowser.open(url)
            except Exception:
                pass
            return
        time.sleep(wait)

@app.post("/shutdown")
def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    # Return first so the HTTP response completes before exiting.
    try:
        msg = "Shutting down…"
        # Schedule the exit on a short timer so response can flush
        def _die():
            time.sleep(0.2)
            if callable(func):
                func()              # graceful (Werkzeug dev server)
            else:
                os._exit(0)         # fallback for packaged/other servers
        threading.Thread(target=_die, daemon=True).start()
        return jsonify({"ok": True, "message": msg}), 200
    except Exception as e:
        # Last-resort exit
        threading.Thread(target=lambda: (time.sleep(0.2), os._exit(0)), daemon=True).start()
        return jsonify({"ok": False, "message": f"Forced exit: {e}"}), 200

if __name__ in ("__main__", "__mp_main__") or getattr(sys, "frozen", False):
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "5000")))
    open_path = os.getenv("OPEN_PATH", "/")
    url = f"http://{host}:{port}{open_path}"

    # If a copy is already running, just open the browser to it and exit
    if _port_open(host, port):
        try:
            webbrowser.open(url)
        finally:
            # When packaged (console=False) this prevents a "zombie" stub process
            if getattr(sys, "frozen", False):
                os._exit(0)
            else:
                sys.exit(0)

    # Open the browser once this instance is up
    threading.Thread(
        target=_open_browser_when_ready, args=(url, host, port), daemon=True
    ).start()

    # IMPORTANT: no reloader (prevents double processes), enable threads
    app.run(host=host, port=port, threaded=True, use_reloader=False)

    