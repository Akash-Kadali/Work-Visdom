# backend/inference_jobs/crnn_ocr.py  (PURE LOCAL • resilient networking • backward-compatible)

from __future__ import annotations

import os
import io
import csv
import time
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.modal_request import send_to_crnn

# ===================== CONFIG =====================
MAX_IMAGES           = int(os.getenv("MAX_IMAGES", 100000))
CRNN_MAX_WORKERS     = int(os.getenv("CRNN_MAX_WORKERS", 2))
DRY_RUN              = os.getenv("DRY_RUN", "0") == "1"
CRNN_FORCE           = os.getenv("CRNN_FORCE", "0") == "1"

# Networking / retry (tuneable via env)
ENDPOINT_RETRIES       = int(os.getenv("ENDPOINT_RETRIES", "4"))          # total attempts
ENDPOINT_BACKOFF_BASE  = float(os.getenv("ENDPOINT_BACKOFF_BASE", "1.8")) # exponential base
ENDPOINT_BACKOFF_MAX   = float(os.getenv("ENDPOINT_BACKOFF_MAX", "10"))   # max sleep per attempt (s)
ENDPOINT_JITTER_SEC    = float(os.getenv("ENDPOINT_JITTER_SEC", "0.25"))  # random jitter (s)

# Concurrency cap (helps avoid WinError 10053/10054 under parallel HTTPS)
CRNN_MAX_WORKERS_CAP = int(os.getenv("CRNN_MAX_WORKERS_CAP", "1"))
CRNN_MAX_WORKERS = max(1, min(CRNN_MAX_WORKERS, CRNN_MAX_WORKERS_CAP))

_ALLOWED_EXTS = tuple(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTS", ".jpg,.jpeg,.png,.bmp,.webp").split(",")
    if ext.strip()
)

OCR_ONLY_CODE_SUFFIX = os.getenv("OCR_ONLY_CODE_SUFFIX", "1") == "1"
_CODE_NAME_RE = re.compile(r"_code\.(?:png|jpe?g)$", re.IGNORECASE)

CHARSET = os.getenv("OCR_CHARSET", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
KEY_CANDIDATES = ("predicted_code", "prediction", "text", "code")

SAVE_CRNN_CSV = os.getenv("SAVE_CRNN_CSV", "1") == "1"
CRNN_CSV_NAME = os.getenv("CRNN_CSV_NAME", "crnn_results.csv")

# Fixed-length target (matches your YOLO job default)
CRNN_TAKE_K = int(os.getenv("CRNN_TAKE_K", "6"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================== LOCAL HELPERS =====================
def _iter_images_recursive(root: Path):
    """Yield Path objects for images under root (recursive), honoring allowed exts and *_code filter."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if not any(lower.endswith(ext) for ext in _ALLOWED_EXTS):
            continue
        if OCR_ONLY_CODE_SUFFIX and not _CODE_NAME_RE.search(lower):
            continue
        yield p

def list_images_local(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Return list of dicts:
      {"name": <relative_path_from_input_dir>, "path": <relative_path_from_input_dir>}
    Using relative paths keeps keys stable across subfolders.
    """
    images: List[Dict[str, Any]] = []
    for p in _iter_images_recursive(input_dir):
        rel = p.relative_to(input_dir).as_posix()
        images.append({"name": rel, "path": rel})
    return images

def _load_local_image_bytes(abs_input_dir: Path, path_rel: str) -> io.BytesIO:
    p = (abs_input_dir / path_rel).resolve()
    with open(p, "rb") as f:
        return io.BytesIO(f.read())


# ===================== OCR HELPERS =====================
def _sanitize_pred(pred: str, k: int) -> str:
    if not isinstance(pred, str):
        pred = str(pred or "")
    pred = pred.strip().upper()
    if CHARSET:
        pred = "".join(c for c in pred if c in CHARSET)
    if len(pred) > k:
        pred = pred[:k]
    if len(pred) < k:
        pred = pred + ("?" * (k - len(pred)))
    return pred

def _extract_pred_from_result(result: Dict[str, Any]) -> str:
    for k in KEY_CANDIDATES:
        val = result.get(k)
        if val:
            return str(val)
    return ""


# ===================== CSV HELPERS =====================
def _rows_to_csv_text(rows: List[Dict[str, str]]) -> str:
    headers = ["image", "pred_crnn"]
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in headers})
    return sio.getvalue()

def _load_existing_csv_map(csv_dir: Path, filename: str) -> Dict[str, str]:
    """If a per-model CSV already exists, load image->pred_crnn so we can keep prior results."""
    p = (csv_dir / filename)
    if not p.exists():
        return {}
    try:
        csv_text = p.read_text(encoding="utf-8")
        rdr = csv.DictReader(io.StringIO(csv_text))
        out: Dict[str, str] = {}
        for row in rdr:
            img = (row.get("image") or "").strip()
            pred = _sanitize_pred(row.get("pred_crnn") or "", CRNN_TAKE_K)
            if img:
                out[img] = pred
        return out
    except Exception as e:
        log(f"⚠️ Could not read existing {filename}: {e}")
        return {}

def _write_csv(csv_dir: Path, filename: str, rows: List[Dict[str, str]]) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_text = _rows_to_csv_text(rows)
    (csv_dir / filename).write_text(csv_text, encoding="utf-8")
    log(f"📄 Wrote {filename} → {csv_dir.as_posix()}")


# ===================== RETRY WRAPPER =====================
def _call_with_retries(call_fn, name: str, image_bytes: io.BytesIO,
                       retries: int = ENDPOINT_RETRIES,
                       base: float = ENDPOINT_BACKOFF_BASE,
                       max_sleep: float = ENDPOINT_BACKOFF_MAX,
                       jitter: float = ENDPOINT_JITTER_SEC) -> Dict[str, Any]:
    """
    call_fn((name, image_bytes)) -> dict.
    Retries on dicts containing {"error": ...} (assumed transient) or raised exceptions.
    """
    attempt = 0
    last_err: Any = None
    while attempt <= retries:
        try:
            image_bytes.seek(0)
            resp = call_fn((name, image_bytes))
            if isinstance(resp, dict) and "error" in resp:
                msg = str(resp.get("error", ""))
                # Treat as transient if flagged or typical socket hints
                transient = bool(resp.get("transient")) or ("WinError" in msg) or ("connection" in msg.lower())
                if not transient:
                    return resp
                last_err = resp
                raise RuntimeError(msg)
            return resp  # success
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            sleep_s = min((base ** attempt), max_sleep) + random.uniform(0, max(0.0, jitter))
            time.sleep(sleep_s)
            attempt += 1
    # Exhausted
    if isinstance(last_err, dict):
        return last_err
    return {"error": str(last_err or "unknown error"), "transient": True}


# ===================== INFERENCE =====================
def _process_one(abs_input_dir: Path, img: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: {"image": rel_path, "pred_crnn": <code>, "ok": bool, "error": <str|None>}
    On failure, ok=False; caller won't overwrite CSV with placeholders.
    """
    rel = img["name"]
    try:
        t_dl0 = time.time()
        image_bytes = _load_local_image_bytes(abs_input_dir, img["path"])
        t_dl = time.time() - t_dl0

        if DRY_RUN:
            predicted = "?" * CRNN_TAKE_K
            log(f"🧪 DRY_RUN → skipping CRNN call for {rel}")
            return {"image": rel, "pred_crnn": predicted, "ok": True, "error": None}

        # Inference with retries
        t_inf0 = time.time()
        result = _call_with_retries(send_to_crnn, rel, image_bytes)
        t_inf = time.time() - t_inf0

        if not isinstance(result, dict) or "error" in result:
            raise ValueError(f"CRNN endpoint error/shape: {result!r}")

        raw = _extract_pred_from_result(result)
        if not raw:
            raise ValueError("Missing predicted text in CRNN response")

        predicted = _sanitize_pred(raw, CRNN_TAKE_K)
        log(f"🔍 {rel} → CRNN: {predicted} (dl {t_dl:.2f}s, inf {t_inf:.2f}s)")
        return {"image": rel, "pred_crnn": predicted, "ok": True, "error": None}

    except Exception as e:
        log(f"❌ Error with {rel}: {e}")
        return {"image": rel, "pred_crnn": "", "ok": False, "error": str(e)}


def run_inference(*_args, input_folder: Optional[str] = None, output_folder: Optional[str] = None, **_kwargs) -> List[Dict[str, str]]:
    """
    Backward-compatible entrypoint.
    Accepts stray positional/keyword args (ignored), and optional explicit paths:
      - input_folder: local images folder (overrides env)
      - output_folder: local CSV output folder (overrides env)
    Env fallback:
      - INPUT_FOLDER_ID or OUTPUT_CODE_FOLDER_ID
      - OUTPUT_OCR_FOLDER_ID
    """
    log("🚀 Starting CRNN inference job (LOCAL)…")

    in_folder = input_folder or os.getenv("OUTPUT_CODE_FOLDER_ID") or os.getenv("INPUT_FOLDER_ID")
    if not in_folder:
        raise EnvironmentError("❌ Set INPUT_FOLDER_ID (or OUTPUT_CODE_FOLDER_ID) to a local images folder.")
    abs_input_dir = Path(in_folder).resolve()
    if not abs_input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {abs_input_dir}")

    out_dir_env = output_folder or os.getenv("OUTPUT_OCR_FOLDER_ID", "").strip()
    if not out_dir_env:
        raise EnvironmentError("❌ Set OUTPUT_OCR_FOLDER_ID (or pass output_folder=) to a local folder for CSV output.")
    abs_csv_dir = Path(out_dir_env).resolve()
    abs_csv_dir.mkdir(parents=True, exist_ok=True)

    images = list_images_local(abs_input_dir)
    if not images:
        log("❌ No valid images found.")
        return []

    existing_map: Dict[str, str] = {}
    if SAVE_CRNN_CSV and not CRNN_FORCE:
        existing_map = _load_existing_csv_map(abs_csv_dir, CRNN_CSV_NAME)
        if existing_map:
            log(f"⏭️ Will skip {len(existing_map)} image(s) already present in {CRNN_CSV_NAME}")

    # Worklist
    todo = [img for img in images if img.get("name") not in existing_map]
    total = min(MAX_IMAGES, len(todo))

    t0 = time.time()
    new_rows_ok: List[Dict[str, str]] = []
    failed: List[str] = []

    if total:
        with ThreadPoolExecutor(max_workers=CRNN_MAX_WORKERS) as ex:
            futures = {ex.submit(_process_one, abs_input_dir, img): img for img in todo[:total]}
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                done += 1
                nm = res.get("image", "<unknown>")
                if res.get("ok"):
                    new_rows_ok.append({"image": nm, "pred_crnn": res["pred_crnn"]})
                    log(f"[{done}/{total}] ✅ {nm}")
                else:
                    failed.append(nm)
                    log(f"[{done}/{total}] ⚠️ {nm} (skipped due to error)")
    else:
        log("✅ Nothing new to process for CRNN.")

    # Merge: existing + new successes (new wins)
    merged: Dict[str, str] = dict(existing_map)
    for r in new_rows_ok:
        img = r.get("image")
        if img:
            merged[img] = _sanitize_pred(r.get("pred_crnn", ""), CRNN_TAKE_K)

    merged_rows = [{"image": k, "pred_crnn": v} for k, v in merged.items()]
    merged_rows.sort(key=lambda r: r.get("image", ""))

    if SAVE_CRNN_CSV:
        _write_csv(abs_csv_dir, CRNN_CSV_NAME, merged_rows)

    took = time.time() - t0
    if failed:
        log(f"✅ CRNN inference complete. {len(new_rows_ok)} new, {len(merged_rows)} total, {len(failed)} failed in {took:.2f}s.")
    else:
        log(f"✅ CRNN inference complete. {len(new_rows_ok)} new, {len(merged_rows)} total in {took:.2f}s.")
    return merged_rows


if __name__ == "__main__":
    run_inference()
