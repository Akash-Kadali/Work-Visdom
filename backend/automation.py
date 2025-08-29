# backend/automation.py

import os
import io
import csv
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload  # type: ignore
except Exception:  # pragma: no cover
    MediaIoBaseUpload = MediaIoBaseDownload = None  # type: ignore

from backend.drive_utils import authenticate, resolve_folder

try:
    from backend.inference_jobs import yolo_ocr, crnn_ocr
except Exception as e:  # pragma: no cover
    raise ImportError(f"Failed to import OCR inference modules: {e}")

# =========================
# Logging
# =========================
def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# =========================
# ENV & CONFIG
# =========================
def _choose_ocr_input_folder() -> str:
    """
    OCR input MUST be the code-crops folder produced by the crop job.
    Required: OUTPUT_CODE_FOLDER_ID (env key or literal path)
    """
    val = os.getenv("OUTPUT_CODE_FOLDER_ID", "").strip()
    if not val:
        raise EnvironmentError("❌ Set OUTPUT_CODE_FOLDER_ID (code crops folder) for OCR input.")
    return val

def _choose_csv_output_folder() -> str:
    """
    Consolidated OCR results (merged YOLO+CRNN) MUST be saved here.
    Required: OUTPUT_OCR_FOLDER_ID (env key or literal path)
    """
    dst = os.getenv("OUTPUT_OCR_FOLDER_ID", "").strip()
    if not dst:
        raise EnvironmentError("❌ Set OUTPUT_OCR_FOLDER_ID (folder for final OCR CSV).")
    return dst

def _propagate_ocr_input_env(folder_id_or_path: str) -> None:
    """
    Ensure both YOLO and CRNN jobs read the same folder for code crops.
    """
    os.environ["INPUT_FOLDER_ID"] = folder_id_or_path
    os.environ["INPUT_CODE_FOLDER_ID"] = folder_id_or_path  # legacy compatibility

FINAL_CSV_NAME    = os.getenv("FINAL_CSV_NAME", "ocr_results.csv").strip()
OCR_SENTINEL_NAME = os.getenv("OCR_SENTINEL_NAME", "OCR_DONE.txt").strip()
OCR_FORCE         = bool(int(os.getenv("OCR_FORCE", "0")))  # set to 1 to force re-run even if outputs exist

# Extensions & code-crop pattern (used for "new images" detection)
_ALLOWED_EXTS = tuple(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTS", ".jpg,.jpeg,.png,.bmp,.webp").split(",")
    if ext.strip()
)
_CODE_NAME_RE = re.compile(r"_code\.(?:png|jpe?g)$", re.IGNORECASE)

# =========================
# DRIVE/LOCAL HELPERS
# =========================
def _find_files_by_name(name: str, folder_id_or_path: str, drive_service) -> List[Dict]:
    """
    Returns a list of {id, name, ...}.
    Local mode: returns a single stub item when the file exists.
    """
    # Local mode
    if drive_service is None or not hasattr(drive_service, "files"):
        p = resolve_folder(folder_id_or_path) / name
        return ([{"id": "", "name": name, "path": p.as_posix()}] if p.exists() else [])

    # Drive mode
    q = f"'{folder_id_or_path}' in parents and name='{name}' and trashed=false"
    res = drive_service.files().list(
        q=q,
        fields="files(id, name, modifiedTime)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=10,
    ).execute()
    return res.get("files", []) or []

def _drive_file_exists(drive_service, folder_id_or_path: str, exact_name: str) -> bool:
    """
    True if file exists in the folder (local or Drive).
    """
    if not folder_id_or_path or not exact_name:
        return False

    # Local mode
    if drive_service is None or not hasattr(drive_service, "files"):
        return (resolve_folder(folder_id_or_path) / exact_name).exists()

    # Drive mode
    safe = exact_name.replace("'", "\\'")
    q = f"'{folder_id_or_path}' in parents and trashed=false and name='{safe}'"
    res = drive_service.files().list(
        q=q,
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=1,
    ).execute()
    return bool(res.get("files"))

def upload_or_update_single_csv(
    csv_text: str,
    filename: str,
    folder_id_or_path: str,
    drive_service,
) -> str:
    """
    Uploads or updates a single CSV in the target folder.
    Local mode: writes/overwrites a file on disk and returns its path.
    Drive mode: updates existing or creates a new file and returns its fileId.
    """
    # Local mode
    if drive_service is None or not hasattr(drive_service, "files") or MediaIoBaseUpload is None:
        out_dir = resolve_folder(folder_id_or_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_text(csv_text, encoding="utf-8")
        log(f"💾 Saved CSV locally → {out_path}")
        return out_path.as_posix()

    # Drive mode
    buf = io.BytesIO(csv_text.encode("utf-8"))
    media = MediaIoBaseUpload(buf, mimetype="text/csv", resumable=False)

    existing = _find_files_by_name(filename, folder_id_or_path, drive_service)
    if existing:
        file_id = existing[0]["id"]
        drive_service.files().update(
            fileId=file_id,
            media_body=media,
            body={"name": filename},
            supportsAllDrives=True,
        ).execute()
        # Enforce single-file policy
        for dup in existing[1:]:
            try:
                drive_service.files().delete(fileId=dup["id"]).execute()
            except Exception as e:
                log(f"⚠️ Couldn't delete duplicate {dup['id']}: {e}")
        log(f"♻️ Updated existing CSV → {filename}")
        return file_id

    meta = {"name": filename, "parents": [folder_id_or_path]}
    created = drive_service.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    log(f"📤 Uploaded CSV → {filename}")
    return created["id"]

def _upload_text_sentinel(drive_service, folder_id_or_path: str, name: str, text: str) -> Optional[str]:
    """
    Writes a small text sentinel.
    Local mode: writes a .txt file on disk and returns its path.
    Drive mode: uploads the file and returns its fileId.
    """
    # Local mode
    if drive_service is None or not hasattr(drive_service, "files") or MediaIoBaseUpload is None:
        try:
            out_dir = resolve_folder(folder_id_or_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            out_path.write_text(text, encoding="utf-8")
            return out_path.as_posix()
        except Exception as e:
            log(f"⚠️ Could not write local sentinel {name}: {e}")
            return None

    # Drive mode
    try:
        bio = io.BytesIO(text.encode("utf-8"))
        media = MediaIoBaseUpload(bio, mimetype="text/plain", resumable=False)
        meta = {"name": name, "parents": [folder_id_or_path], "mimeType": "text/plain"}
        created = drive_service.files().create(
            body=meta, media_body=media, fields="id", supportsAllDrives=True
        ).execute()
        return created.get("id")
    except Exception as e:
        log(f"⚠️ Could not write sentinel {name}: {e}")
        return None

def _list_code_images(folder_id_or_path: str, drive_service) -> List[str]:
    """
    Return the list of filenames that look like code crops: *_code.(png|jpg)
    """
    # Local mode
    if drive_service is None or not hasattr(drive_service, "files"):
        root = resolve_folder(folder_id_or_path)
        if not root.exists():
            return []
        names: List[str] = []
        for p in root.iterdir():
            if not p.is_file():
                continue
            lower = p.name.lower()
            if (lower.endswith(_ALLOWED_EXTS)) and _CODE_NAME_RE.search(lower):
                names.append(p.name)
        return names

    # Drive mode
    names: List[str] = []
    page_token = None
    q = f"'{folder_id_or_path}' in parents and trashed=false"
    fields = "nextPageToken, files(name, mimeType)"
    while True:
        resp = drive_service.files().list(
            q=q,
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        for f in resp.get("files", []) or []:
            name = (f.get("name") or "").strip()
            mime = f.get("mimeType") or ""
            lower = name.lower()
            if (("image" in mime) or lower.endswith(_ALLOWED_EXTS)) and _CODE_NAME_RE.search(lower):
                names.append(name)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return names

def _read_images_from_final_csv(folder_id_or_path: str, filename: str, drive_service) -> List[str]:
    """
    If final CSV exists, return the 'image' column so we can detect new images.
    """
    # Local mode
    if drive_service is None or not hasattr(drive_service, "files") or MediaIoBaseDownload is None:
        p = resolve_folder(folder_id_or_path) / filename
        if not p.exists():
            return []
        try:
            with p.open("r", encoding="utf-8", newline="") as fh:
                rdr = csv.DictReader(fh)
                return [row.get("image", "") for row in rdr if row.get("image")]
        except Exception:
            return []

    # Drive mode
    existing = _find_files_by_name(filename, folder_id_or_path, drive_service)
    if not existing:
        return []
    try:
        req = drive_service.files().get_media(fileId=existing[0]["id"])
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        csv_text = buf.getvalue().decode("utf-8", errors="replace")
        rdr = csv.DictReader(io.StringIO(csv_text))
        return [row.get("image", "") for row in rdr if row.get("image")]
    except Exception:
        return []

# =========================
# MERGE HELPERS
# =========================
def _to_map(rows: List[Dict], pred_key: str) -> Dict[str, Dict]:
    """
    Map by image name: image -> {pred_key: ...}
    """
    out: Dict[str, Dict] = {}
    for r in rows or []:
        img = r.get("image")
        if not img:
            continue
        out[img] = {pred_key: r.get(pred_key, "??????")}
    return out

def _build_merged_rows(yolo_rows: List[Dict], crnn_rows: List[Dict]) -> List[Dict]:
    """
    Merge on image and compute agreement flag.
    Output schema:
      image, pred_yolo, pred_crnn, agree
    """
    ymap = _to_map(yolo_rows, "pred_yolo")
    cmap = _to_map(crnn_rows, "pred_crnn")

    all_images = sorted(set(ymap.keys()) | set(cmap.keys()))
    merged: List[Dict] = []

    for img in all_images:
        py = (ymap.get(img) or {}).get("pred_yolo", "??????")
        pc = (cmap.get(img) or {}).get("pred_crnn", "??????")
        agree = "YES" if py == pc else "NO"
        merged.append({"image": img, "pred_yolo": py, "pred_crnn": pc, "agree": agree})

    return merged

def _rows_to_csv_text(rows: List[Dict]) -> str:
    """
    Serialize merged rows to CSV text in memory (UTF-8).
    """
    headers = ["image", "pred_yolo", "pred_crnn", "agree"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for r in rows or []:
        writer.writerow({k: r.get(k, "") for k in headers})
    return output.getvalue()

def _run_models_concurrently(drive_service) -> Tuple[List[Dict], List[Dict]]:
    """
    Kick off YOLO and CRNN in parallel and collect results.
    """
    results = {"yolo": [], "crnn": []}

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(yolo_ocr.run_inference, drive_service): "yolo",
            ex.submit(crnn_ocr.run_inference, drive_service): "crnn",
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result() or []
            except Exception as e:
                log(f"❌ {key.upper()} inference failed: {e}")
                results[key] = []

    return results["yolo"], results["crnn"]

# =========================
# PUBLIC API
# =========================
def run_full_pipeline() -> Dict:
    """
    Local/Drive idempotent OCR pipeline:
      - Uses OUTPUT_CODE_FOLDER_ID for OCR input (code crops)
      - Skips the run only if FINAL_CSV_NAME or sentinel exists AND there are no NEW code images
        (unless OCR_FORCE=1)
      - Runs YOLO + CRNN concurrently on code crops
      - Merges predictions
      - Writes/overwrites a single CSV (FINAL_CSV_NAME) to OUTPUT_OCR_FOLDER_ID
      - Writes a small sentinel text file (OCR_SENTINEL_NAME)
    Returns a summary dict.
    """
    log("🚀 Starting OCR pipeline (single-CSV mode)…")
    t0 = time.time()

    # Decide OCR input and CSV output
    ocr_input_folder = _choose_ocr_input_folder()
    csv_output_folder = _choose_csv_output_folder()
    _propagate_ocr_input_env(ocr_input_folder)

    # Auth once for Drive ops (may be None in LOCAL MODE)
    drive_service = authenticate()

    # Idempotency preflight — skip ONLY if no new images to process
    already_has_csv = _drive_file_exists(drive_service, csv_output_folder, FINAL_CSV_NAME)
    already_has_sentinel = _drive_file_exists(drive_service, csv_output_folder, OCR_SENTINEL_NAME)

    current_code_images = set(_list_code_images(ocr_input_folder, drive_service))
    prev_images_in_csv = set(_read_images_from_final_csv(csv_output_folder, FINAL_CSV_NAME, drive_service)) if already_has_csv else set()
    new_images = current_code_images - prev_images_in_csv
    log(f"🔎 OCR preflight: code_images={len(current_code_images)}, in_csv={len(prev_images_in_csv)}, new={len(new_images)}")

    if (already_has_csv or already_has_sentinel) and not OCR_FORCE and not new_images:
        msg = "✅ OCR results already exist and no new images detected. Skipping run."
        log(msg)
        return {
            "status": "skipped",
            "reason": "outputs_exist_no_new_images",
            "final_csv": FINAL_CSV_NAME,
            "sentinel": OCR_SENTINEL_NAME,
            "duration_sec": round(time.time() - t0, 2),
        }

    # Inference
    log(f"🔧 Running YOLO + CRNN inference on folder (code crops): {ocr_input_folder}")
    t_inf0 = time.time()
    yolo_rows, crnn_rows = _run_models_concurrently(drive_service)
    log(f"   ↳ YOLO rows: {len(yolo_rows)} | CRNN rows: {len(crnn_rows)} | took {(time.time() - t_inf0):.2f}s")

    if not yolo_rows and not crnn_rows:
        msg = "❌ No results from either model. Aborting."
        log(msg)
        return {
            "status": "error",
            "reason": "no_results",
            "duration_sec": round(time.time() - t0, 2),
        }

    # Merge
    log("📊 Merging results…")
    t_merge0 = time.time()
    merged_rows = _build_merged_rows(yolo_rows, crnn_rows)
    total = len(merged_rows)
    disagree = sum(1 for r in merged_rows if r["agree"] == "NO")
    log(f"   ↳ Total: {total}, Disagreements (YOLO≠CRNN): {disagree}, took {(time.time() - t_merge0):.2f}s")

    # Upload exactly one CSV (Drive or Local)
    csv_text = _rows_to_csv_text(merged_rows)
    t_up0 = time.time()
    file_ref = upload_or_update_single_csv(csv_text, FINAL_CSV_NAME, csv_output_folder, drive_service)
    log(f"📤 Wrote '{FINAL_CSV_NAME}' to OUTPUT_OCR_FOLDER_ID in {(time.time() - t_up0):.2f}s")

    # Sentinel
    _upload_text_sentinel(
        drive_service,
        csv_output_folder,
        OCR_SENTINEL_NAME,
        time.strftime("ocr_done_at=%Y-%m-%d %H:%M:%S")
    )

    duration = round(time.time() - t0, 2)
    log(f"✅ Pipeline finished in {duration:.2f}s. Single CSV saved.")
    return {
        "status": "ok",
        "total_rows": total,
        "disagreements": disagree,
        "final_csv": FINAL_CSV_NAME,
        "csv_file_ref": file_ref,  # fileId in Drive mode, path in Local mode
        "duration_sec": duration,
    }

# For direct execution
if __name__ == "__main__":
    run_full_pipeline()
