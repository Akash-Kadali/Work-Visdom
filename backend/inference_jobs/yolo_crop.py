# backend/inference_jobs/yolo_crop.py
# Saves only laser_die_3..18; skips 1,2,19,20. Idempotent. Works with Drive or Local.

from __future__ import annotations

import os
import io
import ssl
import zipfile
import time
import re
import threading
import requests
from datetime import datetime
from typing import List, Dict, Any, Iterable, Optional, Tuple, Set
from pathlib import Path  # FIX: used in _load_local_image_bytes

# googleapiclient is optional in Local mode
try:
    from googleapiclient.errors import HttpError  # type: ignore
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload  # type: ignore
except Exception:  # pragma: no cover
    class HttpError(Exception):  # minimal shim
        pass
    MediaIoBaseDownload = None  # type: ignore
    MediaIoBaseUpload = None  # type: ignore

from backend.drive_utils import authenticate  # returns Drive v3 client or None (Local mode)
from backend.paths import folder, IMAGES_DIR
from backend.local_store import list_images as _ls_list_images

# Try to load .env early (won't error if python-dotenv is missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# =======================
# CONFIG
# =======================
MAX_IMAGES       = int(os.getenv("MAX_IMAGES", 100000))
MAX_WORKERS      = int(os.getenv("MAX_WORKERS", 5))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
BACKOFF_BASE     = float(os.getenv("BACKOFF_BASE", 1.5))

# Input full images (Drive *or* Local via backend.paths.folder)
INPUT_IMAGE_FOLDER_ID = os.getenv("INPUT_IMAGE_FOLDER_ID", "").strip()  # required

# Choose ONE of these modes:
#   - OUTPUT_ZIP_FOLDER_ID: write ZIPs
#   - OUTPUT_LASER_FOLDER_ID + OUTPUT_CODE_FOLDER_ID: write extracted files
OUTPUT_ZIP_FOLDER_ID   = os.getenv("OUTPUT_ZIP_FOLDER_ID", "").strip()
OUTPUT_LASER_FOLDER_ID = os.getenv("OUTPUT_LASER_FOLDER_ID", "").strip()
OUTPUT_CODE_FOLDER_ID  = os.getenv("OUTPUT_CODE_FOLDER_ID", "").strip()

# Required: your crop endpoint (POST multipart/form-data "file")
CROP_URL = (os.getenv("CROP_URL") or os.getenv("CROP_ENDPOINT") or "").strip()

# Optional crop params (if your service reads query params)
CROP_CONF  = os.getenv("CROP_CONF", "0.20")
CROP_IMGSZ = os.getenv("CROP_IMGSZ", "1024")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", os.getenv("MODAL_TIMEOUT", 120.0)))

# Allowed input extensions when scanning Local folders
_ALLOWED_EXTS = tuple(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTS", ".jpg,.jpeg,.png,.bmp,.webp").split(",")
    if ext.strip()
)

# Idempotency controls
CROP_FORCE         = bool(int(os.getenv("CROP_FORCE", "0")))  # set to 1 to ignore existing per-file outputs
CROP_SENTINEL_NAME = os.getenv("CROP_SENTINEL_NAME", "CROP20_DONE.txt").strip()

# Laser index range to save (inclusive). Default: 3..18 (skip 1,2,19,20).
LASER_MIN_INDEX = int(os.getenv("LASER_MIN_INDEX", "3"))
LASER_MAX_INDEX = int(os.getenv("LASER_MAX_INDEX", "18"))

# Regexes to pick inner files from ZIP
_LASER_RE = re.compile(r"laser[_\-]?die[_\-]?(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)
_CODE_RE  = re.compile(r"_code\.(?:png|jpe?g)$", re.IGNORECASE)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _debug_env_once() -> None:
    """Log key envs once to help diagnose misconfig (safe to call repeatedly)."""
    log(
        "ENV CHECK → "
        f"INPUT_IMAGE_FOLDER_ID={'set' if INPUT_IMAGE_FOLDER_ID else 'MISSING'}, "
        f"OUTPUT_LASER_FOLDER_ID={'set' if OUTPUT_LASER_FOLDER_ID else 'unset'}, "
        f"OUTPUT_CODE_FOLDER_ID={'set' if OUTPUT_CODE_FOLDER_ID else 'unset'}, "
        f"OUTPUT_ZIP_FOLDER_ID={'set' if OUTPUT_ZIP_FOLDER_ID else 'unset'}, "
        f"CROP_URL={'set' if CROP_URL else 'MISSING'}, "
        f"CROP_CONF={CROP_CONF}, CROP_IMGSZ={CROP_IMGSZ}, "
        f"LASER_MIN_INDEX={LASER_MIN_INDEX}, LASER_MAX_INDEX={LASER_MAX_INDEX}"
    )

# =======================
# DRIVE/LOCAL HELPERS
# =======================

def _canon(name: str) -> str:
    """Canonicalize names so .jpeg and .jpg map together."""
    if name.lower().endswith(".jpeg"):
        return name[:-5] + ".jpg"
    return name


def _drive_list_images(folder_id_or_path: str, drive_service=None) -> Iterable[Dict[str, Any]]:
    """
    Iterate image file dicts from either:
      - Local folder (when drive_service is None or lacks .files), or
      - Google Drive (when drive_service has .files()).

    Yields Drive-shaped dicts: {"name", "id", "path"(local-only)}.
    """
    # -------- Local mode --------
    if drive_service is None or not hasattr(drive_service, "files"):
        base = folder(str(folder_id_or_path))
        if not base or not base.exists():
            return
        for p in _ls_list_images(base):
            ext = p.suffix.lower()
            if ext not in _IMG_EXTS:
                continue
            try:
                rel = p.relative_to(IMAGES_DIR).as_posix()
            except Exception:
                rel = p.as_posix()
            yield {"name": p.name, "id": "", "path": rel}
        return

    # -------- Drive mode --------
    page_token: Optional[str] = None
    q = f"'{folder_id_or_path}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime)"
    while True:
        resp = drive_service.files().list(
            q=q,
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token,
        ).execute()

        for f in resp.get("files", []) or []:
            name = f.get("name", "")
            mime = f.get("mimeType", "") or ""
            if "image" in mime or name.lower().endswith(_ALLOWED_EXTS):
                yield f

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def list_images(folder_id: str) -> List[Dict[str, Any]]:
    """Return a list of Drive-shaped image dicts from either Local or Drive."""
    svc = authenticate()
    return list(_drive_list_images(folder_id, svc))


def _list_names(folder_id_or_path: str, drive_service) -> Set[str]:
    """Return a set of file names in a folder (canonicalized: .jpeg → .jpg)."""
    names: Set[str] = set()
    if not folder_id_or_path:
        return names

    # Local mode
    if drive_service is None or not hasattr(drive_service, "files"):
        base = folder(str(folder_id_or_path))
        if base and base.exists():
            for p in _ls_list_images(base):
                names.add(_canon(p.name))
        return names

    # Drive mode
    page_token = None
    q = f"'{folder_id_or_path}' in parents and trashed=false"
    fields = "nextPageToken, files(name)"
    while True:
        resp = drive_service.files().list(
            q=q,
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []) or []:
            n = (f.get("name") or "").strip()
            if n:
                names.add(_canon(n))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return names


def _drive_file_exists_by_name(folder_id_or_path: str, exact_name: str, drive_service) -> bool:
    # Local
    if drive_service is None or not hasattr(drive_service, "files"):
        base = folder(str(folder_id_or_path))
        return bool(base and (base / exact_name).exists())
    # Drive
    safe = exact_name.replace("'", "\\'")
    q = f"'{folder_id_or_path}' in parents and trashed=false and name='{safe}'"
    resp = drive_service.files().list(
        q=q, fields="files(id)", pageSize=1, supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    return bool(resp.get("files"))


def _load_local_image_bytes(path_or_rel: str) -> io.BytesIO:
    """
    Load a local image by absolute path or path relative to IMAGES_DIR.
    """
    p = os.path.normpath(path_or_rel)
    if not os.path.isabs(p):
        full = (IMAGES_DIR / p).resolve()
    else:
        full = Path(os.path.abspath(p))
    with open(str(full), "rb") as f:
        data = f.read()
    return io.BytesIO(data)


def download_image(file_id_or_path: str, drive_service=None) -> io.BytesIO:
    """
    Download an image from Drive (drive_service) or load from local path.
    Accepts Drive file_id (Drive mode) or a relative/absolute path (Local mode).
    Retries only apply to Drive mode.
    """
    # Local path (no retries required)
    if drive_service is None or not hasattr(drive_service, "files"):
        return _load_local_image_bytes(file_id_or_path)

    # Drive mode with retries
    attempt = 0
    while attempt < DOWNLOAD_RETRIES:
        try:
            svc = drive_service or authenticate()
            request = svc.files().get_media(fileId=file_id_or_path)
            fh = io.BytesIO()
            dl = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = dl.next_chunk()
            fh.seek(0)
            return fh
        except HttpError as he:
            attempt += 1
            if attempt >= DOWNLOAD_RETRIES:
                raise
            sleep_s = BACKOFF_BASE ** attempt
            log(f"⚠️ Drive HttpError (attempt {attempt}/{DOWNLOAD_RETRIES}): {he}. Retrying in {sleep_s:.1f}s…")
            time.sleep(sleep_s)
        except Exception as e:
            attempt += 1
            if attempt >= DOWNLOAD_RETRIES:
                raise
            sleep_s = BACKOFF_BASE ** attempt
            if _is_tls_error(e):
                log(f"⚠️ TLS/SSL error (attempt {attempt}/{DOWNLOAD_RETRIES}): {e}. Retrying in {sleep_s:.1f}s…")
            else:
                log(f"⚠️ Download error (attempt {attempt}/{DOWNLOAD_RETRIES}): {e}. Retrying in {sleep_s:.1f}s…")
            time.sleep(sleep_s)
    raise RuntimeError("Unreachable: download retries exhausted")


def _is_tls_error(e: Exception) -> bool:
    s = str(e)
    return (
        "WRONG_VERSION_NUMBER" in s
        or "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in s
        or isinstance(e, ssl.SSLError)
    )


def upload_bytes_as_file(folder_id_or_path: str, name: str, data: bytes, mime_type: str) -> str:
    """
    Write bytes to either:
      - Local folder (returns absolute path) when Drive is not configured.
      - Google Drive (returns file id) when service is available.
    """
    svc = authenticate()

    # Local mode
    if svc is None or not hasattr(svc, "files") or MediaIoBaseUpload is None:
        out_dir = folder(str(folder_id_or_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / name).resolve()
        with open(out_path, "wb") as f:
            f.write(data)
        return str(out_path)

    # Drive mode
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
    meta = {"name": name, "parents": [folder_id_or_path]}
    created = svc.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    return created["id"]


def _upload_text_sentinel(folder_id_or_path: str, name: str, text: str) -> Optional[str]:
    """
    Create/overwrite a small text sentinel file in either local or Drive folder.
    Returns the path (local) or file id (Drive), or None on error.
    """
    try:
        svc = authenticate()
        # Local
        if svc is None or not hasattr(svc, "files") or MediaIoBaseUpload is None:
            out_dir = folder(str(folder_id_or_path))
            out_dir.mkdir(parents=True, exist_ok=True)
            p = (out_dir / name).resolve()
            with open(p, "w", encoding="utf-8") as f:
                f.write(text or "")
            return str(p)
        # Drive
        bio = io.BytesIO((text or "").encode("utf-8"))
        media = MediaIoBaseUpload(bio, mimetype="text/plain", resumable=False)
        meta = {"name": name, "parents": [folder_id_or_path], "mimeType": "text/plain"}
        created = svc.files().create(
            body=meta, media_body=media, fields="id", supportsAllDrives=True
        ).execute()
        return created.get("id")
    except Exception:
        return None

# =======================
# MODAL CLIENT
# =======================
def _crop_endpoint_url() -> str:
    if not CROP_URL:
        raise EnvironmentError("CROP_URL (or CROP_ENDPOINT) env var is required.")
    url = CROP_URL
    # Append query params (conf, imgsz) only if not already present
    if ("conf=" not in url) and ("imgsz=" not in url):
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}conf={CROP_CONF}&imgsz={CROP_IMGSZ}"
    return url


def send_to_crop(name_and_bytes: Tuple[str, io.BytesIO]) -> bytes:
    """
    POST to crop endpoint, return raw ZIP bytes.
    The endpoint returns a ZIP with 20 laser crops and 1 code crop.
    """
    url = _crop_endpoint_url()
    name, bio = name_and_bytes
    bio.seek(0)
    files = {"file": (name, bio, "application/octet-stream")}
    resp = requests.post(url, files=files, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Crop endpoint error {resp.status_code}: {resp.text[:300]}")
    return resp.content  # ZIP bytes

# =======================
# CORE PIPELINE
# =======================
def _should_split_upload() -> bool:
    return bool(OUTPUT_LASER_FOLDER_ID and OUTPUT_CODE_FOLDER_ID)


def _infer_ext_and_mime(inner_name: str) -> Tuple[str, str]:
    """Return (extension_with_dot, mime) based on inner filename."""
    lower = inner_name.lower()
    if lower.endswith(".png"):
        return ".png", "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return ".jpg", "image/jpeg"  # FIX: correct MIME
    # default/fallback
    return ".jpg", "image/jpeg"


def _exists(names: Set[str], name: str) -> bool:
    """Check if a name already exists (canonicalized .jpeg → .jpg)."""
    return _canon(name) in names


def _add_name(names: Set[str], name: str) -> None:
    names.add(_canon(name))


def _upload_extracted(
    base_name: str,
    zip_bytes: bytes,
    code_names: Set[str],
    laser_names: Set[str],
    names_lock: threading.Lock,
) -> Dict[str, int]:
    """
    Extract ZIP in-memory and upload files with enforced naming while preserving extension:
      - <base>_laser_die_<i>.(png|jpg) -> OUTPUT_LASER_FOLDER_ID  [ONLY i in LASER_MIN_INDEX..LASER_MAX_INDEX]
      - <base>_code.(png|jpg)          -> OUTPUT_CODE_FOLDER_ID
    Skips files that already exist (idempotent).
    Returns counts actually uploaded (not total present).
    """
    if not OUTPUT_CODE_FOLDER_ID and not OUTPUT_LASER_FOLDER_ID:
        return {"lasers": 0, "codes": 0}

    lasers = 0
    codes = 0
    next_idx = 1  # track next laser index in case filenames lack indices

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info)
            inner_name = os.path.basename(info.filename)
            inner_lower = inner_name.lower()

            # Detect code vs laser
            is_code = bool(_CODE_RE.search(inner_lower)) or inner_lower.startswith("unique_code")
            ext, mime = _infer_ext_and_mime(inner_lower)

            if is_code:
                if OUTPUT_CODE_FOLDER_ID:
                    out_name = f"{base_name}_code{ext}"
                    with names_lock:
                        if _exists(code_names, out_name):
                            continue
                    upload_bytes_as_file(OUTPUT_CODE_FOLDER_ID, out_name, data, mime)
                    with names_lock:
                        _add_name(code_names, out_name)
                    codes += 1
                continue

            # Laser: parse index from inner filename; else use running counter
            m = _LASER_RE.search(inner_lower)
            if m:
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = next_idx
            else:
                idx = next_idx

            # Skip if outside desired range (default keeps 3..18)
            if not (LASER_MIN_INDEX <= idx <= LASER_MAX_INDEX):
                next_idx = max(next_idx, idx + 1)
                continue

            if OUTPUT_LASER_FOLDER_ID:
                out_name = f"{base_name}_laser_die_{idx}{ext}"
                with names_lock:
                    if _exists(laser_names, out_name):
                        pass
                    else:
                        upload_bytes_as_file(OUTPUT_LASER_FOLDER_ID, out_name, data, mime)
                        _add_name(laser_names, out_name)
                        lasers += 1

            next_idx = max(next_idx, idx + 1)

    return {"lasers": lasers, "codes": codes}


def _all_selected_lasers_exist(base: str, laser_names: Set[str]) -> bool:
    """
    Check that lasers LASER_MIN_INDEX..LASER_MAX_INDEX exist (either .png or .jpg).
    """
    for i in range(LASER_MIN_INDEX, LASER_MAX_INDEX + 1):
        a = f"{base}_laser_die_{i}.png"
        b = f"{base}_laser_die_{i}.jpg"
        if not (_exists(laser_names, a) or _exists(laser_names, b)):
            return False
    return True


def _code_exists(base: str, code_names: Set[str]) -> bool:
    # FIX: check both jpg and png
    return _exists(code_names, f"{base}_code.jpg") or _exists(code_names, f"{base}_code.png")


def _image_bytes_for(img: Dict[str, Any], drive_service=None) -> io.BytesIO:
    """
    Get BytesIO for the given image dict (local or drive).
    Local dicts include {'path': 'rel/path', 'id': ''}; Drive include an 'id'.
    """
    file_id = (img.get("id") or "").strip()
    if file_id:
        return download_image(file_id, drive_service)
    # Local path fallback
    path_rel = img.get("path") or img.get("name")
    if not path_rel:
        raise RuntimeError("Image has neither Drive id nor local path.")
    p = path_rel
    # Load from disk:
    if not os.path.isabs(p):
        full_path = (IMAGES_DIR / p).resolve()
    else:
        full_path = p
    with open(full_path, "rb") as f:
        return io.BytesIO(f.read())


def process_single_image(
    img: Dict[str, Any],
    code_names: Set[str],
    laser_names: Set[str],
    names_lock: threading.Lock,
) -> Dict[str, Any]:
    name = img["name"]
    base = os.path.splitext(name)[0]
    try:
        # If outputs already complete and not forcing, skip this image
        if _should_split_upload() and not CROP_FORCE:
            if _code_exists(base, code_names) and _all_selected_lasers_exist(base, laser_names):
                return {"image": name, "skipped": True, "reason": "outputs_exist"}

        svc = authenticate()

        t_dl0 = time.time()
        image_bytes = _image_bytes_for(img, svc)
        t_dl = time.time() - t_dl0

        t_inf0 = time.time()
        zip_bytes = send_to_crop((name, image_bytes))
        t_inf = time.time() - t_inf0

        result: Dict[str, Any] = {
            "image": name,
            "download_s": round(t_dl, 3),
            "infer_s": round(t_inf, 3),
        }

        if _should_split_upload():
            counts = _upload_extracted(base, zip_bytes, code_names, laser_names, names_lock)
            result.update({"mode": "split", **counts})
            log(
                f"📦 {name}: uploaded {counts['lasers']} lasers (idx {LASER_MIN_INDEX}..{LASER_MAX_INDEX}), "
                f"{counts['codes']} code crops (dl {t_dl:.2f}s, inf {t_inf:.2f}s)"
            )
        elif OUTPUT_ZIP_FOLDER_ID:
            zip_name = f"{base}_crops.zip"
            upload_bytes_as_file(OUTPUT_ZIP_FOLDER_ID, zip_name, zip_bytes, "application/zip")
            result.update({"mode": "zip"})
            log(f"🗜️ {name}: ZIP uploaded (dl {t_dl:.2f}s, inf {t_inf:.2f}s)")
        else:
            result.update({"mode": "none"})
            log(f"⚠️ No output folder env set; processed {name} but did not write results.")

        return result

    except Exception as e:
        log(f"❌ Error with {name}: {e}")
        return {"image": name, "error": str(e)}


def run_inference(drive_service_unused=None) -> List[Dict[str, Any]]:
    """
    Entry point. Reads INPUT_IMAGE_FOLDER_ID, processes up to MAX_IMAGES with MAX_WORKERS threads.
    If OUTPUT_ZIP_FOLDER_ID is set, writes ZIPs.
    If OUTPUT_LASER_FOLDER_ID & OUTPUT_CODE_FOLDER_ID are set, writes extracted crops with enforced naming.
    Idempotent per-file: skips already-present outputs; writes a sentinel when the run completes.
    NOTE: Does NOT abort the job based on a pre-existing sentinel — new images will still be processed.
    """
    _debug_env_once()

    if not INPUT_IMAGE_FOLDER_ID:
        raise EnvironmentError("❌ INPUT_IMAGE_FOLDER_ID is not set.")
    if not CROP_URL:
        raise EnvironmentError("❌ CROP_URL (or CROP_ENDPOINT) is not set.")

    log("🚀 Starting YOLO Crop-20 inference job…")
    images = list_images(INPUT_IMAGE_FOLDER_ID)
    if not images:
        log("❌ No valid images found.")
        return []

    total = min(MAX_IMAGES, len(images))

    # Preload existing names for idempotency (Drive or Local)
    svc = authenticate()
    code_names  = _list_names(OUTPUT_CODE_FOLDER_ID,  svc) if OUTPUT_CODE_FOLDER_ID else set()
    laser_names = _list_names(OUTPUT_LASER_FOLDER_ID, svc) if OUTPUT_LASER_FOLDER_ID else set()
    names_lock = threading.Lock()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_single_image, img, code_names, laser_names, names_lock): img
            for img in images[:total]
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            rows.append(res)
            nm = res.get("image", "<unknown>")
            ok = "error" not in res
            if res.get("skipped"):
                log(f"[{i}/{total}] ⏭️ {nm} (skipped: outputs exist)")
            else:
                log(f"[{i}/{total}] {'✅' if ok else '❌'} {nm}")

    rows.sort(key=lambda r: r.get("image", ""))

    # Write/overwrite sentinel in the *code* output folder on completion (best-effort)
    if OUTPUT_CODE_FOLDER_ID and CROP_SENTINEL_NAME:
        _upload_text_sentinel(
            OUTPUT_CODE_FOLDER_ID,
            CROP_SENTINEL_NAME,
            time.strftime("crop_done_at=%Y-%m-%d %H:%M:%S")
        )

    log(f"✅ Crop job complete. {len(rows)} rows in {time.time() - t0:.2f}s.")
    return rows


if __name__ == "__main__":
    # For ad-hoc local runs
    run_inference()
