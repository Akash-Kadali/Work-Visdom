from __future__ import annotations

import os
import io
import csv
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# --- env / dotenv ----------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

def _env(key: str, default: str = "") -> str:
    """Read env fresh each call (no stale globals)."""
    return os.getenv(key, default).strip()

# -------------------------------------------------------------------
# General options
# -------------------------------------------------------------------
MAX_IMAGES = int(_env("MAX_IMAGES", "1000000"))

_ALLOWED_EXTS = tuple(
    ext.strip().lower()
    for ext in _env("ALLOWED_EXTS", ".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff").split(",")
    if ext.strip()
)

# Input folders for split crops (LOCAL directories)
def _laser_dir() -> str:    return _env("OUTPUT_LASER2_FOLDER_ID",   "data/images/regions/laser")
def _critical_dir() -> str: return _env("OUTPUT_CRITICAL_FOLDER_ID", "data/images/regions/critical")
def _body_dir() -> str:     return _env("OUTPUT_BODY_FOLDER_ID",     "data/images/regions/body")

# Folder to save CSVs to (LOCAL directory)
def _classify_dir() -> str: return _env("OUTPUT_CLASSIFY_FOLDER_ID", "data/csv")

# Default CSV names (override via env if desired)
def _LASER_SWINV2_CSV()   -> str: return _env("LASER_SWINV2_CSV",   "laser_swinv2_results.csv")
def _LASER_COAT_CSV()     -> str: return _env("LASER_COAT_CSV",     "laser_coat_results.csv")
def _LASER_MAXVIT_CSV()   -> str: return _env("LASER_MAXVIT_CSV",   "laser_maxvit_results.csv")
def _LASER_COMBINED_CSV() -> str: return _env("LASER_COMBINED_CSV", "laser_all_models.csv")

def _CRIT_SWINV2_CSV()    -> str: return _env("CRIT_SWINV2_CSV",    "critical_swinv2_results.csv")
def _CRIT_COAT_CSV()      -> str: return _env("CRIT_COAT_CSV",      "critical_coat_results.csv")
def _CRIT_MAXVIT_CSV()    -> str: return _env("CRIT_MAXVIT_CSV",    "critical_maxvit_results.csv")
def _CRIT_COMBINED_CSV()  -> str: return _env("CRIT_COMBINED_CSV",  "critical_all_models.csv")

# Body CSV name (single-model)
def _BODY_COAT_CSV_NAME() -> str: return _env("BODY_COAT_CSV_NAME", "body_coat_results.csv")

# -------------------------------------------------------------------
# Local path helpers
# -------------------------------------------------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _list_images(root: str | Path, max_images: Optional[int] = None) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    exts = _ALLOWED_EXTS
    out: List[Path] = []
    for ext in exts:
        out.extend(root.rglob(f"*{ext}"))
    out.sort()
    if max_images is not None:
        out = out[:max_images]
    return out

# -------------------------------------------------------------------
# HTTP layer (PATCH): persistent sessions + retries + backoff
# -------------------------------------------------------------------
# We keep the public surface `_post_image(url, (name, BytesIO))` intact,
# but the implementation now uses requests.Session with connection pooling,
# keep-alive, and robust retry logic to avoid WinError 10053 aborts.
HAVE_REQUESTS = True
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry  # type: ignore
except Exception:
    HAVE_REQUESTS = False

from urllib.parse import urlsplit

# Tunables (via env)
_HTTP_CONNECT_TIMEOUT = float(_env("HTTP_CONNECT_TIMEOUT", "5"))     # seconds
_HTTP_READ_TIMEOUT    = float(_env("HTTP_READ_TIMEOUT", "60"))       # seconds
_HTTP_TOTAL_RETRIES   = int(_env("HTTP_TOTAL_RETRIES", "5"))
_HTTP_BACKOFF_FACTOR  = float(_env("HTTP_BACKOFF_FACTOR", "0.6"))
_HTTP_POOL_CONNS      = int(_env("HTTP_POOL_CONNECTIONS", "12"))
_HTTP_POOL_MAXSIZE    = int(_env("HTTP_POOL_MAXSIZE", "12"))
_HTTP_RETRY_STATUSES  = tuple(
    int(x) for x in _env("HTTP_RETRY_STATUSES", "408,429,500,502,503,504").split(",") if x.strip()
)

# One session per origin (scheme://host[:port]) to reuse TLS and sockets
_SESSIONS: Dict[str, "requests.Session"] = {}

def _origin_key(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}"

def _build_session() -> "requests.Session":
    s = requests.Session()
    retry = Retry(
        total=_HTTP_TOTAL_RETRIES,
        connect=_HTTP_TOTAL_RETRIES,
        read=_HTTP_TOTAL_RETRIES,
        status=_HTTP_TOTAL_RETRIES,
        backoff_factor=_HTTP_BACKOFF_FACTOR,
        status_forcelist=_HTTP_RETRY_STATUSES,
        allowed_methods=frozenset(["POST", "GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=_HTTP_POOL_CONNS, pool_maxsize=_HTTP_POOL_MAXSIZE)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Accept": "application/json"})
    return s

def _session_for(url: str) -> "requests.Session":
    key = _origin_key(url)
    sess = _SESSIONS.get(key)
    if sess is None:
        sess = _build_session()
        _SESSIONS[key] = sess
    return sess

def _requests_post_image(url: str, name_and_bytes: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
    """Primary path: multipart/form-data with keep-alive and retries via adapter."""
    filename, bio = name_and_bytes
    bio.seek(0)
    files = {"file": (filename or "image.jpg", bio, "application/octet-stream")}
    try:
        r = _session_for(url).post(
            url,
            files=files,
            timeout=(_HTTP_CONNECT_TIMEOUT, _HTTP_READ_TIMEOUT),
        )
    except Exception as e:
        # Transport-level error
        return {"error": f"Request failed for {url}: {e}", "transient": True}

    # Non-2xx -> may still have JSON body; parse if possible
    text = ""
    try:
        text = r.text
        data = r.json()
    except Exception:
        data = None

    # 2xx with JSON → good
    if 200 <= r.status_code < 300 and isinstance(data, dict):
        return data

    # If 2xx but bad json, surface raw body
    if 200 <= r.status_code < 300:
        return {"error": f"Bad JSON from {url}", "status": r.status_code, "raw": text, "transient": False}

    # Non-2xx
    transient = r.status_code in _HTTP_RETRY_STATUSES
    if isinstance(data, dict):
        # If server already formats {"error": "..."} keep it
        if "error" in data and isinstance(data["error"], str):
            return {"error": data["error"], "status": r.status_code, "transient": transient, "raw": data}
    return {"error": f"HTTP {r.status_code} for {url}", "raw": text, "transient": transient}

# Legacy urllib fallback if requests is missing (unlikely in your env)
import urllib.request
import urllib.error

def _urllib_post_octet(url: str, name_and_bytes: Tuple[str, io.BytesIO]) -> Dict[str, Any]:
    name, bio = name_and_bytes
    data = bio.getvalue()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream", "X-Filename": name},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_CONNECT_TIMEOUT + _HTTP_READ_TIMEOUT) as r:
            body = r.read().decode("utf-8", "ignore")
        try:
            return json.loads(body)
        except Exception:
            return {"error": f"Bad JSON from {url}", "raw": body}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        return {"error": f"HTTP {e.code} for {url}", "body": body, "transient": e.code in _HTTP_RETRY_STATUSES}
    except Exception as e:
        return {"error": f"Request failed for {url}: {e}", "transient": True}

def _urllib_post_multipart(url: str, name_and_bytes: Tuple[str, io.BytesIO], field_name: str = "file") -> Dict[str, Any]:
    filename, bio = name_and_bytes
    boundary = f"----VisdomBoundary{int(time.time()*1000)}"
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    parts = []
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.append(bio.getvalue())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_CONNECT_TIMEOUT + _HTTP_READ_TIMEOUT) as r:
            payload = r.read().decode("utf-8", "ignore")
        try:
            return json.loads(payload)
        except Exception:
            return {"error": f"Bad JSON from {url}", "raw": payload}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        return {"error": f"HTTP {e.code} for {url}", "body": body, "transient": e.code in _HTTP_RETRY_STATUSES}
    except Exception as e:
        return {"error": f"Request failed for {url}: {e}", "transient": True}

def _sleep_with_jitter(base: float, attempt: int) -> None:
    # Exponential backoff with a little jitter
    delay = base * (2 ** (attempt - 1))
    delay *= 0.75 + 0.5 * random.random()
    time.sleep(delay)

def _post_image(url: str, name_and_bytes: tuple[str, io.BytesIO], retries: int = 3, backoff: float = 0.6) -> dict:
    """POST an image with retries. Returns JSON dict or {'error': '...', 'transient': bool}."""
    if not url:
        return {"error": "Empty URL", "transient": False}

    for attempt in range(1, max(1, retries) + 1):
        if HAVE_REQUESTS:
            res = _requests_post_image(url, name_and_bytes)
        else:
            # legacy fallback path: try multipart first, then octet-stream
            res = _urllib_post_multipart(url, name_and_bytes)
            if res.get("error"):
                res = _urllib_post_octet(url, name_and_bytes)

        err = (res or {}).get("error")
        if not err:
            return res  # success

        transient = bool(res.get("transient", False))
        # If 4xx (except 408/429) and we have raw/body complaining about form encoding,
        # try the alternate body shape once (requests path already uses multipart).
        if HAVE_REQUESTS and (not transient) and attempt == 1:
            # Attempt an octet-stream fallback once in the requests path:
            # send as raw bytes with header if server expects it (rare).
            try:
                filename, bio = name_and_bytes
                bio.seek(0)
                r = _session_for(url).post(
                    url,
                    data=bio.read(),
                    headers={"Content-Type": "application/octet-stream", "X-Filename": filename},
                    timeout=(_HTTP_CONNECT_TIMEOUT, _HTTP_READ_TIMEOUT),
                )
                try:
                    data = r.json()
                except Exception:
                    data = None
                if 200 <= r.status_code < 300 and isinstance(data, dict):
                    return data
                # Fall through to retry logic
            except Exception:
                pass

        if attempt < retries and transient:
            _sleep_with_jitter(backoff, attempt)
            continue

        # Final failure
        return {"error": err, "transient": transient}

# -------------------------------------------------------------------
# Runtime URL getters (fixes “read at import-time” bug)
# -------------------------------------------------------------------
def _yolo_url() -> str: return _env("YOLO_URL")
def _crnn_url() -> str: return _env("CRNN_URL")

def _urls_for(prefix: str) -> Dict[str, str]:
    if prefix == "laser":
        return {
            "swinv2": _env("SWINV2_LASER_URL"),
            "coat":   _env("COAT_LASER_URL"),
            "maxvit": _env("MAXVIT_LASER_URL"),
        }
    return {
        "swinv2": _env("SWINV2_CRITICAL_URL"),
        "coat":   _env("COAT_CRITICAL_URL"),
        "maxvit": _env("MAXVIT_CRITICAL_URL"),
    }

def _body_endpoint() -> str: return _env("COAT_BODY_URL")

# -------------------------------------------------------------------
# Public helpers kept for YOLO / CRNN (LOCAL or remote HTTP, but NO Drive)
# -------------------------------------------------------------------
def send_to_yolo(name_and_bytes: Tuple[str, io.BytesIO], endpoint: Optional[str] = None) -> Dict[str, Any]:
    url = (endpoint or _yolo_url() or "").strip()
    if not url:
        return {"error": "YOLO_URL is not configured"}
    return _post_image(url, name_and_bytes)

def send_to_crnn(name_and_bytes: Tuple[str, io.BytesIO], endpoint: Optional[str] = None) -> Dict[str, Any]:
    url = (endpoint or _crnn_url() or "").strip()
    if not url:
        return {"error": "CRNN_URL is not configured"}
    return _post_image(url, name_and_bytes)

# -------------------------------------------------------------------
# CSV / label helpers
# -------------------------------------------------------------------
def _normalize_label(raw: Any) -> str:
    s = (str(raw or "").strip().lower())
    if s in ("good", "ok", "okay", "0", "pass", "clean"):
        return "GOOD"
    if s in ("bad", "defect", "defective", "1", "fail", "dirty", "ng"):
        return "DEFECTIVE"
    return "DEFECTIVE"

def _extract_label_and_score(resp: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """Return ('', None) if the response is an error so callers can SKIP writing a row."""
    if not isinstance(resp, dict) or resp.get("error"):
        return "", None  # <-- critical: don't convert errors into predictions
    label = resp.get("label", resp.get("prediction", resp.get("class", "")))
    score = resp.get("score", resp.get("confidence", resp.get("prob", None)))
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    return _normalize_label(label), score

def _rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=["image", "classification", "score", "model"], quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({
            "image": r.get("image", ""),
            "classification": r.get("classification", ""),
            "score": "" if r.get("score") is None else r.get("score"),
            "model": r.get("model", ""),
        })
    return sio.getvalue()

def _rows_to_combined_csv_general(rows: List[Dict[str, Any]], model_order: List[str]) -> str:
    fields = ["image"]
    for m in model_order:
        fields += [f"{m}_label", f"{m}_score"]
    fields += ["consensus"]

    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        out = {"image": r.get("image", ""), "consensus": r.get("consensus", "")}
        for m in model_order:
            out[f"{m}_label"] = r.get(f"{m}_label", "")
            val = r.get(f"{m}_score", None)
            out[f"{m}_score"] = "" if val is None else val
        w.writerow(out)
    return sio.getvalue()

def _consensus_from_labels(labels: List[str]) -> str:
    labs = set(labels)
    if labs == {"GOOD"}:
        return "ALL_GOOD"
    if labs == {"DEFECTIVE"}:
        return "ALL_DEFECTIVE"
    if "DEFECTIVE" in labs:
        return "ANY_DEFECTIVE"
    return "MIXED"

def _dedup_by_image(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    for r in rows:
        img = (r.get("image") or "").strip()
        if not img:
            continue
        seen[img] = r  # last win
    return [seen[k] for k in sorted(seen.keys())]

# -------------------------------------------------------------------
# Local CSV I/O (no Drive): read/write under OUTPUT_CLASSIFY_FOLDER_ID
# -------------------------------------------------------------------
def _read_local_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    data = csv_path.read_text(encoding="utf-8", errors="ignore")
    if not data.strip():
        return []
    try:
        return list(csv.DictReader(io.StringIO(data)))
    except Exception:
        return []

def _write_local_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(_rows_to_csv(rows), encoding="utf-8")

# -------------------------------------------------------------------
# Combined CSV (local): join per-model CSVs on "image"
# -------------------------------------------------------------------
def _rebuild_combined_from_per_model_local(
    out_dir: Path,
    selected_models: List[str],
    per_model_csv_names: Dict[str, str],
    combined_csv_name: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / combined_csv_name

    # Read each model CSV into dict image -> (label, score)
    per_model_map: Dict[str, Dict[str, Tuple[str, Optional[float]]]] = {}
    image_set = set()
    for m in selected_models:
        p = out_dir / per_model_csv_names[m]
        rows = _read_local_csv(p)
        mp: Dict[str, Tuple[str, Optional[float]]] = {}
        for r in rows:
            img = (r.get("image") or "").strip()
            if not img:
                continue
            lab = _normalize_label(r.get("classification", ""))
            sc = r.get("score")
            try:
                sc = float(sc) if sc not in (None, "") else None
            except Exception:
                sc = None
            mp[img] = (lab, sc)
            image_set.add(img)
        per_model_map[m] = mp

    rows_combined: List[Dict[str, Any]] = []
    for img in sorted(image_set):
        row: Dict[str, Any] = {"image": img}
        labels_for_consensus: List[str] = []
        for m in selected_models:
            lab, sc = per_model_map.get(m, {}).get(img, ("", None))
            row[f"{m}_label"] = lab
            row[f"{m}_score"] = sc
            if lab:
                labels_for_consensus.append(_normalize_label(lab))
        row["consensus"] = _consensus_from_labels(labels_for_consensus) if labels_for_consensus else ""
        rows_combined.append(row)

    csv_text = _rows_to_combined_csv_general(rows_combined, selected_models)
    combined.write_text(csv_text, encoding="utf-8")
    return combined

# -------------------------------------------------------------------
# Incremental classification core (LOCAL filesystem + HTTP endpoints)
# -------------------------------------------------------------------
def _anchor_for(models: List[str]) -> str:
    """Choose a stable anchor model for idempotency."""
    order = {"maxvit": 0, "coat": 1, "swinv2": 2}
    return sorted(models, key=lambda m: order.get(m, 99))[0]

def _model_csv_name(prefix: str, model: str) -> str:
    if prefix == "laser":
        return {"swinv2": _LASER_SWINV2_CSV(), "coat": _LASER_COAT_CSV(), "maxvit": _LASER_MAXVIT_CSV()}[model]
    else:
        return {"swinv2": _CRIT_SWINV2_CSV(), "coat": _CRIT_COAT_CSV(), "maxvit": _CRIT_MAXVIT_CSV()}[model]

def _combined_csv_name(prefix: str) -> str:
    return _LASER_COMBINED_CSV() if prefix == "laser" else _CRIT_COMBINED_CSV()

def _classify_models_for_folder_local(
    *,
    input_folder: str,
    model_urls: Dict[str, str],
    selected_models: List[str],
    output_csv_folder: str,
    per_model_csv_names: Dict[str, str],
    combined_csv_name: str,
    model_tag_prefix: str,
    max_images: Optional[int] = None
) -> Dict[str, Any]:

    if not input_folder:
        raise EnvironmentError("input_folder is empty.")
    if not output_csv_folder:
        raise EnvironmentError("output_csv_folder is empty.")
    if not selected_models:
        raise ValueError("selected_models is empty.")

    # Validate URLs for selected models (if you're using remote HTTP classifiers)
    missing = [m for m in selected_models if not (model_urls.get(m) or "").strip()]
    if missing:
        raise EnvironmentError(f"Missing URL for selected model(s): {', '.join(missing)}")

    # Stable ordering (also defines anchor priority)
    order = {"maxvit": 0, "coat": 1, "swinv2": 2}
    selected_models = sorted(dict.fromkeys(selected_models), key=lambda m: order.get(m, 99))
    anchor = selected_models[0]

    img_dir = Path(input_folder)
    out_dir = _ensure_dir(output_csv_folder)

    images = _list_images(img_dir, max_images=max_images)
    img_paths = [p.as_posix() for p in images]

    # Determine NEW images using anchor model's CSV
    anchor_csv = out_dir / per_model_csv_names[anchor]
    existing_anchor_rows = _read_local_csv(anchor_csv)
    existing_anchor_imgs = {(r.get("image") or "").strip() for r in existing_anchor_rows}

    to_run_paths = [p for p in img_paths if p and p not in existing_anchor_imgs]
    if not to_run_paths:
        combined = _rebuild_combined_from_per_model_local(out_dir, selected_models, per_model_csv_names, combined_csv_name)
        csv_paths = {m: (out_dir / per_model_csv_names[m]).as_posix() for m in selected_models}
        csv_paths["combined"] = combined.as_posix()
        return {
            "count": 0,
            "seconds": 0.0,
            "selected_models": selected_models,
            "csv_paths": csv_paths,
            "csv_names": {**per_model_csv_names, "combined": combined_csv_name},
            "no_new": True,
            "message": "No new images to classify. Outputs are up-to-date."
        }

    # Accumulate per-model rows for just the new images
    new_rows_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in selected_models}
    failed_by_model: Dict[str, List[str]] = {m: [] for m in selected_models}
    t0 = time.time()

    for path in to_run_paths:
        try:
            data = Path(path).read_bytes()
        except Exception:
            data = None

        for m in selected_models:
            tag = f"{model_tag_prefix}_{m}"
            if data is None:
                resp = {"error": "could not read image bytes"}
            else:
                resp = _post_image((model_urls[m] or "").strip(), (Path(path).name, io.BytesIO(data)))

            lab, sc = _extract_label_and_score(resp)

            # Skip writing rows on error; keep for retry next run
            if not lab:
                failed_by_model[m].append(path)
                continue

            new_rows_by_model[m].append({
                "image": path,                    # store FULL path for local workflows
                "classification": lab,
                "score": sc,
                "model": tag
            })

    # Append de-duped rows to per-model CSVs
    csv_paths: Dict[str, str] = {}
    for m in selected_models:
        per_path = out_dir / per_model_csv_names[m]
        existing = _read_local_csv(per_path)
        merged = _dedup_by_image(existing + new_rows_by_model[m])
        _write_local_csv(per_path, merged)
        csv_paths[m] = per_path.as_posix()

    # Rebuild combined
    combined = _rebuild_combined_from_per_model_local(out_dir, selected_models, per_model_csv_names, combined_csv_name)
    csv_paths["combined"] = combined.as_posix()

    processed_ct = sum(len(v) for v in new_rows_by_model.values())
    failed_ct = sum(len(v) for v in failed_by_model.values())

    return {
        "count": processed_ct,
        "failed": failed_ct,
        "seconds": round(time.time() - t0, 2),
        "selected_models": selected_models,
        "csv_paths": csv_paths,
        "csv_names": {**per_model_csv_names, "combined": combined_csv_name},
        "processed_images_preview_by_model": {m: [r["image"] for r in rows[:50]] for m, rows in new_rows_by_model.items()},
        "failed_images_by_model": {m: v[:50] for m, v in failed_by_model.items()},
    }

# -------------------------------------------------------------------
# Public API used by backend/app.py (Laser / Critical) — LOCAL variant
# -------------------------------------------------------------------
def classify_laser_to_csv(
    *,
    models: List[str],
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    per_model_csv_names: Optional[Dict[str, str]] = None,
    combined_csv_name: str = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    input_folder_id      = input_folder_id      or _laser_dir()
    output_csv_folder_id = output_csv_folder_id or _classify_dir()
    combined_csv_name    = combined_csv_name    or _LASER_COMBINED_CSV()
    names = per_model_csv_names or {
        "swinv2": _LASER_SWINV2_CSV(),
        "coat":   _LASER_COAT_CSV(),
        "maxvit": _LASER_MAXVIT_CSV(),
    }
    urls = _urls_for("laser")
    return _classify_models_for_folder_local(
        input_folder=input_folder_id,
        model_urls=urls,
        selected_models=models,
        output_csv_folder=output_csv_folder_id,
        per_model_csv_names=names,
        combined_csv_name=combined_csv_name,
        model_tag_prefix="laser",
        max_images=max_images,
    )

def classify_critical_to_csv(
    *,
    models: List[str],
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    per_model_csv_names: Optional[Dict[str, str]] = None,
    combined_csv_name: str = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    input_folder_id      = input_folder_id      or _critical_dir()
    output_csv_folder_id = output_csv_folder_id or _classify_dir()
    combined_csv_name    = combined_csv_name    or _CRIT_COMBINED_CSV()
    names = per_model_csv_names or {
        "swinv2": _CRIT_SWINV2_CSV(),
        "coat":   _CRIT_COAT_CSV(),
        "maxvit": _CRIT_MAXVIT_CSV(),
    }
    urls = _urls_for("critical")
    return _classify_models_for_folder_local(
        input_folder=input_folder_id,
        model_urls=urls,
        selected_models=models,
        output_csv_folder=output_csv_folder_id,
        per_model_csv_names=names,
        combined_csv_name=combined_csv_name,
        model_tag_prefix="critical",
        max_images=max_images,
    )

# -------------------------------------------------------------------
# Backward-compatible wrappers (run all three)
# -------------------------------------------------------------------
def classify_laser_all_models_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    swinv2_csv: str = None,
    coat_csv: str   = None,
    maxvit_csv: str = None,
    combined_csv: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_laser_to_csv(
        models=["swinv2", "coat", "maxvit"],
        input_folder_id=input_folder_id or _laser_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": swinv2_csv or _LASER_SWINV2_CSV(),
                             "coat":   coat_csv   or _LASER_COAT_CSV(),
                             "maxvit": maxvit_csv or _LASER_MAXVIT_CSV()},
        combined_csv_name=combined_csv or _LASER_COMBINED_CSV(),
        max_images=max_images,
    )

def classify_critical_all_models_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    swinv2_csv: str = None,
    coat_csv: str   = None,
    maxvit_csv: str = None,
    combined_csv: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_critical_to_csv(
        models=["swinv2", "coat", "maxvit"],
        input_folder_id=input_folder_id or _critical_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": swinv2_csv or _CRIT_SWINV2_CSV(),
                             "coat":   coat_csv   or _CRIT_COAT_CSV(),
                             "maxvit": maxvit_csv or _CRIT_MAXVIT_CSV()},
        combined_csv_name=combined_csv or _CRIT_COMBINED_CSV(),
        max_images=max_images,
    )

# -------------------------------------------------------------------
# Optional single-model convenience wrappers (kept for compatibility)
# -------------------------------------------------------------------
def classify_laser_swinv2_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_laser_to_csv(
        models=["swinv2"],
        input_folder_id=input_folder_id or _laser_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": csv_name or _LASER_SWINV2_CSV(),
                             "coat":   _LASER_COAT_CSV(),
                             "maxvit": _LASER_MAXVIT_CSV()},
        max_images=max_images,
    )

def classify_laser_coat_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_laser_to_csv(
        models=["coat"],
        input_folder_id=input_folder_id or _laser_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": _LASER_SWINV2_CSV(),
                             "coat":   csv_name or _LASER_COAT_CSV(),
                             "maxvit": _LASER_MAXVIT_CSV()},
        max_images=max_images,
    )

def classify_laser_maxvit_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_laser_to_csv(
        models=["maxvit"],
        input_folder_id=input_folder_id or _laser_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": _LASER_SWINV2_CSV(),
                             "coat":   _LASER_COAT_CSV(),
                             "maxvit": csv_name or _LASER_MAXVIT_CSV()},
        max_images=max_images,
    )

def classify_critical_swinv2_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_critical_to_csv(
        models=["swinv2"],
        input_folder_id=input_folder_id or _critical_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": csv_name or _CRIT_SWINV2_CSV(),
                             "coat":   _CRIT_COAT_CSV(),
                             "maxvit": _CRIT_MAXVIT_CSV()},
        max_images=max_images,
    )

def classify_critical_coat_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_critical_to_csv(
        models=["coat"],
        input_folder_id=input_folder_id or _critical_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": _CRIT_SWINV2_CSV(),
                             "coat":   csv_name or _CRIT_COAT_CSV(),
                             "maxvit": _CRIT_MAXVIT_CSV()},
        max_images=max_images,
    )

def classify_critical_maxvit_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    return classify_critical_to_csv(
        models=["maxvit"],
        input_folder_id=input_folder_id or _critical_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        per_model_csv_names={"swinv2": _CRIT_SWINV2_CSV(),
                             "coat":   _CRIT_COAT_CSV(),
                             "maxvit": csv_name or _CRIT_MAXVIT_CSV()},
        max_images=max_images,
    )

# -------------------------------------------------------------------
# Body classification (single-model CoaT) — LOCAL (HTTP endpoint optional)
# -------------------------------------------------------------------
def classify_body_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    endpoint_url: str = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify BODY crops with CoaT (Good/Defective) via HTTP endpoint (if provided) and write a single CSV.
       If you don't use HTTP for body, prefer your local PyTorch runner (backend/inference_jobs/coat_body.py).
    """
    input_folder_id      = input_folder_id      or _body_dir()
    output_csv_folder_id = output_csv_folder_id or _classify_dir()
    csv_name             = csv_name             or _BODY_COAT_CSV_NAME()
    endpoint_url         = (endpoint_url or _body_endpoint() or "").strip()

    if not input_folder_id:
        raise EnvironmentError("input_folder_id is empty.")
    if not output_csv_folder_id:
        raise EnvironmentError("output_csv_folder_id is empty.")
    if not endpoint_url:
        raise EnvironmentError("COAT_BODY_URL is not configured.")

    img_dir = Path(input_folder_id)
    out_dir = _ensure_dir(output_csv_folder_id)
    out_csv = out_dir / csv_name

    images = _list_images(img_dir, max_images=max_images)
    img_paths = [p.as_posix() for p in images]

    # Determine new images by reading existing CSV
    existing = _read_local_csv(out_csv)
    existing_imgs = {(r.get("image") or "").strip() for r in existing}

    to_run_paths = [p for p in img_paths if p not in existing_imgs]
    if not to_run_paths:
        return {
            "count": 0,
            "seconds": 0.0,
            "csv_path": out_csv.as_posix(),
            "csv_name": csv_name,
            "no_new": True,
            "message": "No new body images to classify. Output is up-to-date.",
        }

    t0 = time.time()
    new_rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for path in to_run_paths:
        try:
            data = Path(path).read_bytes()
        except Exception:
            data = None

        if data is None:
            resp = {"error": "could not read image bytes"}
        else:
            resp = _post_image(endpoint_url, (Path(path).name, io.BytesIO(data)))

        lab, sc = _extract_label_and_score(resp)
        if not lab:
            failed.append(path)
            continue

        new_rows.append({
            "image": path,
            "classification": lab,
            "score": sc,
            "model": "body_coat",
        })

    merged = _dedup_by_image(existing + new_rows)
    _write_local_csv(out_csv, merged)

    return {
        "count": len(new_rows),
        "failed": len(failed),
        "seconds": round(time.time() - t0, 2),
        "csv_path": out_csv.as_posix(),
        "csv_name": csv_name,
        "processed_images": [r.get("image") for r in new_rows[:50]],
        "failed_images_preview": failed[:50],
    }

def classify_body_coat_to_csv(
    *,
    input_folder_id: str = None,
    output_csv_folder_id: str = None,
    csv_name: str = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Compatibility alias used by backend/inference_jobs/coat_body.py (if you still call via modal_request)."""
    return classify_body_to_csv(
        input_folder_id=input_folder_id or _body_dir(),
        output_csv_folder_id=output_csv_folder_id or _classify_dir(),
        csv_name=csv_name or _BODY_COAT_CSV_NAME(),
        endpoint_url=_body_endpoint(),
        max_images=max_images,
    )
