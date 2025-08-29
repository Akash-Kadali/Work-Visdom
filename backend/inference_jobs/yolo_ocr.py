# backend/inference_jobs/yolo_ocr.py  (PURE LOCAL VERSION – resilient)
import os, io, csv, json, time, re, math, socket
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.modal_request import send_to_yolo

# ========= Config =========
MAX_IMAGES   = int(os.getenv("MAX_IMAGES", 100000))
MAX_WORKERS  = max(1, int(os.getenv("MAX_WORKERS", 2)))
DRY_RUN      = os.getenv("DRY_RUN", "0") == "1"

# -- NEW: robust retries for endpoint calls
YOLO_RETRIES       = max(1, int(os.getenv("YOLO_RETRIES", 3)))
YOLO_BACKOFF_BASE  = float(os.getenv("YOLO_BACKOFF_BASE", 0.75))  # seconds, exponential
YOLO_TIMEOUT       = float(os.getenv("YOLO_TIMEOUT", 30))         # not used here unless send_to_yolo honors it

YOLO_CONF_MIN = float(os.getenv("YOLO_CONF_MIN", 0.20))
YOLO_TAKE_K   = int(os.getenv("YOLO_TAKE_K", 6))

# IMPORTANT: align to your real classes (17: 0-9 + A-G)
CHARACTER_SET = os.getenv("YOLO_CHARSET", os.getenv("OCR_CHARSET", "0123456789ABCDEFG"))

YOLO_FORCE    = os.getenv("YOLO_FORCE", "0") == "1"   # re-run even if CSV contains rows

# Only process filenames that look like code crops: *_code.(png|jpg)
OCR_ONLY_CODE_SUFFIX = os.getenv("OCR_ONLY_CODE_SUFFIX", "1") == "1"
_CODE_NAME_RE = re.compile(r"_code\.(?:png|jpe?g)$", re.IGNORECASE)

# Optional class label → id mapping (if endpoint returns strings)
try:
    LABEL_MAP = json.loads(os.getenv("YOLO_LABEL_MAP", "{}"))
except Exception:
    LABEL_MAP = {}

# -- CHANGED: allow relative dedup threshold (safer across scales)
# If YOLO_X_DEDUP_REL>0, use that fraction of width; else fallback to px
YOLO_X_DEDUP_PX  = float(os.getenv("YOLO_X_DEDUP_PX", "8"))
YOLO_X_DEDUP_REL = float(os.getenv("YOLO_X_DEDUP_REL", "0.0"))   # e.g., 0.02 → 2% of width

_ALLOWED_EXTS = tuple(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTS", ".jpg,.jpeg,.png,.bmp,.webp").split(",")
    if ext.strip()
)

SAVE_YOLO_CSV = os.getenv("SAVE_YOLO_CSV", "1") == "1"
YOLO_CSV_NAME = os.getenv("YOLO_CSV_NAME", "yolo_results.csv")

CLASS_MAP = {i: c for i, c in enumerate(CHARACTER_SET)}

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ========= Local helpers =========
def _iter_images_recursive(root: Path):
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
    images: List[Dict[str, Any]] = []
    for p in _iter_images_recursive(input_dir):
        rel = p.relative_to(input_dir).as_posix()
        images.append({"name": rel, "path": rel})
    # deterministic order
    images.sort(key=lambda d: d["path"])
    return images

def _load_local_image_bytes(abs_dir: Path, path_rel: str) -> io.BytesIO:
    p = (abs_dir / path_rel).resolve()
    with open(p, "rb") as f:
        return io.BytesIO(f.read())

# ========= OCR helpers =========
def _sanitize_pred(pred: str, k: int = YOLO_TAKE_K) -> str:
    if not isinstance(pred, str):
        pred = str(pred or "")
    pred = pred.strip().upper()
    if CHARACTER_SET:
        pred = "".join(c for c in pred if c in CHARACTER_SET)
    # pad or clip
    pred = (pred[:k]).ljust(k, "?")
    return pred

# -- CHANGED: broader key coverage incl. 'cls'
_CLS_KEYS  = ("class_id", "class", "category_id", "cls", "class_idx")
_CONF_KEYS = ("confidence", "score", "conf", "prob")

def _decode_box_and_class(det: Dict[str, Any]) -> Optional[Tuple[float, int, float, Optional[float]]]:
    # Class id
    cid = None
    for key in _CLS_KEYS:
        if key in det:
            try:
                cid = int(det[key]); break
            except Exception:
                pass
    if cid is None:
        label = det.get("label") or det.get("name")
        if isinstance(label, str):
            if label in LABEL_MAP:
                try: cid = int(LABEL_MAP[label])
                except Exception: cid = None
            elif label in CHARACTER_SET:
                cid = CHARACTER_SET.index(label)
            else:
                try: cid = int(label)
                except Exception: cid = None
    if cid is None:
        return None

    # Confidence
    conf = None
    for key in _CONF_KEYS:
        if key in det:
            try:
                conf = float(det[key]); break
            except Exception:
                pass
    if conf is None:
        conf = 1.0

    # X-center + width if available
    x_center, width = None, None
    box = det.get("box") or det.get("bbox")
    if isinstance(box, dict):
        # accept xyxy or xywh dicts
        if {"x1", "x2"} <= set(box.keys()):
            x_center = (float(box["x1"]) + float(box["x2"])) / 2.0
            width = abs(float(box["x2"]) - float(box["x1"]))
        elif {"x", "w"} <= set(box.keys()):
            x_center = float(box["x"]) + float(box["w"]) / 2.0
            width = float(box["w"])
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        x, y, w, h = [float(v) for v in box[:4]]
        x_center = x + w / 2.0
        width = w

    return (x_center, cid, conf, width) if x_center is not None else None

def decode_detections_to_code(dets: List[Dict[str, Any]], image_width: Optional[float] = None) -> str:
    parsed: List[Tuple[float, int, float, Optional[float]]] = []
    for d in dets or []:
        t = _decode_box_and_class(d)
        if t and t[2] >= YOLO_CONF_MIN:
            parsed.append(t)
    if not parsed:
        log("⚠️ No detections above conf; returning all '?'")
        return "?" * YOLO_TAKE_K

    parsed.sort(key=lambda t: t[0])  # left-to-right

    # dedup neighbor centers
    if image_width and YOLO_X_DEDUP_REL > 0:
        dedup_thresh = YOLO_X_DEDUP_REL * float(image_width)
    else:
        dedup_thresh = YOLO_X_DEDUP_PX

    dedup: List[Tuple[float, int, float, Optional[float]]] = []
    last_x: Optional[float] = None
    for t in parsed:
        if last_x is None or abs(t[0] - last_x) > dedup_thresh:
            dedup.append(t)
            last_x = t[0]

    chars = [CLASS_MAP.get(cid, "?") for _, cid, _, _ in dedup[:YOLO_TAKE_K]]
    return _sanitize_pred("".join(chars), YOLO_TAKE_K)

# ========= CSV helpers =========
def _rows_to_csv_text(rows: List[Dict[str, str]]) -> str:
    headers = ["image", "pred_yolo"]
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in headers})
    return sio.getvalue()

def _load_existing_csv_map(csv_dir: Path, filename: str) -> Dict[str, str]:
    p = (csv_dir / filename)
    if not p.exists():
        return {}
    try:
        rdr = csv.DictReader(io.StringIO(p.read_text(encoding="utf-8")))
        out: Dict[str, str] = {}
        for row in rdr:
            img = row.get("image", "")
            pred = row.get("pred_yolo", "")
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

# ========= Inference =========
def _call_yolo_with_retries(rel: str, image_bytes: io.BytesIO) -> Dict[str, Any]:
    last_err = None
    for i in range(YOLO_RETRIES):
        try:
            image_bytes.seek(0)
            return send_to_yolo((rel, image_bytes))
        except Exception as e:
            last_err = e
            if i + 1 < YOLO_RETRIES:
                sleep_s = YOLO_BACKOFF_BASE * (2 ** i)
                log(f"↻ YOLO retry {i+1}/{YOLO_RETRIES} for {rel} after error: {e} (sleep {sleep_s:.2f}s)")
                time.sleep(sleep_s)
    raise RuntimeError(f"YOLO failed after {YOLO_RETRIES} attempts: {last_err}")

def process_single_image(abs_input_dir: Path, img: Dict[str, Any]) -> Dict[str, Any]:
    rel = img["path"]
    try:
        t_dl0 = time.time()
        image_bytes = _load_local_image_bytes(abs_input_dir, rel)
        t_dl = time.time() - t_dl0

        if DRY_RUN:
            predicted_code = "?" * YOLO_TAKE_K
            log(f"🧪 DRY_RUN → skipping Modal call for {rel}")
            return {"image": rel, "pred_yolo": predicted_code, "ok": True, "error": None}

        t_inf0 = time.time()
        result = _call_yolo_with_retries(rel, image_bytes)
        t_inf = time.time() - t_inf0

        if not isinstance(result, dict) or "error" in result:
            raise ValueError(f"Modal error: {result!r}")

        # Try to recover image width if provided
        # Expecting optional `meta: {"width": W, "height": H}` in result; safe default None
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        img_w = None
        if "width" in meta:
            try: img_w = float(meta["width"])
            except Exception: img_w = None

        detections = result.get("detections", [])
        predicted_code = decode_detections_to_code(detections, image_width=img_w)

        log(f"🔍 {rel} → YOLO: {predicted_code} (dl {t_dl:.2f}s, inf {t_inf:.2f}s)")
        return {"image": rel, "pred_yolo": predicted_code, "ok": True, "error": None}

    except Exception as e:
        log(f"❌ Error with {rel}: {e}")
        return {"image": rel, "pred_yolo": "", "ok": False, "error": str(e)}

def run_inference(*_args, input_folder: Optional[str] = None, output_folder: Optional[str] = None, **_kwargs) -> List[Dict[str, str]]:
    log("🚀 Starting YOLO OCR inference job (LOCAL)…")

    # Resolve input dir
    in_folder = input_folder or os.getenv("OUTPUT_CODE_DIR") or os.getenv("OUTPUT_CODE_FOLDER_ID") or os.getenv("INPUT_FOLDER_ID")
    if not in_folder:
        raise EnvironmentError("❌ Set OUTPUT_CODE_DIR (or OUTPUT_CODE_FOLDER_ID/INPUT_FOLDER_ID) to a local images folder.")
    abs_input_dir = Path(in_folder).resolve()
    if not abs_input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {abs_input_dir}")

    # Resolve output dir
    out_dir_env = output_folder or os.getenv("OCR_CSV_DIR") or os.getenv("OUTPUT_OCR_FOLDER_ID", "").strip()
    if not out_dir_env:
        raise EnvironmentError("❌ Set OCR_CSV_DIR (or OUTPUT_OCR_FOLDER_ID / pass output_folder=) to a local folder for CSV output.")
    abs_csv_dir = Path(out_dir_env).resolve()
    abs_csv_dir.mkdir(parents=True, exist_ok=True)

    images = list_images_local(abs_input_dir)
    if not images:
        log("❌ No valid images found.")
        return []

    existing_map: Dict[str, str] = {}
    if SAVE_YOLO_CSV and not YOLO_FORCE:
        existing_map = _load_existing_csv_map(abs_csv_dir, YOLO_CSV_NAME)
        if existing_map:
            log(f"⏭️ Will skip {len(existing_map)} image(s) already present in {YOLO_CSV_NAME}")

    # Only process missing
    todo = [img for img in images if img.get("path") not in existing_map]
    total = min(MAX_IMAGES, len(todo))

    t0 = time.time()
    new_rows_ok: List[Dict[str, str]] = []
    failed: List[str] = []

    if total > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_image, abs_input_dir, img): img for img in todo[:total]}
            done = 0
            for future in as_completed(futures):
                res = future.result()
                done += 1
                if res.get("ok"):
                    new_rows_ok.append({"image": res["image"], "pred_yolo": res["pred_yolo"]})
                    log(f"[{done}/{total}] ✅ Processed {res['image']}")
                else:
                    failed.append(res.get("image", ""))
                    log(f"[{done}/{total}] ⚠️ Skipped {res.get('image','(unknown)')} due to error")
    else:
        log("✅ Nothing new to process for YOLO.")

    merged_map = dict(existing_map)
    for r in new_rows_ok:
        merged_map[r["image"]] = r.get("pred_yolo", "")

    # deterministic union
    union_rows = [{"image": k, "pred_yolo": merged_map[k]} for k in sorted(merged_map.keys())]

    if SAVE_YOLO_CSV:
        _write_csv(abs_csv_dir, YOLO_CSV_NAME, union_rows)

    took = time.time() - t0
    if failed:
        log(f"✅ YOLO inference complete. {len(new_rows_ok)} new, {len(union_rows)} total, {len(failed)} failed in {took:.2f}s.")
    else:
        log(f"✅ YOLO inference complete. {len(new_rows_ok)} new, {len(union_rows)} total in {took:.2f}s.")
    return union_rows

if __name__ == "__main__":
    run_inference()
