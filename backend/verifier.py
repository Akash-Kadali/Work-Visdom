# backend/verifier.py
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# Local path resolvers & helpers
from backend.paths import folder, IMAGES_DIR
from backend.local_store import read_csv, write_csv

# =========================
# Logging
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =========================
# Config / Conventions (env-driven; mapped to local paths)
# =========================
FINAL_CSV_NAME = os.getenv("FINAL_CSV_NAME", "ocr_results.csv").strip()
CRNN_CSV_NAME  = os.getenv("CRNN_CSV_NAME", "crnn_results.csv").strip()
YOLO_CSV_NAME  = os.getenv("YOLO_CSV_NAME", "yolo_results.csv").strip()

# Folders
OCR_DIR         = folder("OUTPUT_OCR_FOLDER_ID")      # CSVs
CODE_IMAGES_DIR = folder("OUTPUT_CODE_FOLDER_ID")     # raw code-crop images
# Where reviewed images should be moved. If not configured, default to CODE_IMAGES_DIR/agree
AGREE_DIR       = folder("OUTPUT_AGREE_FOLDER_ID") or (CODE_IMAGES_DIR / "agree" if CODE_IMAGES_DIR else None)

# OCR code constraints
CODE_LEN     = 6
CODE_CHARSET = set("0123456789ABCDEFG")


# =========================
# Local FS helpers
# =========================
def _csv_path(csv_dir: Path, name: str) -> Path:
    return csv_dir / name

def _download_csv_by_name_local(csv_dir: Path, filename: str) -> pd.DataFrame:
    if not filename:
        return pd.DataFrame()
    path = _csv_path(csv_dir, filename)
    if not path.exists():
        log(f"⚠️ CSV not found: {path}")
        return pd.DataFrame()
    return read_csv(path)

def _get_latest_with_prefix_local(csv_dir: Path, prefix: str) -> pd.DataFrame:
    if not csv_dir.exists():
        log(f"❌ CSV directory not found: {csv_dir}")
        return pd.DataFrame()
    cands = sorted(
        (p for p in csv_dir.glob(f"{prefix}*.csv") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        log(f"❌ No CSV files with prefix '{prefix}' in {csv_dir}")
        return pd.DataFrame()
    log(f"📄 Using latest fallback: {cands[0].name}")
    return read_csv(cands[0])

def _upload_csv_df_local(csv_dir: Path, filename: str, df: pd.DataFrame) -> Path:
    return write_csv(df, _csv_path(csv_dir, filename))

def _list_paths_by_name(dir_path: Path, filename: str) -> List[Path]:
    """
    Return likely matches for filename inside dir_path.
    Accepts basenames or full paths. Falls back to rglob(basename).
    """
    out: List[Path] = []
    name = str(filename or "").strip()
    if not name:
        return out

    p = Path(name)
    if p.exists():
        return [p]

    if dir_path:
        cand = dir_path / name
        if cand.exists():
            out.append(cand)

        base = p.name
        if base and base != name:
            cand2 = dir_path / base
            if cand2.exists():
                out.append(cand2)

        if not out and base:
            out.extend(dir_path.rglob(base))

    # de-dup
    seen = set()
    uniq: List[Path] = []
    for ap in out:
        s = ap.resolve().as_posix()
        if s not in seen:
            seen.add(s)
            uniq.append(ap)
    return uniq

def _ensure_dir(d: Optional[Path]) -> Optional[Path]:
    if d is None:
        return None
    d.mkdir(parents=True, exist_ok=True)
    return d

def _unique_dst(dst: Path) -> Path:
    """Avoid collisions when moving files by adding a numeric suffix if needed."""
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


# =========================
# Column normalization helpers
# =========================
_PRED_COL_CANDIDATES_YOLO = ("pred_yolo", "yolo", "yolo_code", "yolo_text", "prediction_yolo")
_PRED_COL_CANDIDATES_CRNN = ("pred_crnn", "crnn", "crnn_code", "crnn_text", "prediction_crnn")
_IMAGE_COL_CANDIDATES     = ("image", "img", "filename", "file", "path")

def _normalize_df_schema(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Ensure df contains 'image' and 'pred_<model>' columns.
    model in {"yolo","crnn"}.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["image", f"pred_{model}"])
    df = df.copy()

    # image column
    if "image" not in df.columns:
        for c in _IMAGE_COL_CANDIDATES:
            if c in df.columns:
                df = df.rename(columns={c: "image"})
                break
    if "image" not in df.columns:
        df["image"] = ""

    # prediction column
    want_col = f"pred_{model}"
    if want_col not in df.columns:
        candidates = _PRED_COL_CANDIDATES_YOLO if model == "yolo" else _PRED_COL_CANDIDATES_CRNN
        for c in candidates:
            if c in df.columns:
                df = df.rename(columns={c: want_col})
                break
    if want_col not in df.columns:
        df[want_col] = ""
    return df[["image", want_col]]

def _norm_series(s: pd.Series) -> pd.Series:
    """Uppercase + strip all whitespace, vectorized."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.upper()

def _derive_agree(df: pd.DataFrame) -> pd.Series:
    """Agree is TRUE only when normalized predictions match exactly."""
    py = _norm_series(df.get("pred_yolo", pd.Series([], dtype=str)))
    pc = _norm_series(df.get("pred_crnn", pd.Series([], dtype=str)))
    return (py == pc)


# =========================
# Public: build verification items (for UI)
# =========================
def get_verification_items(only_disagreements: bool = True) -> List[Dict]:
    """
    Returns list of dicts:
      { "image", "pred_yolo", "pred_crnn", "agree": "YES"|"NO", "img_path", "proxy_url" }

    Behavior:
      - When only_disagreements=True  → show only rows where models disagree AND not yet reviewed (actual_code == "").
      - When only_disagreements=False → show ALL rows (reviewed + unreviewed, agree + disagree).
    """
    if not FINAL_CSV_NAME:
        log("❌ FINAL_CSV_NAME is not configured.")
        return []
    if not OCR_DIR:
        log("❌ OCR_DIR is not configured.")
        return []

    # Try consolidated CSV first
    df = _download_csv_by_name_local(OCR_DIR, FINAL_CSV_NAME)

    # Fallback: merge per-model CSVs
    if df is None or df.empty or "image" not in df.columns:
        crnn_df = _download_csv_by_name_local(OCR_DIR, CRNN_CSV_NAME)
        if crnn_df.empty:
            crnn_df = _get_latest_with_prefix_local(OCR_DIR, "crnn_results")
        yolo_df = _download_csv_by_name_local(OCR_DIR, YOLO_CSV_NAME)
        if yolo_df.empty:
            yolo_df = _get_latest_with_prefix_local(OCR_DIR, "yolo_results")

        if crnn_df.empty or yolo_df.empty:
            log("❌ Could not build verification items (missing CSVs).")
            return []

        crnn_df = _normalize_df_schema(crnn_df, "crnn")
        yolo_df = _normalize_df_schema(yolo_df, "yolo")
        df = pd.merge(crnn_df, yolo_df, on="image", how="outer")

    # Ensure columns
    if "pred_yolo" not in df.columns or "pred_crnn" not in df.columns:
        df = _normalize_df_schema(df, "yolo").merge(_normalize_df_schema(df, "crnn"), on="image", how="outer")
    for col in ("pred_yolo", "pred_crnn"):
        if col not in df.columns:
            df[col] = ""

    # Derive agreement from predictions only
    df["agree"] = _derive_agree(df)

    # Ensure actual_code exists
    if "actual_code" not in df.columns:
        df["actual_code"] = ""

    # Filtering policy
    if only_disagreements:
        # Show only NOT-reviewed mismatches
        df = df[(df["actual_code"].astype(str).str.len() == 0) & (~df["agree"])]

    items: List[Dict] = []
    for _, r in df.iterrows():
        img_name = str(r.get("image") or "").strip()
        py = str(r.get("pred_yolo") or "").strip()
        pc = str(r.get("pred_crnn") or "").strip()
        agree_str = "YES" if bool(r.get("agree", False)) else "NO"

        img_path_rel = ""
        proxy_url = ""

        # Look under CODE_IMAGES_DIR first; if moved, also try AGREE_DIR
        paths = _list_paths_by_name(CODE_IMAGES_DIR, img_name) if CODE_IMAGES_DIR else []
        if not paths and AGREE_DIR:
            paths = _list_paths_by_name(AGREE_DIR, img_name)

        if paths:
            p = paths[0]
            try:
                img_path_rel = p.relative_to(IMAGES_DIR).as_posix()
            except Exception:
                img_path_rel = p.name

        items.append(
            {
                "image": img_name,
                "pred_yolo": py,
                "pred_crnn": pc,
                "agree": agree_str,
                "img_path": img_path_rel,
                "proxy_url": proxy_url,
            }
        )

    log(f"🔎 Verification items: {len(items)} (only_disagreements={only_disagreements})")
    return items

# =========================
# Public: optional image copy for UI (static)
# =========================
def download_images_to_static(
    image_names: List[str],
    image_dir: Optional[Path] = None,
    static_rel: str = os.path.join("frontend", "static", "uploads"),
) -> None:
    src_dir = image_dir or CODE_IMAGES_DIR
    out_dir = Path(static_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in image_names:
        hits = _list_paths_by_name(src_dir, name)
        if not hits:
            log(f"❌ Image not found locally: {src_dir / name}")
            continue
        src = hits[0]
        dst = out_dir / Path(name).name
        shutil.copy2(src, dst)
        log(f"📥 Copied image → {dst}")


# =========================
# Code normalization / validation
# =========================
def _normalize_code(code: str) -> str:
    if code is None:
        return ""
    return (code or "").strip().upper()

def _is_valid_code(code: str) -> bool:
    if len(code) != CODE_LEN:
        return False
    return all(c in CODE_CHARSET for c in code)


# =========================
# Move reviewed images to agree/{YES|NO}/
# =========================
def _move_image_to_agree(img_name: str, agree_bool: bool) -> Optional[Path]:
    """
    Move the image into AGREE_DIR/YES or AGREE_DIR/NO.
    Returns the destination path (or None on failure).
    """
    if not CODE_IMAGES_DIR:
        return None
    if not AGREE_DIR:
        return None

    hits = _list_paths_by_name(CODE_IMAGES_DIR, img_name)
    if not hits:
        log(f"⚠️ Image not found to move: {img_name}")
        return None

    src = hits[0]
    label = "YES" if agree_bool else "NO"
    dst_dir = _ensure_dir(Path(AGREE_DIR) / label)
    if not dst_dir:
        return None

    dst = _unique_dst(dst_dir / src.name)
    try:
        shutil.move(str(src), str(dst))
        log(f"📦 Moved reviewed image → {dst}")
        return dst
    except Exception as e:
        log(f"❌ Failed to move image {src} → {dst}: {e}")
        return None


# =========================
# Public: apply user corrections (LOCAL)
# =========================
def apply_user_corrections(corrections: Dict[str, str]) -> Dict[str, int]:
    """
    corrections: {"IMG123_code.png": "ABC12G", ...}
    - Loads FINAL_CSV_NAME from OCR_DIR
    - Writes/updates 'actual_code' (manual entry)
    - DOES NOT modify 'agree' (model agreement only)
    - Moves the image to AGREE_DIR/YES or AGREE_DIR/NO based on current predictions
    - Writes 'verified_codes.csv' (image, actual_code)
    - Subsequent /verify_data calls will not include reviewed rows (actual_code != '')
    """
    if not OCR_DIR or not FINAL_CSV_NAME:
        raise RuntimeError("OCR_DIR or FINAL_CSV_NAME not configured.")

    df = _download_csv_by_name_local(OCR_DIR, FINAL_CSV_NAME)
    if df is None or df.empty:
        raise RuntimeError(f"Cannot load '{FINAL_CSV_NAME}' from {OCR_DIR}.")

    df = df.copy()

    # Schema
    if "image" not in df.columns:
        for c in _IMAGE_COL_CANDIDATES:
            if c in df.columns:
                df = df.rename(columns={c: "image"})
                break
        if "image" not in df.columns:
            raise RuntimeError("CSV missing 'image' column.")

    for col in ("pred_yolo", "pred_crnn"):
        if col not in df.columns:
            df[col] = ""

    if "actual_code" not in df.columns:
        df["actual_code"] = ""

    # Recompute agreement from predictions
    df["agree"] = _derive_agree(df)

    total = 0
    applied = 0
    invalid = 0
    moved = 0

    # Indexes
    index_full = {str(img).strip(): i for i, img in enumerate(df["image"].astype(str))}
    index_base = {Path(img).name: i for i, img in enumerate(df["image"].astype(str))}

    for img_name, code in corrections.items():
        total += 1
        norm = _normalize_code(code)
        if not norm:
            continue  # ignore empty

        row_idx = index_full.get(img_name, index_base.get(Path(img_name).name))
        if row_idx is None:
            log(f"⚠️ Image not found in CSV: {img_name}")
            continue

        # Write what the user typed (even if invalid), but track validity for stats
        if not _is_valid_code(norm):
            invalid += 1

        df.at[row_idx, "actual_code"] = norm
        applied += 1

        # Move image to agree/{YES|NO} based on current predictions (status NO for mismatches)
        agree_bool = bool(df.at[row_idx, "agree"])
        if _move_image_to_agree(img_name, agree_bool):
            moved += 1

    # Save updated master CSV
    _upload_csv_df_local(OCR_DIR, FINAL_CSV_NAME, df)
    log(f"📤 Updated {FINAL_CSV_NAME}: applied={applied}, invalid={invalid}, moved={moved}, total_received={total}")

    # Slim verified file
    verified = df.loc[df["actual_code"].astype(str).str.len() > 0, ["image", "actual_code"]].copy()
    _upload_csv_df_local(OCR_DIR, "verified_codes.csv", verified)
    log(f"📤 Wrote verified_codes.csv (rows={len(verified)})")

    return {"received": total, "applied": applied, "invalid": invalid, "moved": moved, "verified_rows": int(len(verified))}


# =========================
# Back-compat helpers
# =========================
def get_mismatches_from_drive() -> List[Dict]:
    items = get_verification_items(only_disagreements=True)
    out = []
    for it in items:
        out.append(
            {
                "image": it.get("image", ""),
                "predicted_code_crnn": it.get("pred_crnn", ""),
                "predicted_code_yolo": it.get("pred_yolo", ""),
            }
        )
    return out

def upload_verified_csv_to_drive(df: pd.DataFrame) -> None:
    if not OCR_DIR:
        log("❌ OCR_DIR is not set.")
        return
    _upload_csv_df_local(OCR_DIR, "verified_codes.csv", df)
    log("📤 Uploaded → verified_codes.csv (local)")
