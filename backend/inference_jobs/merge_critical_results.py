from __future__ import annotations
import os, csv, tempfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    from backend.paths import CSV_DIR
except Exception:
    CSV_DIR = Path(os.getenv("CSV_DIR", "data/csv")).resolve()
    CSV_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(os.getenv("OUTPUT_CLASSIFY_FOLDER_ID", CSV_DIR.as_posix())).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRIT_COMBINED_CSV_NAME = os.getenv("CRIT_COMBINED_CSV", "critical_all_models.csv").strip()
CRITICAL_COMPARE_NAME  = os.getenv("CRITICAL_COMPARE_NAME", "classify/critical_compare.csv").strip()

CRIT_SWINV2_CSV_NAME = os.getenv("CRIT_SWINV2_CSV_NAME", "critical_swinv2_results.csv").strip()
CRIT_MAXVIT_CSV_NAME = os.getenv("CRIT_MAXVIT_CSV_NAME", "critical_maxvit_results.csv").strip()
CRIT_COAT_CSV_NAME   = os.getenv("CRIT_COAT_CSV_NAME",   "critical_coat_results.csv").strip()

COMPARE_PATH_OUT   = OUT_DIR / CRITICAL_COMPARE_NAME
COMBINED_PATH_CSV  = CSV_DIR / CRIT_COMBINED_CSV_NAME
COMBINED_PATH_OUT  = OUT_DIR / CRIT_COMBINED_CSV_NAME

def _read_csv_first(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f)]

def _read_candidates(name: str) -> List[Dict[str, str]]:
    rows = _read_csv_first(OUT_DIR / name)
    if rows:
        return rows
    return _read_csv_first(CSV_DIR / name)

def _basename(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    return Path(p).name

def _pick_display_image(paths: List[str]) -> str:
    for p in paths:
        if p and ("/regions/critical/" in p or "\\regions\\critical\\" in p):
            return p
    for p in paths:
        if p:
            return p
    return _basename(paths[0] if paths else "")

def _first_nonempty(row: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""

def _norm_label(s: str) -> str:
    s = (s or "").strip().upper()
    if s in ("DEFECT", "BAD"): return "DEFECTIVE"
    if s in ("OK", "PASS"):    return "GOOD"
    return s

def _index_by_basename(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        img = (r.get("image") or r.get("filename") or "").strip()
        key = _basename(img)
        if key:
            out[key] = r
    return out

def _consensus(r: Dict[str, str]) -> str:
    vals = [r.get("MaxViT",""), r.get("CoaT",""), r.get("SwinV2","")]
    vals = [v for v in vals if v]
    if not vals:
        return ""
    if all(v == "DEFECTIVE" for v in vals):
        return "ALL_DEFECTIVE"
    if all(v == "GOOD" for v in vals):
        return "ALL_GOOD"
    return "MIXED"

def _atomic_write_csv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="") as tmp:
        w = csv.DictWriter(tmp, fieldnames=["image", "MaxViT", "CoaT", "SwinV2", "consensus"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "image": r.get("image",""),
                "MaxViT": r.get("MaxViT",""),
                "CoaT": r.get("CoaT",""),
                "SwinV2": r.get("SwinV2",""),
                "consensus": r.get("consensus",""),
            })
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)

def run_merge(selected_models: Optional[List[str]] = None) -> Dict[str, str]:
    models = (selected_models or ["maxvit", "coat", "swinv2"])
    want_max  = "maxvit" in models
    want_coat = "coat"   in models
    want_swin = "swinv2" in models

    sw = _read_candidates(CRIT_SWINV2_CSV_NAME) if want_swin else []
    mx = _read_candidates(CRIT_MAXVIT_CSV_NAME) if want_max else []
    ct = _read_candidates(CRIT_COAT_CSV_NAME)   if want_coat else []

    idx_sw = _index_by_basename(sw)
    idx_mx = _index_by_basename(mx)
    idx_ct = _index_by_basename(ct)

    keys = sorted(set(idx_sw) | set(idx_mx) | set(idx_ct))
    rows: List[Dict[str, str]] = []

    for key in keys:
        r_sw = idx_sw.get(key, {})
        r_mx = idx_mx.get(key, {})
        r_ct = idx_ct.get(key, {})

        disp_image = _pick_display_image([
            r_sw.get("image") or r_sw.get("filename") or "",
            r_mx.get("image") or r_mx.get("filename") or "",
            r_ct.get("image") or r_ct.get("filename") or "",
            key,
        ])

        row = {"image": disp_image}
        if want_max:
            lab = _first_nonempty(r_mx, ["MaxViT","maxvit","maxvit_label","classification","pred","label","result","class"])
            row["MaxViT"] = _norm_label(lab)
        else:
            row["MaxViT"] = ""

        if want_coat:
            lab = _first_nonempty(r_ct, ["CoaT","coat","coat_label","classification","pred","label","result","class"])
            row["CoaT"] = _norm_label(lab)
        else:
            row["CoaT"] = ""

        if want_swin:
            lab = _first_nonempty(r_sw, ["SwinV2","swinv2","swinv2_label","classification","pred","label","result","class"])
            row["SwinV2"] = _norm_label(lab)
        else:
            row["SwinV2"] = ""

        row["consensus"] = _consensus(row)
        rows.append(row)

    _atomic_write_csv(COMPARE_PATH_OUT, rows)
    _atomic_write_csv(COMBINED_PATH_CSV, rows)
    if COMBINED_PATH_OUT.resolve() != COMBINED_PATH_CSV.resolve():
        _atomic_write_csv(COMBINED_PATH_OUT, rows)

    return {
        "csv_path": str(COMPARE_PATH_OUT),
        "combined_csv": str(COMBINED_PATH_CSV),
        "rows": str(len(rows)),
        "models": ",".join(models),
        "source": "per_model",
    }

if __name__ == "__main__":
    print(run_merge())
