# backend/local_store.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import shutil, pandas as pd

IMG_PATS = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")

def list_images(root: Path|str, patterns: Iterable[str]=IMG_PATS) -> List[Path]:
    root = Path(root)
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return sorted(out)

def read_csv(path: Path|str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def write_csv(df: pd.DataFrame, path: Path|str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p

def copy_to(src: Path|str, dest_dir: Path|str) -> Path:
    src = Path(src); dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(src, dest_dir / src.name))

def pending_images_from_csv(
    images_root: Path|str,
    review_csv_path: Path|str,
    image_col: str = "image",
    basename_only: bool = True,
) -> List[Path]:
    imgs = list_images(images_root)
    df = read_csv(review_csv_path)
    if df.empty or image_col not in df.columns:
        return imgs

    if basename_only:
        done = {Path(x).stem for x in df[image_col].astype(str).tolist()}
        return [p for p in imgs if p.stem not in done]
    else:
        done = {Path(x).as_posix() for x in df[image_col].astype(str).tolist()}
        return [p for p in imgs if p.as_posix() not in done]
