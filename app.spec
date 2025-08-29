# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

# Use CWD because PyInstaller doesn't define __file__ in .spec
PROJECT_ROOT = Path(os.getcwd())

# ---------- helpers ----------
def walk_datas(root: Path, dest_prefix: str,
               ignore_exts=(".pyc",),
               ignore_names={"__pycache__", ".DS_Store", ".ipynb_checkpoints"}):
    files = []
    if root.exists():
        for p in root.rglob("*"):
            name = p.name
            if name in ignore_names:
                continue
            if any(name.endswith(ext) for ext in ignore_exts):
                continue
            if p.is_file():
                dest = str(Path(dest_prefix) / p.relative_to(root).parent)
                files.append((str(p), dest))
    return files


def ensure_ico(png_path: Path) -> Path:
    """
    Convert PNG -> ICO at build time if needed.
    Requires Pillow installed in the build environment.
    """
    ico_path = png_path.with_suffix(".ico")
    if ico_path.exists():
        return ico_path
    try:
        from PIL import Image  # Pillow must be available in build env
        img = Image.open(png_path).convert("RGBA")
        img.save(ico_path, sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
        print(f"[spec] Converted {png_path} -> {ico_path}")
        return ico_path
    except Exception as e:
        raise SystemExit(
            f"[spec] Missing icon and cannot convert {png_path} -> .ico. "
            f"Install Pillow or pre-create the .ico. Error: {e}"
        )


def find_icon() -> Path | None:
    """Try a few common icon names in frontend/static/uploads."""
    cand_dir = PROJECT_ROOT / "frontend" / "static" / "uploads"
    candidates = ["Logo3.png", "Logo4.png", "icon.png", "icon.ico"]
    for name in candidates:
        p = cand_dir / name
        if p.exists():
            return p if p.suffix.lower() == ".ico" else ensure_ico(p)
    return None


# ---------- data files ----------
datas = []
# Frontend assets
datas += walk_datas(PROJECT_ROOT / "frontend" / "static",    "frontend/static")
datas += walk_datas(PROJECT_ROOT / "frontend" / "templates", "frontend/templates")

# Include full data tree (recursively)
datas += walk_datas(PROJECT_ROOT / "data", "data")

# Optional extras if present (harmless if missing)
datas += walk_datas(PROJECT_ROOT / "models", "models")
datas += walk_datas(PROJECT_ROOT / "backend" / "assets", "backend/assets")

# Loose config files
if (PROJECT_ROOT / ".env").exists():
    datas.append((str(PROJECT_ROOT / ".env"), "."))
if (PROJECT_ROOT / "routes.yaml").exists():
    datas.append((str(PROJECT_ROOT / "routes.yaml"), "."))

# Optional Google OAuth files if present
for p in list(PROJECT_ROOT.glob("client_secret*.json")) + list(PROJECT_ROOT.glob("token*.json")):
    datas.append((str(p), "."))

# ---------- hidden imports (safe/optional) ----------
hiddenimports = []
try:
    hiddenimports += collect_submodules("backend")
except Exception:
    pass  # ok if no 'backend' package

def opt(modname):
    try:
        __import__(modname)
        return [modname]
    except Exception:
        return []

hiddenimports += opt("googleapiclient.http")
hiddenimports += opt("googleapiclient.discovery")
hiddenimports += opt("dotenv")

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Ensure icon exists (convert your PNG if necessary)
icon_file = find_icon()

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],                     # keep this positional arg for compatibility
    name="Visdom",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # ok if UPX not installed; PyInstaller will skip
    console=False,          # set True if you want a console for logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=(str(icon_file) if icon_file else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Visdom",
)
