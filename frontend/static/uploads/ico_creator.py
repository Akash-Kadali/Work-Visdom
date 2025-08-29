from PIL import Image
from pathlib import Path

src = Path("Logo3.png")
dst = src.with_suffix(".ico")
img = Image.open(src).convert("RGBA")
img.save(dst, sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
print("Wrote", dst)
