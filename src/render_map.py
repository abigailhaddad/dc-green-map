"""Render the DC Green Map to a PNG.  Usage:  python src/render_map.py [height_px]"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from dc_build import render

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

S = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
r = render(S)
Image.fromarray((r["out"] * 255).astype(np.uint8)).save(OUT / "dc_greenmap.png")
print(f"height={S}px  canvas={r['out'].shape[1]}x{r['out'].shape[0]}  "
      f"{r['nbig']} tiles  grout {r['gpx']}px  ->  output/dc_greenmap.png")
