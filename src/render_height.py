"""Render a HEIGHT MAP for making a raised/tactile version of the map (UV-flatbed texture
layer, 3D-relief displacement, CNC, etc.):  white = raised tile top, black = recessed grout.
Each tile domes slightly and sits at a slightly different height (hand-laid feel); the Capitol
disc stands proud.  Pair it with output/dc_greenmap.png (the colour) for a textured print.

Usage:  python src/render_height.py [height_px]   (default 2000)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from dc_build import render

OUT = Path(__file__).resolve().parents[1] / "output"
S = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
r = render(S)
big, region, gpx = r["big"], r["region"], r["gpx"]
H, W = big.shape

# tile boundaries -> grout; distance-to-edge -> per-tile dome
bound = np.zeros((H, W), bool)
bound[:, :-1] |= big[:, :-1] != big[:, 1:]
bound[:-1, :] |= big[:-1, :] != big[1:, :]
grout = ndimage.binary_dilation(bound, iterations=max(1, gpx // 2))
ed = ndimage.distance_transform_edt(~bound)
dome = np.clip(ed / 4.0, 0, 1)                     # rounded top within ~4px of each edge

# each tile a slightly different base height (hand-laid variation)
rng = np.random.default_rng(0)
base_lut = 0.55 + 0.25 * rng.random(int(big.max()) + 1)
height = base_lut[big] + 0.20 * dome

height[grout] = 0.0                                # recessed grout
height[region == 0] = 0.0                          # outside the District
cap = region == 4                                  # Capitol disc stands proud, smooth dome
if cap.any():
    e4 = ndimage.distance_transform_edt(cap)
    height[cap] = 0.85 + 0.15 * e4[cap] / (e4.max() or 1.0)

height = np.clip(height, 0, 1)
Image.fromarray((height * 65535).astype(np.uint16), mode="I;16").save(OUT / "dc_greenmap_height.png")
# save the COLOUR at the same size so the pair is guaranteed pixel-aligned (same tiles)
Image.fromarray((r["out"] * 255).astype(np.uint8)).save(OUT / "dc_greenmap_print.png")
print(f"{W}x{H} aligned pair for a textured / raised print:")
print("  output/dc_greenmap_print.png   colour")
print("  output/dc_greenmap_height.png  height (16-bit: white=raised tile, black=grout)")
