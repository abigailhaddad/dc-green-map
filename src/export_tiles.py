"""Export the DC mosaic tiles (polygon + colour + region) as web/dc_tiles.js for the browser
viewer (a .js global avoids file:// fetch/CORS issues). Built at a lighter resolution so the
polygon count stays interactive."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from contourpy import contour_generator

from dc_build import build_region, build_tiles, color_tiles, grout_px, WALL, GROUT

WEB = Path(__file__).resolve().parents[1] / "web"
S = 920
region = build_region(S)
H, W = region.shape
big, nbig = build_tiles(region)
lut, tile_region = color_tiles(big, nbig, region)
gpx, diam = grout_px(big, nbig)
objs = ndimage.find_objects(big)


def rdp(pts, eps=1.2):
    pts = np.asarray(pts, float)
    if len(pts) < 5:
        return pts
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a; L = np.hypot(*ab) + 1e-9
        d = np.abs((pts[i+1:j, 0]-a[0])*ab[1] - (pts[i+1:j, 1]-a[1])*ab[0]) / L
        k = int(d.argmax())
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


tiles = []
for lab in range(1, nbig + 1):
    if tile_region[lab] == 0:
        continue
    sl = objs[lab - 1]
    if sl is None:
        continue
    y0 = max(0, sl[0].start - 1); x0 = max(0, sl[1].start - 1)
    sub = (big[y0:sl[0].stop + 1, x0:sl[1].stop + 1] == lab).astype(float)
    if sub.sum() < 18:
        continue
    lines = contour_generator(z=sub).lines(0.5)
    if not lines:
        continue
    poly = rdp(max(lines, key=len) + [x0, y0], 1.2)
    tiles.append({
        "p": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
        "c": [round(float(v), 3) for v in lut[lab]],
        "r": int(tile_region[lab]),
    })

data = dict(S=S, W=int(W), H=int(H), grout=gpx, wall=[round(float(v), 3) for v in WALL],
            grout_rgb=[round(float(v), 3) for v in GROUT], tiles=tiles)
with open(WEB / "dc_tiles.js", "w") as f:
    f.write("window.DC_TILES=")
    json.dump(data, f)
    f.write(";")
print(f"exported {len(tiles)} tiles, grout {gpx}px -> web/dc_tiles.js")
