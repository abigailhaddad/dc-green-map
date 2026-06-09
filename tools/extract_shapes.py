import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa
"""Vectorize the REAL mosaic: turn every segmented tile into an exact polygon, and save a
shape library we can build new mosaics FROM (instead of inventing shapes from scratch).

Proof of capture: we reconstruct the mosaic purely from the extracted polygons. If that
looks like the photo's tiling, we have the exact shapes."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from contourpy import contour_generator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

import metrics as M

SRC = "data/sfo_greenmap.jpg"
img = np.asarray(Image.open(SRC).convert("RGB"), float)
H, W, _ = img.shape
lum = img @ [0.299, 0.587, 0.114]

# --- robust mosaic segmentation: gradient watershed -------------------------
# grout = gradient ridges (visible even in the faint white field); markers = flat
# tile interiors (any brightness); watershed boundaries land on the grout.
def segment(lum, grad_pct=50, min_marker=16):
    s = ndimage.gaussian_filter(lum, 1.0)
    gx = ndimage.sobel(s, 0); gy = ndimage.sobel(s, 1)
    grad = ndimage.gaussian_filter(np.hypot(gx, gy), 1.0)
    flat = grad < np.percentile(grad, grad_pct)
    markers, n = ndimage.label(flat)
    sizes = ndimage.sum(np.ones_like(markers), markers, range(1, n + 1))
    markers[np.isin(markers, np.nonzero(sizes < min_marker)[0] + 1)] = 0
    markers, n = ndimage.label(markers > 0)
    surf = np.clip(grad / grad.max() * 255, 0, 255).astype(np.uint8)
    labels = ndimage.watershed_ift(surf, markers.astype(np.int32))
    labels[labels < 0] = 0
    return labels, n

labels, nmark = segment(lum)
print(f"markers (tiles): {nmark}")
props = M.region_props(labels, min_area=60)
objs = ndimage.find_objects(labels)          # bbox per label, computed once
print(f"{len(props)} tiles to vectorize")


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def classify(lab):
    sub = labels == lab
    r, g, b = (img[..., k][sub].mean() for k in range(3))
    sat = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1e-6)
    lm = 0.299*r + 0.587*g + 0.114*b
    if g > r*1.08 and g > b*1.05 and sat > 0.12: return "green", (r, g, b)
    if r > 150 and g > 120 and b < 110 and sat > 0.25: return "gold", (r, g, b)
    if lm > 150 and sat < 0.18: return "light", (r, g, b)
    return "gray", (r, g, b)


shapes = []
for p in props:
    lab = p["label"]
    sl = objs[lab - 1]
    if sl is None:
        continue
    pad = 2
    y0 = max(0, sl[0].start - pad); x0 = max(0, sl[1].start - pad)
    sub = (labels[y0:sl[0].stop + pad, x0:sl[1].stop + pad] == lab).astype(float)
    lines = contour_generator(z=sub).lines(0.5)
    if not lines:
        continue
    poly = max(lines, key=shoelace)              # outer boundary, (x,y)=(col,row)
    poly = poly + [x0, y0]                        # back to global coords
    cls, rgb = classify(lab)
    shapes.append(dict(cls=cls, rgb=[c/255 for c in rgb],
                       area=p["area"], n_verts=len(poly),
                       poly=poly.tolist()))

# ---- save library ---------------------------------------------------------
json.dump(shapes, open("output/tile_shapes.json", "w"))
nv = np.array([s["n_verts"] for s in shapes])
print(f"saved {len(shapes)} polygons   verts/tile: median {int(np.median(nv))}, "
      f"p10 {int(np.percentile(nv,10))}, p90 {int(np.percentile(nv,90))}")
from collections import Counter
print("by class:", dict(Counter(s["cls"] for s in shapes)))

# ---- proof: reconstruct mosaic from polygons only -------------------------
CCOL = dict(green=(0.30,0.55,0.20), light=(0.85,0.85,0.88), gray=(0.55,0.55,0.55), gold=(0.85,0.66,0.13))
fig, ax = plt.subplots(1, 2, figsize=(17, 7))
ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("REAL photo"); ax[0].axis("off")
patches = [MplPoly(np.array(s["poly"]), closed=True) for s in shapes]
pc = PatchCollection(patches, facecolors=[s["rgb"] for s in shapes],
                     edgecolors="none")
ax[1].add_collection(pc); ax[1].set_xlim(0, W); ax[1].set_ylim(H, 0); ax[1].set_aspect("equal")
ax[1].set_facecolor("#1a1410")   # grout colour, so light tiles are visible
ax[1].set_title(f"RECONSTRUCTED from {len(shapes)} extracted polygons"); ax[1].axis("off")
fig.tight_layout(); fig.savefig("output/shapes_reconstruction.png", dpi=85)

# ---- catalogue of individual real tile shapes -----------------------------
fig2, axs = plt.subplots(6, 8, figsize=(13, 10))
rng = np.random.default_rng(0)
pick = rng.choice(len(shapes), size=48, replace=False)
for a, i in zip(axs.ravel(), pick):
    s = shapes[i]; pg = np.array(s["poly"])
    c = pg.mean(0); pg = pg - c
    a.fill(pg[:, 0], -pg[:, 1], color=CCOL.get(s["cls"]), edgecolor="black", lw=0.6)
    a.set_aspect("equal"); a.axis("off"); a.set_title(s["cls"], fontsize=7)
fig2.suptitle("Catalogue — exact real tile shapes (centred)")
fig2.tight_layout(); fig2.savefig("output/shapes_catalogue.png", dpi=85)
print("wrote shapes_reconstruction.png, shapes_catalogue.png, tile_shapes.json")
