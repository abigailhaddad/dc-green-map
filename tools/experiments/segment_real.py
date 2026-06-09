"""Segment the real SFO 'Green Map' close-up into individual tiles, classify each by
colour (green park shard vs light tessera vs gold/gray), and measure them.

Grout detection: black top-hat on luminance. The grout is a thin network that is darker
than its *local* surroundings; a top-hat with a structuring element larger than the grout
width but smaller than a tile isolates exactly that thin dark network, independent of
whether the tiles themselves are bright (white) or dark (green)."""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gm3c.png"
TOPHAT_SE = 15      # > grout width, < tile size
TOPHAT_C  = 10      # luminance margin to call a pixel 'grout'
MIN_AREA  = 120

img = np.asarray(Image.open(SRC).convert("RGB"), float)
H, W, _ = img.shape
lum = img @ [0.299, 0.587, 0.114]

# black top-hat = closing - image  -> bright where there are thin dark lines (grout)
closed = ndimage.grey_closing(lum, size=TOPHAT_SE)
tophat = closed - lum
grout = tophat > TOPHAT_C
grout = ndimage.binary_closing(grout, iterations=1)        # connect the network
tiles_mask = ~grout
tiles_mask = ndimage.binary_opening(tiles_mask, iterations=1)  # drop specks

labels, n = ndimage.label(tiles_mask)
print(f"raw components: {n}")
props = M.region_props(labels, min_area=MIN_AREA)
print(f"tiles after area filter: {len(props)}")

# ---- classify each tile by mean colour -----------------------------------
def classify(p):
    lab = p["label"]
    sub = labels == lab
    r, g, b = (img[..., k][sub].mean() for k in range(3))
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / (mx + 1e-6)
    if g > r * 1.08 and g > b * 1.05 and sat > 0.12:
        return "green"
    if r > 150 and g > 120 and b < 110 and sat > 0.25:
        return "gold"
    if lum_mean := (0.299*r + 0.587*g + 0.114*b):
        if lum_mean > 150 and sat < 0.18:
            return "light"
    return "gray"

for p in props:
    p["cls"] = classify(p)

by = {}
for p in props:
    by.setdefault(p["cls"], []).append(p)

# ---- grout width estimate -------------------------------------------------
edt = ndimage.distance_transform_edt(grout)
ridge = edt[edt > 0]
grout_w = float(2 * np.median(ridge)) if ridge.size else 0.0

# ---- report ---------------------------------------------------------------
out = dict(source=SRC, image=[W, H], grout_width_px=grout_w,
           overall=M.summarize(props),
           by_class={k: M.summarize(v) for k, v in by.items()},
           class_counts={k: len(v) for k, v in by.items()})
with open("analysis/target_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\ngrout width ~ {grout_w:.1f}px")
print(M.fmt(out["overall"], "ALL TILES"))
for k in ("green", "light", "gray", "gold"):
    if k in by:
        print(M.fmt(out["by_class"][k], f"{k.upper()} ({len(by[k])})"))

# ---- visualization --------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("real"); ax[0].axis("off")
# segmentation overlay coloured by class
CLS_COL = dict(green=(0.1,0.8,0.2), light=(0.9,0.9,0.95), gray=(0.6,0.6,0.6), gold=(0.95,0.75,0.1))
overlay = np.zeros((H, W, 3))
for p in props:
    overlay[labels == p["label"]] = CLS_COL.get(p["cls"], (1,0,0))
overlay[grout] = (0.08, 0.05, 0.03)
ax[1].imshow(overlay); ax[1].set_title(f"{len(props)} tiles segmented"); ax[1].axis("off")
# area distribution per class (log)
for k, v in by.items():
    a = np.log10([p["area"] for p in v])
    ax[2].hist(a, bins=30, alpha=0.5, label=f"{k} (n={len(v)})", color=CLS_COL.get(k,(1,0,0)))
ax[2].set_xlabel("log10 tile area (px)"); ax[2].set_title("size variation by region"); ax[2].legend()
fig.tight_layout(); fig.savefig("analysis/real_segmentation.png", dpi=90)
print("wrote analysis/target_metrics.json and analysis/real_segmentation.png")
