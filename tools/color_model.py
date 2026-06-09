import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa
"""Characterize the REAL tile colours per region and build a sampler from them.
Key fact: the 'look' comes from within-region tonal VARIATION (silver = warm+cool light
greys with gold flecks; parks = olive→emerald spread), not single mean colours.
Proof: recolour the extracted shapes by SAMPLING the per-class colour distribution; if it
matches the real-coloured reconstruction, the colour model is good."""
import json
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

shapes = json.load(open("output/tile_shapes.json"))
img = np.asarray(Image.open("data/sfo_greenmap.jpg").convert("RGB"))
H, W = img.shape[:2]
CLASSES = ["light", "green", "gray", "gold"]
cols = {c: np.array([s["rgb"] for s in shapes if s["cls"] == c]) for c in CLASSES}

# ---- characterize ---------------------------------------------------------
print("per-region colour distribution (RGB 0-1, HSV):")
for c in CLASSES:
    a = cols[c]
    if len(a) == 0:
        continue
    hsv = rgb_to_hsv(a)
    print(f"  {c:6s} n={len(a):4d}  RGB med=({a[:,0].mean():.2f},{a[:,1].mean():.2f},{a[:,2].mean():.2f})  "
          f"val {hsv[:,2].mean():.2f}±{hsv[:,2].std():.2f}  "
          f"sat {hsv[:,1].mean():.2f}±{hsv[:,1].std():.2f}  "
          f"hue±{hsv[:,0].std():.3f}")

# empirical sampler = bootstrap from the real per-class colours (optionally tiny jitter)
def sample_color(cls, rng, jitter=0.02):
    a = cols[cls] if len(cols.get(cls, [])) else cols["light"]
    c = a[rng.integers(len(a))] + rng.normal(0, jitter, 3)
    return np.clip(c, 0, 1)

json.dump({c: cols[c].tolist() for c in CLASSES if len(cols[c])},
          open("data/color_model.json", "w"))

# ---- figure ---------------------------------------------------------------
rng = np.random.default_rng(0)
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img); ax0.set_title("REAL photo"); ax0.axis("off")

# swatch strips: each class sorted by luminance -> shows the tonal RANGE
ax1 = fig.add_subplot(gs[0, 1])
for r, c in enumerate(CLASSES):
    a = cols[c]
    if len(a) == 0:
        continue
    lum = a @ [0.299, 0.587, 0.114]
    a = a[np.argsort(lum)]
    idx = np.linspace(0, len(a) - 1, 60).astype(int)
    for k, i in enumerate(idx):
        ax1.add_patch(plt.Rectangle((k, r), 1, 0.9, color=a[i]))
    ax1.text(-9, r + 0.45, c, va="center", fontsize=11)
ax1.set_xlim(-10, 60); ax1.set_ylim(-0.2, len(CLASSES)); ax1.invert_yaxis()
ax1.set_title("real tonal range per region (sorted by luminance)"); ax1.axis("off")

# reconstructions: real colours vs colours SAMPLED from the model
patches = [MplPoly(np.array(s["poly"]), closed=True) for s in shapes]
for ax, mode, title in [(fig.add_subplot(gs[1, 0]), "real", "reconstruction — REAL colours"),
                        (fig.add_subplot(gs[1, 1]), "samp", "reconstruction — colours SAMPLED from model")]:
    if mode == "real":
        fc = [s["rgb"] for s in shapes]
    else:
        fc = [sample_color(s["cls"], rng) for s in shapes]
    pc = PatchCollection(patches, facecolors=fc, edgecolors="none")
    ax.add_collection(pc); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_facecolor("#1a1410"); ax.set_title(title); ax.axis("off")

fig.tight_layout(); fig.savefig("output/color_model.png", dpi=85)
print("wrote output/color_model.png and color_model.json")
