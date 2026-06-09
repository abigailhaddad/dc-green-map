import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa
"""Verify the generated DC tiles match the real SFO tiles, per class, with the same metrics.
  real GREEN shards   vs  generated park tiles
  real silver/grey    vs  generated land tiles
Shape metrics (circularity/solidity/extent/perim_ratio/elongation) are scale-invariant; we
also check the green-to-grey SIZE RATIO matches in both."""
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seg, metrics as M
import dc_build as DB

# ---- real SFO tiles, classified ------------------------------------------
img = np.asarray(Image.open("data/sfo_greenmap.jpg").convert("RGB"), float)
labels, n = seg.segment(img @ [0.299, 0.587, 0.114])
real_green, real_tess = [], []
for p in M.region_props(labels, min_area=60):
    sub = labels == p["label"]
    r, g, b = (img[..., k][sub].mean() for k in range(3))
    sat = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1e-6)
    (real_green if (g > r*1.08 and g > b*1.05 and sat > 0.12) else real_tess).append(p)

# ---- generated DC tiles ---------------------------------------------------
R = DB.render(900)
big, tr = R["big"], R["tile_region"]
gp = M.region_props(big, min_area=20)
gen_green = [p for p in gp if tr[p["label"]] == 2]
gen_grey  = [p for p in gp if tr[p["label"]] == 1]

KEYS = ["circularity", "solidity", "extent", "perim_ratio", "elongation"]
def med(props, k): return M.summarize(props)[k]["median"]
def cv(props):     return M.summarize(props)["area_cv"]
def diam(props):   return M.summarize(props)["equiv_diam"]["median"]

def block(title, real, gen):
    print(f"\n{title}   (real n={len(real)}, gen n={len(gen)})")
    print(f"  {'metric':12s}{'real':>8s}{'gen':>8s}{'Δ':>8s}")
    for k in KEYS:
        a, b = med(real, k), med(gen, k)
        print(f"  {k:12s}{a:8.2f}{b:8.2f}{abs(a-b):8.2f}")
    print(f"  {'area_cv':12s}{cv(real):8.2f}{cv(gen):8.2f}{abs(cv(real)-cv(gen)):8.2f}")

block("GREEN shards  (SFO green  vs  DC parks)", real_green, gen_green)
block("TESSERAE      (SFO silver/grey  vs  DC land)", real_tess, gen_grey)
print(f"\nSIZE RATIO green/grey   real {diam(real_green)/diam(real_tess):.2f}   "
      f"gen {diam(gen_green)/diam(gen_grey):.2f}   (should match)")

# ---- figure: distributions, real vs generated ----------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sets = [("GREEN", real_green, gen_green, "green"), ("TESSERAE", real_tess, gen_grey, "gray")]
for row, (name, real, gen, col) in enumerate(sets):
    for j, k in enumerate(["solidity", "extent", "circularity"]):
        a = ax[row, j]
        a.hist([p[k] for p in real], bins=20, density=True, alpha=0.55, label="real SFO", color="#c9923f")
        a.hist([p[k] for p in gen], bins=20, density=True, alpha=0.55, label=f"gen DC", color=col)
        a.set_title(f"{name}: {k}"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("output/metric_match.png", dpi=90)
print("\nwrote output/metric_match.png")
