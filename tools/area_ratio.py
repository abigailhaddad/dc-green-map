import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa
"""Does the green-tile vs grey-tile AREA ratio match the original? (Ratios are scale-free,
so comparable even though the render is at a different absolute scale.)"""
import numpy as np
from PIL import Image
import seg, metrics as M, dc_build as DB

# real SFO: green shards vs silver/grey tesserae
img = np.asarray(Image.open("data/sfo_greenmap.jpg").convert("RGB"), float)
labels, _ = seg.segment(img @ [0.299, 0.587, 0.114])
rg, rt = [], []
for p in M.region_props(labels, min_area=60):
    sub = labels == p["label"]; r, g, b = (img[..., k][sub].mean() for k in range(3))
    sat = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1e-6)
    (rg if (g > r*1.08 and g > b*1.05 and sat > 0.12) else rt).append(p["area"])
rg, rt = np.array(rg), np.array(rt)

# generated DC: parks (green) vs land (grey)
R = DB.render(1600); big, tr = R["big"], R["tile_region"]
gg, ggr = [], []
for p in M.region_props(big, min_area=8):
    if tr[p["label"]] == 2: gg.append(p["area"])
    elif tr[p["label"]] == 1: ggr.append(p["area"])
gg, ggr = np.array(gg), np.array(ggr)

print("green / grey  AREA ratio (green bigger => >1)")
print(f"  by median : real {np.median(rg)/np.median(rt):.2f}   gen {np.median(gg)/np.median(ggr):.2f}")
print(f"  by mean   : real {rg.mean()/rt.mean():.2f}   gen {gg.mean()/ggr.mean():.2f}")
print(f"\nreal: green med {np.median(rg):.0f}px  grey med {np.median(rt):.0f}px   (n {len(rg)}/{len(rt)})")
print(f"gen : green med {np.median(gg):.0f}px  grey med {np.median(ggr):.0f}px   (n {len(gg)}/{len(ggr)})")
