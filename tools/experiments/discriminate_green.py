"""Which metrics SEPARATE real-green from generated-green? Those are the features our
objective was missing. We compute a battery on both and rank by standardized difference
(|mean_r - mean_g| / pooled_std) — a Cohen's-d-style effect size. Big d = the fit was
blind to that feature, so generated could drift on it freely."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage

import metrics as M
import fracture as F

# ---- real green tiles -----------------------------------------------------
real = np.asarray(Image.open("/tmp/gm3c.png").convert("RGB"), float)
lum = real @ [0.299, 0.587, 0.114]
closed = ndimage.grey_closing(lum, size=15)
grout = (closed - lum) > 10
tiles = ndimage.binary_opening(~ndimage.binary_closing(grout), iterations=1)
labels, _ = ndimage.label(tiles)
props = M.region_props(labels, min_area=120)
green_props = []
for p in props:
    sub = labels == p["label"]
    r, g, b = (real[..., k][sub].mean() for k in range(3))
    sat = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + 1e-6)
    if g > r * 1.08 and g > b * 1.05 and sat > 0.12:
        p["hue_g_minus_rb"] = float(g - 0.5 * (r + b))
        p["val"] = float((r + g + b) / 3)
        green_props.append(p)

# ---- adjacency (neighbour count) helper ----------------------------------
def neighbour_counts(lab):
    out = {}
    for ax in (0, 1):
        a = lab
        b = np.roll(lab, 1, axis=ax)
        m = a != b
        for u, v in zip(a[m], b[m]):
            if u and v and u != v:
                out.setdefault(int(u), set()).add(int(v))
                out.setdefault(int(v), set()).add(int(u))
    return out

real_nb = neighbour_counts(labels)
for p in green_props:
    p["neighbours"] = len(real_nb.get(p["label"], ()))

# ---- generated green tiles (best fracture params) ------------------------
gp = json.load(open("analysis/fit_green_fracture.json"))["params"]
gen_all = []
gen_nb_counts = []
for sd in (11, 12, 13):
    lab = F.fracture_patch(N=460, n_pieces=70, seed=sd, **gp)
    pr = M.region_props(lab, min_area=120)
    nb = neighbour_counts(lab)
    for p in pr:
        p["neighbours"] = len(nb.get(p["label"], ()))
    gen_all += pr

# ---- compare --------------------------------------------------------------
FEATURES = ["circularity", "solidity", "perim_ratio", "extent", "elongation", "neighbours"]
def col(props, k):
    return np.array([p[k] for p in props if k in p], float)

print(f"real green tiles: {len(green_props)}   generated tiles: {len(gen_all)}\n")
print(f"{'feature':14s} {'real':>10s} {'gen':>10s} {'|d|':>7s}   {'in objective?':>14s}")
rows = []
for k in FEATURES:
    r, g = col(green_props, k), col(gen_all, k)
    pooled = np.sqrt((r.std()**2 + g.std()**2) / 2) + 1e-9
    d = abs(r.mean() - g.mean()) / pooled
    used = k in ("circularity", "solidity", "elongation")  # area_cv handled separately
    rows.append((d, k, r.mean(), g.mean(), used))
for d, k, rm, gm, used in sorted(rows, reverse=True):
    flag = "in fit" if used else ">> MISSING <<"
    print(f"{k:14s} {rm:10.3f} {gm:10.3f} {d:7.2f}   {flag:>14s}")

print("\n(area_cv was in the fit; colour was never measured at all.)")
