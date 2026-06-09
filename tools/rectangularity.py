import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa
"""Operationalize 'square/rectangular' for the grey tesserae and compare real SFO vs gen DC.

extent (bbox-fill) can't tell a rectangle from a rounded blob. Two measures that can:
  - corner_count: vertices after RDP-simplifying the tile outline (a rectangle ≈ 4)
  - right_angle_frac: share of those corners within 90°±20° (rectangles ≈ 1.0)
Together: a 'squareness' signal that a rounded Voronoi-ish cell fails and a tessera passes."""
import numpy as np
from PIL import Image
from scipy import ndimage
from contourpy import contour_generator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seg, metrics as M, dc_build as DB


def rdp(pts, eps):
    pts = np.asarray(pts, float)
    if len(pts) < 5:
        return pts
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True; st = [(0, len(pts) - 1)]
    while st:
        i, j = st.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]; ab = b - a; L = np.hypot(*ab) + 1e-9
        d = np.abs((pts[i+1:j, 0]-a[0])*ab[1] - (pts[i+1:j, 1]-a[1])*ab[0]) / L
        k = int(d.argmax())
        if d[k] > eps:
            keep[i + 1 + k] = True; st += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def corner_angles(labels, props):
    """Collect EVERY corner's interior angle (degrees) across all tiles."""
    objs = ndimage.find_objects(labels)
    angs = []
    for p in props:
        sl = objs[p["label"] - 1]
        if sl is None:
            continue
        y0 = max(0, sl[0].start - 1); x0 = max(0, sl[1].start - 1)
        sub = (labels[y0:sl[0].stop + 1, x0:sl[1].stop + 1] == p["label"]).astype(float)
        if sub.sum() < 30:
            continue
        lines = contour_generator(z=sub).lines(0.5)
        if not lines:
            continue
        sp = rdp(max(lines, key=len), 0.12 * p["equiv_diam"])
        if len(sp) >= 2 and np.allclose(sp[0], sp[-1]):
            sp = sp[:-1]
        m = len(sp)
        if m < 3:
            continue
        for i in range(m):
            a = sp[(i - 1) % m] - sp[i]; b = sp[(i + 1) % m] - sp[i]
            angs.append(np.degrees(np.arccos(np.clip(a @ b / (np.hypot(*a)*np.hypot(*b)+1e-9), -1, 1))))
    return np.array(angs)


# real SFO tesserae (silver/grey)
img = np.asarray(Image.open("data/sfo_greenmap.jpg").convert("RGB"), float)
labels, _ = seg.segment(img @ [0.299, 0.587, 0.114])
real_tess = []
for p in M.region_props(labels, min_area=80):
    sub = labels == p["label"]; r, g, b = (img[..., k][sub].mean() for k in range(3))
    if not (g > r*1.08 and g > b*1.05 and (max(r,g,b)-min(r,g,b))/(max(r,g,b)+1e-6) > 0.12):
        real_tess.append(p)
ra_ang = corner_angles(labels, real_tess)

# generated grey = the new seamless deformed-quad tessellation, measured at a legible scale
sil = DB._quad_grid(700, 700, 28, seed=1)
gen_grey = M.region_props(sil, min_area=150)
print(f"real tesserae tiles: {len(real_tess)}   gen grey tiles: {len(gen_grey)}")
ge_ang = corner_angles(sil, gen_grey)

# ---- bucketed corner-angle distribution ----------------------------------
edges = np.arange(0, 181, 20)
rh, _ = np.histogram(ra_ang, bins=edges, density=False)
gh, _ = np.histogram(ge_ang, bins=edges, density=False)
rh = 100 * rh / rh.sum(); gh = 100 * gh / gh.sum()
print(f"corner-angle distribution (% of corners)   real n={len(ra_ang)}  gen n={len(ge_ang)}")
print(f"  {'bucket':12s}{'real':>8s}{'gen':>8s}")
for k in range(len(edges) - 1):
    bar_r = "#" * int(rh[k] / 2); bar_g = "." * int(gh[k] / 2)
    print(f"  {edges[k]:3d}-{edges[k+1]:3d}°  {rh[k]:6.1f}%{gh[k]:7.1f}%   {bar_r}|{bar_g}")
print(f"  median angle   real {np.median(ra_ang):.0f}°   gen {np.median(ge_ang):.0f}°")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ra_ang, bins=edges, density=True, alpha=.6, label="real SFO", color="#c9923f")
ax.hist(ge_ang, bins=edges, density=True, alpha=.6, label="gen DC", color="gray")
ax.axvline(90, color="k", ls="--", lw=1, alpha=.5)
ax.set_xlabel("corner interior angle (°)"); ax.set_title("Corner-angle distribution: real tesserae vs generated grey")
ax.legend()
fig.tight_layout(); fig.savefig("output/rectangularity.png", dpi=90)
print("wrote output/rectangularity.png")
