"""Bake-off: fill the SAME area with different 'tiles that fit together' strategies, all
rendered identically (real silver colors sampled from the model + uniform grout at 8% of
tile diameter, warm-grey). Only the tiling strategy varies, so we can judge which looks
most like the real mosaic."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seg import segment
from fracture import fracture_patch

N = 440
GROUT_RGB = np.array([0.34, 0.35, 0.29])
light_colors = np.array(json.load(open("analysis/color_model.json"))["light"])


def render(labels, seed=0):
    """label map -> image: each tile a sampled real silver colour, uniform 8% grout."""
    n = int(labels.max())
    areas = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    med = np.median([a for a in areas if a > 20]) if n else 50
    diam = np.sqrt(4 * med / np.pi)
    gpx = max(1, int(round(0.08 * diam)))
    rng = np.random.default_rng(seed)
    lut = np.zeros((n + 1, 3))
    lut[1:] = light_colors[rng.integers(len(light_colors), size=n)]
    out = lut[labels]
    bound = np.zeros_like(labels, bool)
    bound[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    bound[:-1, :] |= labels[:-1, :] != labels[1:, :]
    g = ndimage.binary_dilation(bound, iterations=max(1, gpx // 2))
    out[g] = GROUT_RGB
    out[labels == 0] = GROUT_RGB
    return np.clip(out, 0, 1), gpx


def voronoi(N, cell, jit, seed, p=2, lloyd=0):
    rng = np.random.default_rng(seed)
    pts = []
    for y in np.arange(cell / 2, N, cell):
        for x in np.arange(cell / 2, N, cell):
            pts.append([y + jit * cell * rng.uniform(-1, 1), x + jit * cell * rng.uniform(-1, 1)])
    pts = np.array(pts)
    Y, X = np.mgrid[0:N, 0:N]
    grid = np.column_stack([Y.ravel(), X.ravel()])
    for _ in range(lloyd):
        _, idx = cKDTree(pts).query(grid)
        idx = idx.reshape(N, N)
        for i in range(len(pts)):
            m = idx == i
            if m.any():
                pts[i] = [Y[m].mean(), X[m].mean()]
    _, idx = cKDTree(pts).query(grid, p=p)
    return idx.reshape(N, N) + 1


# real-layout transplant: real tile shapes from a silver crop of the photo
img = np.asarray(Image.open("/tmp/gm3c.png").convert("RGB"), float)
crop = img[300:300 + N, 1150:1150 + N]
real_labels, _ = segment(crop @ [0.299, 0.587, 0.114])

strategies = [
    ("REAL crop (ground truth)", None),
    ("real-layout transplant", real_labels),
    ("jittered Voronoi (L2)", voronoi(N, 34, 0.45, 7, p=2)),
    ("relaxed Voronoi (Lloyd x3)", voronoi(N, 34, 0.7, 7, p=2, lloyd=3)),
    ("L-inf Voronoi (rectangles)", voronoi(N, 34, 0.45, 7, p=np.inf)),
    ("fracture", fracture_patch(N=N, n_pieces=150, crack_rough=0.25, sel_pow=0.4,
                                axis_align=1.0, tex_amp=4, tex_corr=4, seed=7)),
]

fig, ax = plt.subplots(2, 3, figsize=(16, 11))
for a, (title, lab) in zip(ax.ravel(), strategies):
    if lab is None:
        a.imshow(crop.astype(np.uint8)); a.set_title(title)
    else:
        rgb, gpx = render(lab, seed=3)
        a.imshow(rgb); a.set_title(f"{title}  (grout {gpx}px)")
    a.axis("off")
fig.suptitle("Tiling strategies — same area, same colors + grout; which fits like a real mosaic?",
             fontsize=13)
fig.tight_layout(); fig.savefig("analysis/strategies.png", dpi=85)
print("wrote analysis/strategies.png")
