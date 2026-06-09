"""Render generated patches at the fitted params and put them next to real crops,
so we judge by eye whether matching the metrics actually bought the look."""
import json
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate as G
import fracture as F
import bricks as B
import metrics as M

rng_hash = lambda n: (np.sin(n * 12.9898) * 43758.5453) % 1.0


def render(labels, kind):
    """Colour each tile; draw dark grout where adjacent labels differ."""
    H, W = labels.shape
    out = np.zeros((H, W, 3))
    ids = np.unique(labels)
    col = {}
    for i in ids:
        h = rng_hash(i); k = rng_hash(i * 1.7 + 5.3)
        if kind == "green":
            # muted olive→emerald, sampled-ish from real park tones
            col[i] = np.array([0.16 + 0.16*k, 0.34 + 0.24*h, 0.13 + 0.12*k])
        else:  # silver tesserae
            v = 0.78 + 0.18*(k-0.5)
            col[i] = np.array([v, v, v*1.02])
    flat = np.zeros((H, W, 3))
    for i in ids:
        flat[labels == i] = col[i]
    # grout: pixel differs from right or down neighbour
    g = np.zeros((H, W), bool)
    g[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    g[:-1, :] |= labels[:-1, :] != labels[1:, :]
    flat[g] = (0.10, 0.07, 0.05)
    return flat


# fitted params (full 5-metric objective, with edge texture)
green_p = json.load(open("analysis/fit_green_full.json"))["params"]
light_p = json.load(open("analysis/fit_light_full.json"))["params"]
print("green:", green_p)
print("light:", light_p)

gen_green = render(F.fracture_patch(N=460, n_pieces=80, crack_freq=0.3, seed=7, **green_p), "green")
gen_light = render(B.flow_brick_patch(N=460, cell=34, aniso=1.6, flow_swing=0.8, seed=7), "light")

# auto-pick real crops: greenest window for parks, brightest-desaturated for background
real = np.asarray(Image.open("/tmp/gm3c.png").convert("RGB")).astype(float)
H, W, _ = real.shape
S = 460
def best_window(scoremap):
    best, by, bx = -1e9, 0, 0
    for y in range(0, H - S, 60):
        for x in range(0, W - S, 60):
            m = scoremap[y:y+S, x:x+S].mean()
            if m > best: best, by, bx = m, y, x
    return by, bx
greenness = real[..., 1] - 0.5 * (real[..., 0] + real[..., 2])
lum = real @ [0.299, 0.587, 0.114]
gy, gx = best_window(greenness)
wy, wx = best_window(lum - 0.4 * np.abs(real[...,0]-real[...,2]))
real_green = real[gy:gy+S, gx:gx+S].astype(np.uint8)
real_white = real[wy:wy+S, wx:wx+S].astype(np.uint8)

fig, ax = plt.subplots(2, 2, figsize=(11, 11))
ax[0,0].imshow(real_green); ax[0,0].set_title("REAL — green parks");
ax[0,1].imshow(gen_green);  ax[0,1].set_title("GENERATED — fracture model")
ax[1,0].imshow(real_white); ax[1,0].set_title("REAL — silver background")
ax[1,1].imshow(gen_light);  ax[1,1].set_title("GENERATED — warped Voronoi")
for a in ax.ravel(): a.axis("off")
fig.tight_layout(); fig.savefig("analysis/compare.png", dpi=85)
print("wrote analysis/compare.png")
