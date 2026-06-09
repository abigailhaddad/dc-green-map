"""Characterize the DISTRIBUTION of real tile shapes (a point-distribution / PCA shape model),
then GENERATE new polygons from it and show them beside real ones — per class, because
light (rectangular) and green (irregular interlocking) are different shape families.

Pipeline: resample each tile to K boundary points by arc length -> align (centre, scale to
unit RMS radius, rotate to principal axis, start at rightmost, force CCW) -> PCA over the
2K-vectors. New shapes = mean + sum z_i * sqrt(var_i) * PC_i."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 64


def resample(poly, K=K):
    poly = np.asarray(poly, float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    seg = np.hypot(*np.diff(poly, axis=0).T)
    L = np.concatenate([[0], np.cumsum(seg)])
    tot = L[-1]
    if tot == 0:
        return None
    targets = np.linspace(0, tot, K, endpoint=False)
    x = np.interp(targets, L, poly[:, 0])
    y = np.interp(targets, L, poly[:, 1])
    return np.column_stack([x, y])


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def normalize(poly):
    p = resample(poly)
    if p is None:
        return None
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)   # principal axes
    p = p @ vt.T                                        # major axis -> x
    rms = np.sqrt((p ** 2).sum(1).mean())
    if rms == 0:
        return None
    p = p / rms
    if signed_area(p) < 0:                              # force consistent winding
        p = p[::-1]
    i = int(np.argmax(p[:, 0]))                         # canonical start = rightmost
    p = np.roll(p, -i, axis=0)
    return p


shapes = json.load(open("analysis/tile_shapes.json"))


def model_for(cls, n_show=8, ncomp=6, seed=0):
    polys = [normalize(s["poly"]) for s in shapes if s["cls"] == cls]
    polys = [p for p in polys if p is not None]
    X = np.array([p.ravel() for p in polys])           # (N, 2K)
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = (S ** 2) / (len(X) - 1)
    ev = var / var.sum()
    print(f"{cls}: {len(X)} tiles, PC variance explained "
          f"{', '.join(f'{e*100:.0f}%' for e in ev[:5])}")
    rng = np.random.default_rng(seed)
    gens = []
    for _ in range(n_show):
        z = rng.standard_normal(ncomp)
        v = mean + (z * np.sqrt(var[:ncomp])) @ Vt[:ncomp]
        gens.append(v.reshape(K, 2))
    reals = [polys[i] for i in rng.choice(len(polys), n_show, replace=False)]
    return reals, gens, ev


fig, axes = plt.subplots(4, 8, figsize=(15, 8))
for r, cls in enumerate(["light", "green"]):
    reals, gens, ev = model_for(cls)
    col = (0.85, 0.85, 0.88) if cls == "light" else (0.30, 0.55, 0.20)
    for j in range(8):
        ar = axes[2 * r][j]
        ar.fill(reals[j][:, 0], -reals[j][:, 1], color=col, ec="black", lw=0.7)
        ar.set_aspect("equal"); ar.axis("off")
        if j == 0:
            ar.set_title(f"REAL {cls}", loc="left", fontsize=10)
        ag = axes[2 * r + 1][j]
        ag.fill(gens[j][:, 0], -gens[j][:, 1], color=col, ec="crimson", lw=0.7)
        ag.set_aspect("equal"); ag.axis("off")
        if j == 0:
            ag.set_title(f"GENERATED {cls}", loc="left", fontsize=10)
fig.suptitle("Real (black) vs PCA-generated (red) tile shapes, aligned & normalized")
fig.tight_layout(); fig.savefig("analysis/shapes_generated_vs_real.png", dpi=95)
print("wrote analysis/shapes_generated_vs_real.png")
