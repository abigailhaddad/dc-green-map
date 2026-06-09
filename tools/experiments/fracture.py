"""Fracture model for the green park shards: recursively shatter a region with JAGGED cuts.

Why this and not Voronoi:
  - recursively splitting area-weighted pieces gives a HEAVY-TAILED size mix (few big + many
    tiny) => high area CV, which Voronoi can't reach.
  - jagged (noisy) cut paths make the boundary between two pieces concave and interlocking
    => low solidity / low circularity, the 'pieces that fit into each other' look.

Knobs:
  n_pieces    -> absolute count / median size
  crack_rough -> cut waviness as a fraction of local scale (drives solidity & circularity down)
  crack_freq  -> wavelength of the jaggedness (fraction of local scale)
  sel_pow     -> how strongly big pieces are preferred for splitting (shapes the size tail)
"""
import numpy as np


def _noise1d(b, wavelen, rng):
    bmin, bmax = b.min(), b.max()
    span = max(bmax - bmin, 1.0)
    ncp = max(3, int(span / max(wavelen, 2)) + 2)
    xs = np.linspace(bmin, bmax, ncp)
    cps = rng.standard_normal(ncp)
    return np.interp(b, xs, cps)


def fracture_patch(N=600, n_pieces=400, crack_rough=0.6, crack_freq=0.5, sel_pow=1.0,
                   axis_align=0.0, tex_amp=0.0, tex_corr=5.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.ones((N, N), int)
    coords = {1: np.argwhere(labels == 1)}      # label -> (k,2) row,col pixels
    areas = {1: N * N}
    nxt = 2
    typical = np.sqrt(N * N / n_pieces)          # expected piece scale (px)
    for _ in range(n_pieces - 1):
        labs = np.fromiter(areas.keys(), int)
        w = np.array([areas[l] for l in labs], float) ** sel_pow
        w /= w.sum()
        p = int(rng.choice(labs, p=w))
        pix = coords[p]
        if len(pix) < 16:
            areas[p] = -1e9 if False else areas[p]   # too small to split; skip by zeroing weight
            areas[p] = 1                              # keep but unlikely re-picked
            continue
        ys = pix[:, 0].astype(float)
        xs = pix[:, 1].astype(float)
        if rng.random() < axis_align and len(pix) > 32:
            # cut across the long axis so shards stay equant (lowers elongation)
            cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
            ev, evec = np.linalg.eigh(cov)
            major = evec[:, 1]
            th = np.arctan2(major[1], major[0]) + rng.normal(0, 0.25)
        else:
            th = rng.uniform(0, np.pi)
        a = xs * np.cos(th) + ys * np.sin(th)         # along cut normal
        b = -xs * np.sin(th) + ys * np.cos(th)        # perpendicular
        # local scale of THIS piece, so roughness scales with piece size
        scale = np.sqrt(len(pix))
        q = rng.uniform(0.35, 0.65)
        cut = np.quantile(a, q)
        jag = crack_rough * scale * _noise1d(b, crack_freq * scale, rng)
        side = a < (cut + jag)
        if side.sum() < 8 or (~side).sum() < 8:
            continue
        new = nxt; nxt += 1
        moved = pix[~side]
        labels[moved[:, 0], moved[:, 1]] = new
        coords[p] = pix[side]
        coords[new] = moved
        areas[p] = len(coords[p])
        areas[new] = len(moved)
    if tex_amp > 0:
        from generate import add_edge_texture
        labels = add_edge_texture(labels, tex_amp, tex_corr, seed + 4242)
    return labels


if __name__ == "__main__":
    import metrics as M
    for kw in [dict(n_pieces=300, crack_rough=0.3, crack_freq=0.5),
               dict(n_pieces=300, crack_rough=0.8, crack_freq=0.4),
               dict(n_pieces=450, crack_rough=1.1, crack_freq=0.3, sel_pow=1.4)]:
        lab = fracture_patch(seed=1, **kw)
        s = M.summarize(M.region_props(lab, min_area=30))
        print(M.fmt(s, str(kw)))
