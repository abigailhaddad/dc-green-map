"""Procedural tile patch generator, measured by the SAME metrics.py as the real photo.

Two knobs map directly onto the two target numbers:
  - warp_amp   -> solidity / circularity   (domain warp bends edges concave => pieces interlock)
  - size_var   -> area CV                   (low-frequency density variation => big+small mix)
base controls absolute tile size.  Everything returns an integer label map (one int per tile)."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def _vnoise(N, corr, rng):
    """Smooth value noise in [-1,1], correlation length ~corr px."""
    c = max(2, int(round(N / max(corr, 1))))
    coarse = rng.standard_normal((c + 2, c + 2))
    z = ndimage.zoom(coarse, (N + 1) / coarse.shape[0], order=3)[:N, :N]
    z -= z.mean()
    m = np.abs(z).max() or 1.0
    return z / m


def _variable_poisson(N, base, size_var, rng, k_darts=60):
    """Dart-throwing Poisson with spacing that varies spatially (=> size variation).
    Two octaves of density noise give the heavy-tailed big+small mix seen in real parks."""
    n = 0.65 * _vnoise(N, base * 5, rng) + 0.45 * _vnoise(N, base * 1.5, rng)
    field = base * (1.0 + size_var * n)
    field = np.clip(field, base * 0.25, base * 6)
    cell = max(1.0, field.min() / np.sqrt(2))
    g = int(np.ceil(N / cell))
    grid = -np.ones((g, g), int)
    pts = []
    darts = int(k_darts * (N / base) ** 2)
    for _ in range(darts):
        r, c = rng.random() * N, rng.random() * N
        s = field[min(N - 1, int(r)), min(N - 1, int(c))]
        gr, gc = int(r / cell), int(c / cell)
        rad = int(np.ceil(s / cell)) + 1
        ok = True
        for ar in range(max(0, gr - rad), min(g, gr + rad + 1)):
            for ac in range(max(0, gc - rad), min(g, gc + rad + 1)):
                q = grid[ar, ac]
                if q < 0:
                    continue
                pr, pc = pts[q]
                if (r - pr) ** 2 + (c - pc) ** 2 < s * s:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[gr, gc] = len(pts)
            pts.append((r, c))
    return np.array(pts, float)


def add_edge_texture(labels, amp, corr, seed):
    """Jitter every tile boundary at the pixel scale (high-freq, low-amp backward warp).
    Raises perimeter (lowers circularity, raises perim_ratio) WITHOUT big lobes, so extent
    and solidity stay put — the roughness of hand-cut glass that clean cells lack."""
    if amp <= 0:
        return labels
    N = labels.shape[0]
    rng = np.random.default_rng(seed)
    dr = amp * _vnoise(N, corr, rng)
    dc = amp * _vnoise(N, corr, np.random.default_rng(seed + 7))
    rr, cc = np.mgrid[0:N, 0:N]
    sr = np.clip(np.round(rr + dr).astype(int), 0, N - 1)
    sc = np.clip(np.round(cc + dc).astype(int), 0, N - 1)
    return labels[sr, sc]


def make_patch(N=600, base=37.0, size_var=0.25, warp_amp=0.0, warp_corr=0.45, seed=0,
               metric="l2", tex_amp=0.0, tex_corr=5.0):
    """warp_corr is a FRACTION of base: <1 => sub-tile wiggles that fold boundaries
    concave (lowers solidity), which is what makes pieces interlock.
    metric='linf' => Chebyshev Voronoi => RECTANGULAR cells (for the silver tesserae)."""
    rng = np.random.default_rng(seed)
    seeds = _variable_poisson(N, base, size_var, rng)
    if len(seeds) < 4:
        return np.ones((N, N), int)
    rr, cc = np.mgrid[0:N, 0:N]
    if warp_amp > 0:
        corr = base * warp_corr
        wr = warp_amp * _vnoise(N, corr, rng)
        wc = warp_amp * _vnoise(N, corr, np.random.default_rng(seed + 999))
        qr = (rr + wr).ravel()
        qc = (cc + wc).ravel()
    else:
        qr, qc = rr.ravel().astype(float), cc.ravel().astype(float)
    tree = cKDTree(seeds)
    p = np.inf if metric == "linf" else 2
    _, idx = tree.query(np.column_stack([qr, qc]), p=p)
    labels = idx.reshape(N, N) + 1
    return add_edge_texture(labels, tex_amp, tex_corr, seed + 4242)


if __name__ == "__main__":
    import metrics as M
    # quick smoke test: a 'light/blocky' patch vs a 'green/interlocking' patch
    for name, kw in [
        ("light-ish  (low warp, low var)", dict(base=37, size_var=0.20, warp_amp=3)),
        ("green-ish  (high warp, high var)", dict(base=26, size_var=0.9, warp_amp=16)),
    ]:
        lab = make_patch(seed=1, **kw)
        s = M.summarize(M.region_props(lab, min_area=40))
        print(M.fmt(s, name))
