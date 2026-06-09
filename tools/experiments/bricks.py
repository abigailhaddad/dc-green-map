"""Brick model for the silver background: genuine rectangles laid in rows (opus tessellatum).
Unlike L-inf Voronoi this makes actual 4-sided tiles AND gives row alignment (andamento)
for free. Light edge texture adds the hand-cut wobble without destroying rectangularity."""
import numpy as np
from generate import add_edge_texture, _vnoise


def flow_brick_patch(N=460, cell=34.0, jitter=0.35, aniso=1.6, flow_swing=0.7,
                     flow_corr=170.0, size_var=0.5, tex_amp=1.2, tex_corr=4.0, seed=0):
    """Rectangular tesserae oriented by a smooth FLOW FIELD, so courses wander and curve
    (andamento) instead of forming one rigid global grid. aniso>1 => tiles longer along
    the local course direction. size_var => spatially-varying tile size."""
    rng = np.random.default_rng(seed)
    # jittered-grid seeds, with spatially-varying spacing for size variation
    dens = 1.0 + size_var * _vnoise(N, cell * 5, rng)
    sx, sy = [], []
    yy = cell / 2
    while yy < N:
        xx = cell / 2
        while xx < N:
            local = cell * dens[min(N - 1, int(yy)), min(N - 1, int(xx))]
            sx.append(xx + jitter * cell * rng.uniform(-1, 1))
            sy.append(yy + jitter * cell * rng.uniform(-1, 1))
            xx += max(cell * 0.5, local)
        yy += cell
    sx, sy = np.array(sx), np.array(sy)
    # smooth flow-angle field; each seed takes the local course direction
    fnoise = _vnoise(N, flow_corr, np.random.default_rng(seed + 3))
    si = np.clip(sy.astype(int), 0, N - 1)
    sj = np.clip(sx.astype(int), 0, N - 1)
    th = flow_swing * fnoise[si, sj]
    cs, sn = np.cos(th), np.sin(th)
    Y, X = np.mgrid[0:N, 0:N]
    best = np.full((N, N), np.inf)
    idx = np.zeros((N, N), int)
    for s in range(len(sx)):
        dx = X - sx[s]; dy = Y - sy[s]
        u = dx * cs[s] + dy * sn[s]          # along the local course
        v = -dx * sn[s] + dy * cs[s]         # across it
        d = np.maximum(np.abs(u) / aniso, np.abs(v))   # anisotropic Chebyshev => oriented rectangle
        m = d < best
        best[m] = d[m]; idx[m] = s + 1
    return add_edge_texture(idx, tex_amp, tex_corr, seed + 1)


def brick_patch(N=460, row_h=34.0, col_w=38.0, jit=0.45, offset_jit=0.7,
                tex_amp=1.5, tex_corr=4.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros((N, N), int)
    lab = 1
    # row boundaries (slightly varying heights)
    rows, y = [], 0.0
    while y < N:
        h = row_h * (1 + jit * rng.uniform(-1, 1))
        rows.append((int(round(y)), min(N, int(round(y + h)))))
        y += h
    for (y0, y1) in rows:
        if y1 <= y0:
            continue
        x = -rng.uniform(0, 1) * col_w * offset_jit          # brick phase offset per row
        while x < N:
            w = col_w * (1 + jit * rng.uniform(-1, 1))
            x0, x1 = max(0, int(round(x))), min(N, int(round(x + w)))
            if x1 > x0:
                labels[y0:y1, x0:x1] = lab
                lab += 1
            x += w
    return add_edge_texture(labels, tex_amp, tex_corr, seed + 1)


if __name__ == "__main__":
    import metrics as M
    print("TARGET light: extent 0.72  circ 0.54  solidity 0.92  area_cv 0.79")
    for jit in (0.3, 0.45, 0.6):
        for ta in (0, 1.5, 3):
            lab = brick_patch(N=460, jit=jit, tex_amp=ta, seed=2)
            s = M.summarize(M.region_props(lab, min_area=150))
            print("jit=%.2f tex=%.1f  extent=%.3f circ=%.2f sol=%.3f cv=%.2f"
                  % (jit, ta, s["extent"]["median"], s["circularity"]["median"],
                     s["solidity"]["median"], s["area_cv"]))
