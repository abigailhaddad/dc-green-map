"""Re-fit BOTH regions on the full objective {area_cv, circularity, solidity, extent,
perim_ratio} now that we have the fine-edge-texture mechanism.
  background (silver) = Chebyshev Voronoi + texture   (rectangular, ragged, size-varied)
  parks (green)       = fracture + texture             (interlocking, compact, ragged)"""
import json, itertools
import numpy as np
import generate as G
import fracture as F
import metrics as M

T = json.load(open("analysis/target_metrics.json"))["by_class"]
KEYS = ["area_cv", "circularity", "solidity", "extent", "perim_ratio"]
W = dict(area_cv=1.0, circularity=1.0, solidity=1.3, extent=1.3, perim_ratio=1.0)


def target(cls):
    c = T[cls]
    return dict(area_cv=c["area_cv"], circularity=c["circularity"]["median"],
                solidity=c["solidity"]["median"], extent=c["extent"]["median"],
                perim_ratio=c["perim_ratio"]["median"])


def got(summ):
    return dict(area_cv=summ["area_cv"], circularity=summ["circularity"]["median"],
                solidity=summ["solidity"]["median"], extent=summ["extent"]["median"],
                perim_ratio=summ["perim_ratio"]["median"])


def score(summ, t):
    if not summ:
        return 1e9, {}
    g = got(summ)
    return sum(W[k] * ((g[k] - t[k]) / (abs(t[k]) + 1e-6)) ** 2 for k in W), g


def run(name, t, make, grid, min_area):
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"\n=== {name} ===  want " +
          " ".join(f"{k}={t[k]:.2f}" for k in KEYS), flush=True)
    print(f"searching {len(combos)} combos x2 seeds", flush=True)
    best = (1e9, None, None)
    for i, vals in enumerate(combos, 1):
        kw = dict(zip(keys, vals))
        ss, gg = [], []
        for sd in (1, 2):
            lab = make(kw, sd)
            sc, g = score(M.summarize(M.region_props(lab, min_area=min_area)), t)
            ss.append(sc); gg.append(g)
        m = float(np.mean(ss))
        if m < best[0] and all(gg):
            avg = {k: float(np.mean([x[k] for x in gg])) for k in gg[0]}
            best = (m, kw, avg)
            print(f"  [{i}/{len(combos)}] score={m:.3f}  " +
                  " ".join(f"{k}={avg[k]:.2f}" for k in KEYS) + f"  {kw}", flush=True)
        elif i % 10 == 0:
            print(f"  [{i}/{len(combos)}]...", flush=True)
    print(f"BEST {name}: {best[1]}\n  got " +
          " ".join(f"{k}={best[2][k]:.2f}" for k in KEYS) + f"  (score {best[0]:.3f})", flush=True)
    json.dump({"params": best[1], "achieved": best[2], "target": t},
              open(f"analysis/fit_{name}_full.json", "w"), indent=2)
    return best


run("light", target("light"),
    lambda kw, sd: G.make_patch(N=480, base=40, warp_corr=0.35, metric="linf", seed=sd, **kw),
    dict(size_var=[1.0, 1.6, 2.2], warp_amp=[2, 6], tex_amp=[0, 2, 4], tex_corr=[4, 7]),
    min_area=180)

run("green", target("green"),
    lambda kw, sd: F.fracture_patch(N=460, n_pieces=120, crack_freq=0.3, seed=sd, **kw),
    dict(crack_rough=[0.15, 0.25, 0.4], sel_pow=[0.4], axis_align=[0.5, 1.0],
         tex_amp=[0, 2, 4], tex_corr=[4, 7]),
    min_area=120)
