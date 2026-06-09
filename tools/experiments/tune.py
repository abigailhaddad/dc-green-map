"""Search generator parameters until the generated patch's metrics match a real region.

Targets come from analysis/target_metrics.json (produced by segment_real.py). We match the
three numbers that capture what the eye notices:
  solidity  (interlocking),  area_cv (size variation),  circularity (raggedness).
base is matched separately to the real median equivalent diameter (absolute scale)."""
import json, itertools
import numpy as np
import generate as G
import metrics as M

with open("analysis/target_metrics.json") as f:
    T = json.load(f)

# weight solidity & area_cv most — they're the traits called out by eye
WEIGHTS = dict(solidity=2.0, area_cv=2.0, circularity=1.0, elongation=0.5)


def score(summary, tgt):
    if not summary:
        return 1e9, {}
    got = dict(
        solidity=summary["solidity"]["median"],
        area_cv=summary["area_cv"],
        circularity=summary["circularity"]["median"],
        elongation=summary["elongation"]["median"],
    )
    s = 0.0
    for k, w in WEIGHTS.items():
        s += w * ((got[k] - tgt[k]) / (abs(tgt[k]) + 1e-6)) ** 2
    return s, got


def target_for(cls):
    c = T["by_class"][cls]
    return dict(
        solidity=c["solidity"]["median"],
        area_cv=c["area_cv"],
        circularity=c["circularity"]["median"],
        elongation=c["elongation"]["median"],
        equiv_diam=c["equiv_diam"]["median"],
    )


def search(cls, seeds=(1, 2, 3)):
    tgt = target_for(cls)
    base = tgt["equiv_diam"] * 1.15   # cell diam ~ a bit over equiv-diam of the inscribed tile
    grid = dict(
        size_var=[0.2, 0.5, 0.9, 1.3, 1.8],
        warp_amp=[0, base * 0.12, base * 0.25, base * 0.4, base * 0.6],
        warp_corr=[0.3, 0.5, 0.8],
    )
    best = (1e9, None, None)
    for sv, wa, wc in itertools.product(grid["size_var"], grid["warp_amp"], grid["warp_corr"]):
        sols, gots = [], []
        for sd in seeds:
            lab = G.make_patch(N=600, base=base, size_var=sv, warp_amp=wa, warp_corr=wc, seed=sd)
            summ = M.summarize(M.region_props(lab, min_area=max(30, base * base * 0.15)))
            sc, got = score(summ, tgt)
            sols.append(sc); gots.append(got)
        msc = float(np.mean(sols))
        if msc < best[0]:
            avg = {k: float(np.mean([g[k] for g in gots])) for k in gots[0]}
            best = (msc, dict(base=round(base, 1), size_var=sv, warp_amp=round(wa, 1), warp_corr=wc), avg)
    return tgt, best


for cls in ("light", "green"):
    tgt, (sc, params, got) = search(cls)
    print(f"\n=== TARGET: {cls} ===")
    print("  want:", {k: round(v, 3) for k, v in tgt.items() if k != 'equiv_diam'})
    print("  best params:", params)
    print("  got :", {k: round(v, 3) for k, v in got.items()}, f"(score {sc:.3f})")
    json.dump({"class": cls, "params": params, "achieved": got, "target": tgt},
              open(f"analysis/fit_{cls}.json", "w"), indent=2)
