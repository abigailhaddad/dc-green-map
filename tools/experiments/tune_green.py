"""Fit the fracture model to the real GREEN park shards. Scale-invariant traits only
(solidity, area_cv, circularity, elongation) — absolute size is set later by the renderer."""
import json, itertools
import numpy as np
import fracture as F
import metrics as M

T = json.load(open("analysis/target_metrics.json"))["by_class"]["green"]
tgt = dict(solidity=T["solidity"]["median"], area_cv=T["area_cv"],
           circularity=T["circularity"]["median"], elongation=T["elongation"]["median"])
W = dict(solidity=2.0, area_cv=1.5, circularity=1.5, elongation=1.0)


def score(s):
    if not s:
        return 1e9, {}
    got = dict(solidity=s["solidity"]["median"], area_cv=s["area_cv"],
               circularity=s["circularity"]["median"], elongation=s["elongation"]["median"])
    return sum(W[k] * ((got[k] - tgt[k]) / (abs(tgt[k]) + 1e-6)) ** 2 for k in W), got


grid = dict(
    crack_rough=[0.25, 0.4, 0.6],
    crack_freq=[0.2, 0.35, 0.55],
    sel_pow=[0.0, 0.4],
    axis_align=[0.5, 1.0],
)
combos = list(itertools.product(*grid.values()))
print(f"want: {{'solidity': {tgt['solidity']:.2f}, 'area_cv': {tgt['area_cv']:.2f}, "
      f"'circularity': {tgt['circularity']:.2f}, 'elongation': {tgt['elongation']:.2f}}}", flush=True)
print(f"searching {len(combos)} combos x2 seeds...", flush=True)
best = (1e9, None, None)
for i, (cr, cf, sp, aa) in enumerate(combos, 1):
    ss, gg = [], []
    for sd in (1, 2):
        lab = F.fracture_patch(N=520, n_pieces=300, crack_rough=cr, crack_freq=cf,
                               sel_pow=sp, axis_align=aa, seed=sd)
        sc, got = score(M.summarize(M.region_props(lab, min_area=30)))
        ss.append(sc); gg.append(got)
    m = float(np.mean(ss))
    if m < best[0]:
        avg = {k: float(np.mean([g[k] for g in gg])) for k in gg[0]}
        best = (m, dict(crack_rough=cr, crack_freq=cf, sel_pow=sp, axis_align=aa), avg)
        print(f"  [{i}/{len(combos)}] new best score={m:.3f}  "
              f"sol={avg['solidity']:.2f} cv={avg['area_cv']:.2f} "
              f"circ={avg['circularity']:.2f} elong={avg['elongation']:.2f}  {best[1]}", flush=True)
    elif i % 6 == 0:
        print(f"  [{i}/{len(combos)}]...", flush=True)

print("\nwant:", {k: round(v, 3) for k, v in tgt.items()})
print("best:", best[1])
print("got :", {k: round(v, 3) for k, v in best[2].items()}, f"(score {best[0]:.3f})")
json.dump({"params": best[1], "achieved": best[2], "target": tgt},
          open("analysis/fit_green_fracture.json", "w"), indent=2)
