"""Shared tile metrics — the SAME code measures real (segmented) and generated tiles,
so the numbers are comparable even if any individual measure is approximate."""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


def region_props(labels, min_area=60, max_area_frac=0.25):
    """Per-tile geometry from an integer label map (0 = background/grout).
    Returns a list of dicts. Metrics are pixel-based but mostly scale-relative."""
    h, w = labels.shape
    max_area = max_area_frac * h * w
    props = []
    slices = ndimage.find_objects(labels)
    for lab, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub = labels[sl] == lab
        area = int(sub.sum())
        if area < min_area or area > max_area:
            continue
        # perimeter = boundary pixels (region pixel touching non-region, 4-conn)
        er = ndimage.binary_erosion(sub, border_value=0)
        perim = int((sub & ~er).sum())
        if perim == 0:
            continue
        circularity = float(4 * np.pi * area / (perim * perim))
        # solidity = area / convex-hull area  (LOW => concave / interlocking edges)
        coords = np.argwhere(sub)
        solidity = 1.0
        perim_ratio = 1.0       # boundary length / hull-boundary length: edge roughness, area-independent
        hull = None
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                if hull.volume > 0:
                    solidity = float(area / hull.volume)
                if hull.area > 0:          # in 2D scipy: .area == hull PERIMETER
                    perim_ratio = float(perim / hull.area)
            except Exception:
                hull = None
        cy, cx = coords.mean(0)
        ys = coords[:, 0] - cy
        xs = coords[:, 1] - cx
        cov = np.array([[ (xs*xs).mean(), (xs*ys).mean() ],
                        [ (xs*ys).mean(), (ys*ys).mean() ]])
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-6, None)
        elong = float(np.sqrt(ev[1] / ev[0]))
        # rectangularity = area / minimum-area oriented bbox (sweep angles on the hull;
        # robust to the square-degeneracy that principal-axis boxes suffer)
        hp = coords[hull.vertices] if hull is not None else coords
        hy = hp[:, 0] - cy; hx = hp[:, 1] - cx
        best_bbox = np.inf
        for ang in range(0, 90, 3):
            t = np.radians(ang); ct, st = np.cos(t), np.sin(t)
            u = hx * ct - hy * st; v = hx * st + hy * ct
            best_bbox = min(best_bbox, (np.ptp(u) + 1) * (np.ptp(v) + 1))
        extent = float(area / best_bbox) if best_bbox > 0 else 1.0
        props.append(dict(
            label=lab, area=area, perim=perim,
            circularity=min(circularity, 1.0),
            solidity=min(solidity, 1.0),
            perim_ratio=max(perim_ratio, 1.0),
            extent=min(extent, 1.0),
            elongation=elong,
            equiv_diam=float(np.sqrt(4 * area / np.pi)),
            cy=float(cy + sl[0].start), cx=float(cx + sl[1].start),
        ))
    return props


def _stats(vals):
    a = np.asarray(vals, float)
    if a.size == 0:
        return {}
    return dict(
        n=int(a.size),
        median=float(np.median(a)),
        mean=float(a.mean()),
        std=float(a.std()),
        p10=float(np.percentile(a, 10)),
        p90=float(np.percentile(a, 90)),
        cv=float(a.std() / a.mean()) if a.mean() else 0.0,  # size/shape spread
    )


def summarize(props):
    """Distribution summary used to compare a real patch against generated output."""
    if not props:
        return {}
    areas = [p["area"] for p in props]
    return dict(
        count=len(props),
        area=_stats(areas),
        # area spread normalized to its own median => scale-invariant "size variation"
        area_cv=_stats(areas)["cv"],
        equiv_diam=_stats([p["equiv_diam"] for p in props]),
        circularity=_stats([p["circularity"] for p in props]),
        solidity=_stats([p["solidity"] for p in props]),
        perim_ratio=_stats([p.get("perim_ratio", 1.0) for p in props]),
        extent=_stats([p.get("extent", 1.0) for p in props]),
        elongation=_stats([p["elongation"] for p in props]),
    )


def fmt(summary, title=""):
    if not summary:
        return f"{title}: (no tiles)\n"
    s = summary
    L = [f"== {title} ==", f"tiles: {s['count']}"]
    L.append(f"area px      median={s['area']['median']:.0f}  cv={s['area']['cv']:.2f}  "
             f"p10={s['area']['p10']:.0f} p90={s['area']['p90']:.0f}")
    L.append(f"equiv diam   median={s['equiv_diam']['median']:.1f}px  "
             f"p10={s['equiv_diam']['p10']:.1f} p90={s['equiv_diam']['p90']:.1f}")
    L.append(f"circularity  median={s['circularity']['median']:.2f}  (1=round, low=ragged)")
    L.append(f"solidity     median={s['solidity']['median']:.3f}  (1=convex, low=interlocking)")
    L.append(f"extent       median={s['extent']['median']:.3f}  (1=fills bbox = RECTANGULAR)")
    L.append(f"perim_ratio  median={s['perim_ratio']['median']:.3f}  (1=smooth, high=ragged edge)")
    L.append(f"elongation   median={s['elongation']['median']:.2f}  (1=equant, high=long)")
    return "\n".join(L) + "\n"
