"""Aggregate every state into ONE national mosaic, then render all 343 family combos for it
(adds a 'United States' entry to the site's state dropdown).

    python3 src/build_national.py                    # Albers CONUS + AK/HI insets, height 900
    python3 src/build_national.py --height 1100 --no-insets

Consistent look: all states composite into a SINGLE canvas and the families fill the whole
national mask in ONE pass with ONE tile size — every tesser is the same size everywhere
(small states just get fewer tiles). The contiguous US is drawn in an Albers Equal-Area
projection (the USGS standard — avoids the ~20% north-south stretch of a plain lat/lon grid);
Alaska and Hawaii are scaled insets in the bottom-left. Reuses each state's cached GeoJSON.
"""
import argparse
import json
from math import cos, radians, sin, sqrt
from pathlib import Path

import numpy as np

import families as F
import state_data as sd
from build_site import write_index
from gallery import COMBOS, render_combos
from make_state import scaled_base

ROOT = Path(__file__).resolve().parents[1]
CONUS = [c for c in sorted(sd.STATE_NAMES) if c not in ("AK", "HI")]   # lower 48 + DC


# ---------------------------------------------------------------------------
# projections -> pixel mappers, and panels (a set of states placed in the canvas)
# ---------------------------------------------------------------------------
def _albers(lon0=-96.0, lat0=23.0, lat1=29.5, lat2=45.5):
    """Albers Equal-Area Conic (CONUS standard parallels) -> planar (X, Y), unit sphere."""
    l0, l1, l2 = radians(lat0), radians(lat1), radians(lat2)
    n = (sin(l1) + sin(l2)) / 2
    C = cos(l1) ** 2 + 2 * n * sin(l1)
    rho0 = sqrt(C - 2 * n * sin(l0)) / n
    lon0r = radians(lon0)

    def proj(lons, lats):
        lonr = np.radians(np.asarray(lons, float)); latr = np.radians(np.asarray(lats, float))
        theta = n * (lonr - lon0r)
        rho = np.sqrt(np.maximum(C - 2 * n * np.sin(latr), 0.0)) / n
        return rho * np.sin(theta), rho0 - rho * np.cos(theta)

    return proj


def _feats_of(codes):
    """Load cached outline features per state (skips states with no cache)."""
    feats, allsf = {}, []
    for ST in codes:
        sp = sd.CACHE / f"{ST}_state.json"
        if not sp.exists():
            continue
        sf = [f for f in json.load(open(sp))["features"] if f.get("geometry")]
        if sf:
            feats[ST] = sf; allsf += sf
    return feats, allsf


def _build_panel(codes, S, mode, clip_bbox=None):
    """A panel: states `codes` rendered to a (W, H) sub-image via `to_px`. mode='albers'
    (CONUS) or 'equirect' (insets, with antimeridian handling for Alaska). clip_bbox
    (lonmin,latmin,lonmax,latmax) overrides the extent — used to crop Hawaii to its main
    islands (its true bbox reaches Midway, which would shrink the inhabited isles to specks)."""
    feats, allsf = _feats_of(codes)
    if not feats:
        return None
    if mode == "albers":
        proj = _albers()
        xs, ys = [], []
        for f in allsf:
            for poly in sd._iter_polys(f.get("geometry")):
                for ring in poly:
                    a = np.asarray(ring, float); X, Y = proj(a[:, 0], a[:, 1])
                    xs.append(X); ys.append(Y)
        X = np.concatenate(xs); Y = np.concatenate(ys)
        x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
        scale = S / (y1 - y0); W = max(1, int(round((x1 - x0) * scale)))

        def to_px(lons, lats):
            Xp, Yp = proj(lons, lats)
            return (Xp - x0) * scale, (y1 - Yp) * scale
        H = S
    else:                                                # equirectangular (insets)
        norm, _ = sd._lon_norm(allsf)
        bbox = clip_bbox if clip_bbox else sd._bbox(allsf, norm)
        W, H, _ = sd._canvas(bbox, S)
        lonmin, latmin, lonmax, latmax = bbox
        dlon, dlat = lonmax - lonmin, latmax - latmin

        def to_px(lons, lats):
            return ((norm(np.asarray(lons, float)) - lonmin) / dlon * W,
                    (latmax - np.asarray(lats, float)) / dlat * H)
    return dict(codes=list(feats), feats=feats, to_px=to_px, W=W, H=H)


def _panel_masks(panel):
    """Rasterize a panel's land / water / park masks at its own resolution."""
    W, H, to_px = panel["W"], panel["H"], panel["to_px"]
    state = np.zeros((H, W), bool); water = state.copy(); parks = state.copy()
    for ST in panel["codes"]:
        state |= sd.rasterize_px(panel["feats"][ST], to_px, W, H)
        wp, pp = sd.CACHE / f"{ST}_water.json", sd.CACHE / f"{ST}_parks.json"
        if wp.exists():
            water |= sd.rasterize_px(sd._keep_water(json.load(open(wp))["features"]), to_px, W, H)
        if pp.exists():
            parks |= sd.rasterize_px(json.load(open(pp))["features"], to_px, W, H)
    return state, water & state, parks & state


def _paste(dst, src, ox, oy):
    """OR a panel mask into the final canvas at (ox, oy), clipping to bounds."""
    H, W = dst.shape; h, w = src.shape
    x0, y0 = max(0, ox), max(0, oy); x1, y1 = min(W, ox + w), min(H, oy + h)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] |= src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]


def national_panels(S, insets=True):
    """The placed panels (CONUS Albers + AK/HI inset equirect) and the final canvas size.
    Cheap — builds projections only, no rasterization (used for the clickable/label overlay)."""
    conus = _build_panel(CONUS, S, "albers")
    if conus is None:
        raise RuntimeError("no cached state geometry — run build_site first")
    Wn, panels = conus["W"], [(conus, 0, 0)]
    Hn = S
    if insets:                                       # AK + HI tucked just under the map (their own
        ak = _build_panel(["AK"], int(0.28 * S), "equirect")          # band, no mainland overlap)
        hi = _build_panel(["HI"], int(0.20 * S), "equirect",
                          clip_bbox=(-160.6, 18.7, -154.7, 22.5))     # main islands only
        top = int(0.03 * S)                          # small gap under the map + room for the label
        band = max((ak["H"] if ak else 0), (hi["H"] if hi else 0)) + top + int(0.015 * S)
        Hn = S + band
        x = int(0.04 * Wn)
        if ak:
            panels.append((ak, x, S + top)); x += ak["W"] + int(0.05 * Wn)
        if hi:
            panels.append((hi, x, S + top))
    return panels, Wn, Hn


def inset_frames(panels, pad=4):
    """Box rects (x0,y0,x1,y1,label) around the AK/HI insets — drawn on each render so the
    insets read as framed boxes, not part of the mainland."""
    names = {"AK": "Alaska", "HI": "Hawaii"}
    out = []
    for panel, ox, oy in panels:
        if len(panel["codes"]) == 1 and panel["codes"][0] in names:
            out.append((ox - pad, oy - pad, ox + panel["W"] + pad, oy + panel["H"] + pad,
                        names[panel["codes"][0]]))
    return out


def national_region(S, capital_r=0.006, insets=True):
    """Composite CONUS (Albers) + optional AK/HI insets into one region map + a ctx with the
    panels (their to_px and offsets) so the clickable overlay can be projected to match."""
    panels, Wf, Hf = national_panels(S, insets)
    state = np.zeros((Hf, Wf), bool); water = state.copy(); parks = state.copy()
    for panel, ox, oy in panels:
        ps, pw, pp = _panel_masks(panel)
        _paste(state, ps, ox, oy); _paste(water, pw, ox, oy); _paste(parks, pp, ox, oy)
    water &= state; parks &= state

    # tighter despeckle than a single state: at ~3 km/px every 1-px river thread over-reads,
    # so drop short/minor blobs (major rivers and the Great Lakes are long/large -> survive)
    base = S / 55.0
    parks = sd._despeckle(parks, (0.7 * base) ** 2)
    water = sd._despeckle(water, (0.9 * base) ** 2)
    region = state.astype(int)
    region[parks] = 2
    region[water] = 3                                    # water last: rivers cut through parks

    if capital_r:
        yy, xx = np.mgrid[0:Hf, 0:Wf]; rad = capital_r * S
        for panel, ox, oy in panels:
            for ST in panel["codes"]:
                if ST not in sd.CAPITALS:
                    continue
                clat, clon = sd.CAPITALS[ST]
                x, y = panel["to_px"](np.array([clon]), np.array([clat]))
                cx, cy = ox + float(x[0]), oy + float(y[0])
                region[((xx - cx) ** 2 + (yy - cy) ** 2 < rad ** 2) & (region > 0)] = 4
    print(f"  national canvas {Wf}x{Hf}, {len(panels)} panels "
          f"({sum(len(p['codes']) for p, _, _ in panels)} states)")
    return region, dict(W=Wf, H=Hf, panels=panels)


# ---------------------------------------------------------------------------
# clickable overlay (state outlines in final-canvas px, matching the panels)
# ---------------------------------------------------------------------------
def _svg_path(feats, to_px, ox, oy, W, H):
    subs = []
    for f in feats:
        for poly in sd._iter_polys(f.get("geometry")):
            a = np.asarray(poly[0], float)                # outer ring = hit area
            x, y = to_px(a[:, 0], a[:, 1])
            if x.max() < 0 or x.min() > W or y.max() < 0 or y.min() > H:   # clip off-panel
                continue                                  # (e.g. Hawaii's remote NW islands)
            pts = np.column_stack([x + ox, y + oy]).round().astype(int)
            keep = np.ones(len(pts), bool)
            keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
            pts = pts[keep]
            if len(pts) >= 3:
                subs.append("M" + " L".join(f"{px} {py}" for px, py in pts) + " Z")
    return "".join(subs)


def build_overlay(ctx):
    """{W, H, states:{folder:{name, path}}} — clickable state shapes over the national map."""
    states = {}
    for panel, ox, oy in ctx["panels"]:
        for ST in panel["codes"]:
            name = sd.STATE_NAMES[ST]
            states[name.replace(" ", "_")] = {
                "name": name.title(),
                "path": _svg_path(panel["feats"][ST], panel["to_px"], ox, oy,
                                  panel["W"], panel["H"])}
    return {"W": ctx["W"], "H": ctx["H"], "states": states}


def main():
    ap = argparse.ArgumentParser(description="Render the whole-US aggregate mosaic gallery.")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--format", choices=["png", "webp", "jpeg"], default="webp")
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--no-capitals", action="store_true")
    ap.add_argument("--no-insets", action="store_true", help="lower 48 only (skip AK/HI)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    print(f"building national region at height {args.height}px ...")
    region, ctx = national_region(args.height, 0.0 if args.no_capitals else 0.006,
                                  insets=not args.no_insets)

    default_base = args.height / 55.0
    bases = {1: default_base,
             2: scaled_base(region == 2, default_base),
             3: scaled_base(region == 3, default_base)}
    print(f"  bases: land={bases[1]:.1f} green={bases[2]:.1f} water={bases[3]:.1f}")
    cm = json.load(open(ROOT / "data" / "color_model.json"))
    palettes = {1: np.array(cm["gray"]), 2: np.array(cm["green"]), 3: np.array(cm["light"])}

    site = ROOT / "output" / "site"
    (site / "usa").mkdir(parents=True, exist_ok=True)
    json.dump(build_overlay(ctx), open(site / "usa" / "overlay.json", "w"))  # clickable states
    frames = inset_frames(ctx["panels"])
    print(f"rendering {len(COMBOS)} combos -> output/site/usa/  (inset frames: {len(frames)})")
    render_combos(region, bases, palettes, site / "usa", args.format, args.quality, args.seed,
                  progress=True, frames=frames)
    states = write_index(site, args.format)
    print(f"done -> 'United States' added; site now lists {len(states)} galleries")


if __name__ == "__main__":
    main()
