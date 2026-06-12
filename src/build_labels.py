"""Build hover-label overlay data for the site: per state, the capital marker plus the major
named parks (and later, named water). Stored as POINTS (x, y, radius, name) so the data stays
small enough to inline. Reuses cached GeoJSON — capital + parks need no fetch.

    python3 src/build_labels.py            # -> output/site/labels.json

Independent of the rendered images (pure geometry), so it runs alongside the render.
"""
import json
from collections import defaultdict
from math import cos, pi, radians
from pathlib import Path

import numpy as np
import us                                            # capital city names (no hand-typed dict)
from scipy import ndimage

import state_data as sd

ROOT = Path(__file__).resolve().parents[1]
# park designations worth labelling (skip the local "Dog Park" noise)
MAJOR_PARK = {"NP", "SP", "NF", "NWR", "NM", "WA", "WSA", "NRA", "NLS",
              "NCA", "SCA", "LCA", "SREC", "LREC"}
S = 620                                              # the gallery render height


def capital_city(ST):
    if ST == "DC":
        return "Washington"
    st = us.states.lookup(ST)
    return st.capital if st else None


def _projector(sfeat):
    """The same equirectangular projection build_region uses, as to_px + (W, H)."""
    norm, _ = sd._lon_norm(sfeat)
    bbox = sd._bbox(sfeat, norm)
    W, H, _ = sd._canvas(bbox, S)
    lonmin, latmin, lonmax, latmax = bbox
    dlon, dlat = lonmax - lonmin, latmax - latmin

    def to_px(lons, lats):
        return ((norm(np.asarray(lons, float)) - lonmin) / dlon * W,
                (latmax - np.asarray(lats, float)) / dlat * H)
    return to_px, W, H, bbox, norm


def park_points(ST, to_px, W, H):
    """Major named parks as {x, y, r, name} points (one per name, area-weighted),
    keeping only those big enough to see."""
    pp = sd.CACHE / f"{ST}_parks.json"
    if not pp.exists():
        return []
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])       # name -> [area, area*cx, area*cy]
    for f in json.load(open(pp))["features"]:
        props = f.get("properties", {})
        if props.get("Des_Tp") not in MAJOR_PARK or not f.get("geometry"):
            continue
        name = props.get("Unit_Nm") or ""
        for poly in sd._iter_polys(f["geometry"]):
            a = np.asarray(poly[0], float)
            x, y = to_px(a[:, 0], a[:, 1])
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))  # shoelace
            agg[name][0] += area
            agg[name][1] += area * x.mean(); agg[name][2] += area * y.mean()
    out = []
    for name, (area, sx, sy) in agg.items():
        if area < 0.0002 * W * H or not name:        # drop only the truly tiny / unnamed
            continue
        out.append({"x": int(round(sx / area)), "y": int(round(sy / area)),
                    "r": int(round(max(7, (area / pi) ** 0.5))), "name": name})
    out.sort(key=lambda p: -p["r"])
    return out[:70]                                  # cap labels per state


# TIGER abbreviates water names; expand the common ones for display
_ABBR = {"Crk": "Creek", "Riv": "River", "Lk": "Lake", "Resvr": "Reservoir",
         "Res": "Reservoir", "Br": "Branch", "Frk": "Fork", "Cnl": "Canal",
         "Bk": "Brook", "Spgs": "Springs", "Mt": "Mount", "Ck": "Creek", "Bay": "Bay"}
HYDRO_URL = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
             "Hydro/MapServer/1/query")


def _expand(name):
    return " ".join(_ABBR.get(w, w) for w in name.split())


def named_water_cache(ST, sfeat):
    """Named hydro for a state (NAME + centroid + area, no geometry), cached. Lightweight:
    server-side NAME-filter + returnGeometry=false, so payloads are small."""
    path = sd.CACHE / f"{ST}_water_named.json"
    if path.exists():
        return json.load(open(path))
    norm, crossing = sd._lon_norm(sfeat)
    lonmin, latmin, lonmax, latmax = sd._bbox(sfeat, norm)
    pad = 0.02
    if crossing:
        envs = [(lonmin - pad, latmin - pad, 180.0, latmax + pad),
                (-180.0, latmin - pad, lonmax - 360.0 + pad, latmax + pad)]
    else:
        envs = [(lonmin - pad, latmin - pad, lonmax + pad, latmax + pad)]
    print(f"    fetching named water for {ST} ...", flush=True)
    feats = []
    for x0, y0, x1, y1 in envs:
        gj = sd._post_paged(HYDRO_URL, dict(
            where="NAME IS NOT NULL", geometry=f"{x0},{y0},{x1},{y1}",
            geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outFields="NAME,CENTLAT,CENTLON,AREAWATER",
            returnGeometry="false", f="geojson"), page_size=100_000)
        feats.extend(gj["features"])
    json.dump({"features": feats}, open(path, "w"))
    return {"features": feats}


def water_points(ST, sfeat, to_px, W, H, bbox, state_mask):
    """Major named water bodies as {x, y, r, name} points. Only pieces whose centroid falls
    INSIDE the state are averaged, so a body that mostly lies outside (Lake Michigan is mostly
    north of Illinois) is labelled on the state's own slice, not the open lake."""
    data = named_water_cache(ST, sfeat)
    px_per_m = H / (max(bbox[3] - bbox[1], 1e-6) * 111_320.0)   # latitude scale -> px/metre
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])       # name -> [area, area*x, area*y]  (pixel space)
    for f in data["features"]:
        p = f.get("properties", {})
        name = (p.get("NAME") or "").strip()
        if not name:
            continue
        try:
            area = max(float(p.get("AREAWATER") or 0), 1.0)
            lon, lat = float(p["CENTLON"]), float(p["CENTLAT"])
        except (TypeError, ValueError):
            continue
        x, y = to_px(np.array([lon]), np.array([lat]))
        ix, iy = int(x[0]), int(y[0])
        if not (0 <= ix < W and 0 <= iy < H) or not state_mask[iy, ix]:   # keep only in-state pieces
            continue
        agg[name][0] += area; agg[name][1] += area * float(x[0]); agg[name][2] += area * float(y[0])
    snap = ndimage.distance_transform_edt(~state_mask, return_distances=False, return_indices=True)
    out = []
    for name, (area, sx, sy) in agg.items():
        if area < 5e5:                               # skip < 0.5 km² named water (noise)
            continue
        ix, iy = int(round(sx / area)), int(round(sy / area))
        if not state_mask[iy, ix]:                   # nudge an off-state centroid onto the state
            iy, ix = int(snap[0][iy, ix]), int(snap[1][iy, ix])
        r = max(7, min(34, (area / pi) ** 0.5 * px_per_m))
        out.append({"x": ix, "y": iy, "r": int(round(r)), "name": _expand(name)})
    out.sort(key=lambda p: -p["r"])
    return out[:55]


def main():
    site = ROOT / "output" / "site"
    folders = sorted(p.name for p in site.iterdir() if p.is_dir() and p.name != "usa")
    labels = {}
    for i, folder in enumerate(folders, 1):
        ST = sd.NAME_TO_ST.get(folder.replace("_", " "))
        sp = sd.CACHE / f"{ST}_state.json" if ST else None
        if not ST or not sp.exists():
            continue
        sfeat = [f for f in json.load(open(sp))["features"] if f.get("geometry")]
        to_px, W, H, bbox, norm = _projector(sfeat)
        state_mask = sd._rasterize(sfeat, bbox, S, norm)
        d = {"W": int(W), "H": int(H), "parks": park_points(ST, to_px, W, H),
             "water": water_points(ST, sfeat, to_px, W, H, bbox, state_mask)}
        clat, clon = sd.CAPITALS.get(ST, (None, None))
        city = capital_city(ST)
        if clat is not None and city:
            cx, cy = to_px(np.array([clon]), np.array([clat]))
            if ST == "DC":                           # the DC dot sits on the U.S. Capitol itself
                label = "⚑ U.S. Capitol — Washington, D.C."
            else:
                label = f"⚑ {city} — {sd.STATE_NAMES[ST].title()} state capital"
            d["capital"] = {"x": int(round(cx[0])), "y": int(round(cy[0])), "label": label}
        labels[folder] = d
        print(f"  [{i}/{len(folders)}] {folder}: {city}, "
              f"{len(d['parks'])} parks, {len(d['water'])} water")
    json.dump(labels, open(site / "labels.json", "w"))
    print(f"wrote {(site / 'labels.json').relative_to(ROOT)} ({len(labels)} states)")


if __name__ == "__main__":
    main()
