"""Render a mosaic 'Green Map' of any US state, in the style of the DC piece.

    python3 src/make_state.py Maryland
    python3 src/make_state.py Michigan --water pebbles --green crackle --height 2000
    python3 src/make_state.py --list                  # the shape families

The user picks which shape family fills each region class (--land/--green/--water);
tile sizes auto-scale to the geography so narrow rivers stay recognizable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import families as F
from state_data import STATE_NAMES, build_region, resolve_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(land="quads", green="shards", water="flow")


def scaled_base(mask, default_base):
    """Tile size scaled to the region's TYPICAL width, not its thinnest threads.

    Principle: tiles small enough to follow narrow features, keyed off the region's MEDIAN
    half-width (not its thinnest threads, which would clamp everything tiny). Since grout is
    now a constant FRACTION of each tile (see families.render_mosaic), small tiles keep their
    colour instead of dissolving into grout — so the floor can be low, letting narrow rivers
    and small parks actually render."""
    if not mask.any():
        return default_base
    ed = ndimage.distance_transform_edt(mask)
    vals = ed[mask & (ed > 0)]
    if vals.size == 0:
        return default_base
    half = np.percentile(vals, 50)                   # median half-width of the region
    floor = max(4.5, 0.4 * default_base)             # low floor is safe with proportional grout
    return float(np.clip(1.5 * half, floor, default_base))


def main():
    ap = argparse.ArgumentParser(description="Mosaic green map of any US state.")
    ap.add_argument("state", nargs="?", help="full name or 2-letter code (case-insensitive)")
    ap.add_argument("--height", type=int, default=1600, help="output height in px")
    ap.add_argument("--land", choices=list(F.FAMILIES), default=DEFAULTS["land"])
    ap.add_argument("--green", choices=list(F.FAMILIES), default=DEFAULTS["green"])
    ap.add_argument("--water", choices=list(F.FAMILIES), default=DEFAULTS["water"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-capital", action="store_true", help="omit the gold capital disc")
    ap.add_argument("--list", action="store_true", help="list the shape families and exit")
    args = ap.parse_args()

    if args.list:
        for name, d in F.FAMILIES.items():
            print(f"  {name:8s} {d['blurb']}")
        return
    if not args.state:
        ap.error("a state is required (or use --list)")

    ST = resolve_state(args.state)
    S = args.height
    print(f"building {STATE_NAMES[ST]} ({ST}) at height {S}px")
    region = build_region(ST, S, capital=not args.no_capital)

    # auto-scale: land keeps the DC look; water/green shrink to fit narrow geometry
    default_base = S / 130.0
    bases = {1: default_base,
             2: scaled_base(region == 2, default_base),
             3: scaled_base(region == 3, default_base)}
    floor = max(7.0, 0.55 * default_base)
    print(f"bases: land={bases[1]:.1f}px  green={bases[2]:.1f}px  water={bases[3]:.1f}px")
    if bases[3] <= floor + 0.01 or bases[2] <= floor + 0.01:
        print("note: tiles are at the size floor — narrow features will read coarse; "
              "consider --height 2400 for more detail")

    assignment = {1: args.land, 2: args.green, 3: args.water}
    big, nbig, forced = F.compose(region, assignment, bases, seed=args.seed)

    cm = json.load(open(ROOT / "data" / "color_model.json"))
    palettes = {1: np.array(cm["gray"]), 2: np.array(cm["green"]), 3: np.array(cm["light"])}
    lut, _ = F.color_lut(big, nbig, region, palettes, seed=args.seed)
    out = F.render_mosaic(big, nbig, region, lut, forced_grout=forced, gold_class=4)

    dest = ROOT / "output" / f"{STATE_NAMES[ST].replace(' ', '_')}_greenmap.png"
    dest.parent.mkdir(exist_ok=True)
    Image.fromarray((out * 255).astype(np.uint8)).save(dest)
    print(f"{nbig} tiles  canvas {out.shape[1]}x{out.shape[0]}  ->  {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
