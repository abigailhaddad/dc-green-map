"""Pre-render every land/green/water family combo for one state, plus an HTML browser.

    python3 src/gallery.py Michigan                 # all 343 combos -> PNG + index.html
    python3 src/gallery.py Maryland --height 520 --format webp

Builds the region ONCE (cached), computes the auto-scaled bases ONCE, then loops the
7 fill families across the three slots (7^3 = 343 images) and writes a self-contained
output/gallery/{state}/index.html with three dropdowns. The render helpers here are
reused by src/build_site.py to assemble a multi-state site.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import families as F
from make_state import scaled_base
from state_data import STATE_NAMES, build_region, resolve_state

ROOT = Path(__file__).resolve().parents[1]
FILLS = ["quads", "shards", "flow", "pebbles", "strata", "crackle", "rings"]  # disc = locator only
DEFAULTS = dict(land="quads", green="shards", water="flow")
COMBOS = list(product(FILLS, FILLS, FILLS))


def prep(ST, S):
    """Region + per-class bases + palettes for one state. Coarser than the S/130 'DC
    look': chunky tiles make each shape family read clearly at thumbnail size."""
    region = build_region(ST, S, capital=True)
    default_base = S / 55.0
    bases = {1: default_base,
             2: scaled_base(region == 2, default_base),
             3: scaled_base(region == 3, default_base)}
    cm = json.load(open(ROOT / "data" / "color_model.json"))
    palettes = {1: np.array(cm["gray"]), 2: np.array(cm["green"]), 3: np.array(cm["light"])}
    return region, bases, palettes


def _font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_combos(region, bases, palettes, out_dir, fmt="png", quality=70, seed=3,
                  progress=True, frames=None):
    """Render all 343 combos into out_dir as {land}_{green}_{water}.{fmt}. `frames` is an
    optional list of (x0,y0,x1,y1,label) boxes drawn on each image (the AK/HI insets)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(16) if frames else None
    for i, (land, green, water) in enumerate(COMBOS, 1):
        big, nbig, forced = F.compose(region, {1: land, 2: green, 3: water}, bases, seed=seed)
        lut, _ = F.color_lut(big, nbig, region, palettes, seed=seed)
        out = F.render_mosaic(big, nbig, region, lut, forced_grout=forced, gold_class=4)
        im = Image.fromarray((out * 255).astype(np.uint8)).convert("RGB")
        if frames:
            d = ImageDraw.Draw(im)
            for x0, y0, x1, y1, label in frames:
                d.rectangle([x0, y0, x1, y1], outline=(125, 122, 108), width=2)
                d.text((x0 + 2, y0 - 18), label, fill=(196, 190, 172), font=font)
        dest = out_dir / f"{land}_{green}_{water}.{fmt}"
        im.save(dest, quality=quality) if fmt in ("webp", "jpeg", "jpg") else im.save(dest)
        if progress and (i % 50 == 0 or i == len(COMBOS)):
            print(f"  {i}/{len(COMBOS)}")
    return len(COMBOS)


def _options():
    return "".join(f'<option value="{n}">{n} — {F.FAMILIES[n]["blurb"]}</option>' for n in FILLS)


def single_page(state_name, fmt):
    """Self-contained HTML for one state: three <select>s swap a pre-rendered <img>."""
    opts = _options()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{state_name} — mosaic family explorer</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #1a1610; color: #cfc9ba;
         font: 15px/1.4 system-ui, sans-serif; }}
  header {{ padding: 18px 22px 6px; }}
  h1 {{ margin: 0 0 2px; font-size: 20px; font-weight: 600; }}
  .sub {{ color: #8c8576; font-size: 13px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 14px 22px; }}
  .ctl {{ display: flex; flex-direction: column; gap: 4px; min-width: 230px; }}
  .ctl label {{ font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: #b6ad94; }}
  select {{ background: #2a241b; color: #e7e1d2; border: 1px solid #4a4232;
           border-radius: 6px; padding: 7px 9px; font-size: 14px; }}
  .stage {{ display: flex; justify-content: center; padding: 6px 22px 30px; }}
  img {{ max-width: 100%; max-height: 78vh; border-radius: 8px;
        box-shadow: 0 8px 30px #0008; background: #100c08; }}
  code {{ background: #2a241b; padding: 2px 6px; border-radius: 4px; color: #d8cfb6; }}
</style></head>
<body>
<header>
  <h1>{state_name} — mosaic family explorer</h1>
  <div class="sub">Pick a shape family for each region class. <span id="cli"></span></div>
</header>
<div class="controls">
  <div class="ctl"><label>Land</label><select id="land">{opts}</select></div>
  <div class="ctl"><label>Parks</label><select id="green">{opts}</select></div>
  <div class="ctl"><label>Water</label><select id="water">{opts}</select></div>
</div>
<div class="stage"><img id="map" alt="mosaic"></div>
<script>
  const D = {json.dumps(DEFAULTS)}, FMT = {json.dumps(fmt)}, NAME = {json.dumps(state_name)};
  for (const k of ["land","green","water"]) document.getElementById(k).value = D[k];
  function update() {{
    const l = land.value, g = green.value, w = water.value;
    map.src = `${{l}}_${{g}}_${{w}}.${{FMT}}`;
    map.alt = `land=${{l}} green=${{g}} water=${{w}}`;
    cli.innerHTML = `<code>python3 src/make_state.py ${{NAME}} --land ${{l}} --green ${{g}} --water ${{w}}</code>`;
  }}
  for (const k of ["land","green","water"]) document.getElementById(k).onchange = update;
  update();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Render every family combo + an HTML browser.")
    ap.add_argument("state", help="full name or 2-letter code")
    ap.add_argument("--height", type=int, default=620, help="thumbnail height in px")
    ap.add_argument("--format", choices=["png", "webp", "jpeg"], default="png")
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ST = resolve_state(args.state)
    name = STATE_NAMES[ST]
    print(f"building {name} ({ST}) region at height {args.height}px ...")
    region, bases, palettes = prep(ST, args.height)
    print(f"bases: land={bases[1]:.1f} green={bases[2]:.1f} water={bases[3]:.1f}")

    out_dir = ROOT / "output" / "gallery" / name.replace(" ", "_")
    print(f"rendering {len(COMBOS)} combos -> {out_dir.relative_to(ROOT)}/")
    render_combos(region, bases, palettes, out_dir, args.format, args.quality, args.seed)
    (out_dir / "index.html").write_text(single_page(name.title(), args.format))
    print(f"done -> open {(out_dir / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
