# Green Map — Washington, DC

A procedurally generated mosaic "Green Map" of Washington, DC, **inspired by Ellen Harvey's
_Green Map_ (2019)** in the lobby of the Grand Hyatt at SFO. Harvey's piece inverts the usual
map (water becomes a shimmering field, parks are jewels, developed land recedes, a gold disc
marks "you are here") — but the specific thing this project takes from it is one principle:

> **different shape distributions for different colours** — each kind of region is rendered in
> its own family of hand-cut-looking tiles, not just its own colour.

This is an interpretation, not a reproduction. Two of the families are tuned to match shape
metrics *measured* from a photo of the real SFO mosaic; the rest is our own:

![the map](output/dc_greenmap.png)

- **grey land** — small rectangular tesserae — *shape metrics matched to the SFO photo*
- **green parks** — big interlocking glass shards — *shape metrics matched to the SFO photo*
- **blue water** — tiles elongated *along the river current* — **genuinely new**: Harvey's
  water is the silver shimmer field, not blue or flowing; this family is our own invention
- **gold disc** — the Capitol locator (a nod to Harvey's gold dot)

Colours are sampled per region from the SFO photo (then the water is recoloured blue — also a
departure). The District outline, rivers, and major green space are read from a real DC
satellite image, so the shape and geography are the real ones.

## Generate it

Requires `python3` with `numpy`, `scipy`, `Pillow`, `contourpy` (and `matplotlib` for the
analysis tools).

```bash
./generate.sh                 # render the PNG + export the interactive viewer data
# or individually:
python3 src/render_map.py 1600        # -> output/dc_greenmap.png  (arg = height in px)
python3 src/export_tiles.py           # -> web/dc_tiles.js  (for the viewer)
open web/viewer.html                  # interactive: grout, brightness, water shimmer, colours
```

## Repo layout

```
data/        inputs (committed so it's reproducible)
  dc_satellite.jpg   real DC satellite — source of the outline, rivers, green space
  sfo_greenmap.jpg   close-up of Harvey's SFO mosaic — source of all the metrics & colours
  color_model.json   per-region colour palettes sampled from the SFO photo
src/         THE GENERATOR (what's used)
  dc_build.py        engine: region map + 3 tile families + colour + shading
  render_map.py      -> output/dc_greenmap.png
  export_tiles.py    -> web/dc_tiles.js
web/         viewer.html (interactive), dc_tiles.js, gallery.html, sliders-demo.html
output/      dc_greenmap.png (the finished piece)
tools/       analysis & verification — how the numbers were derived/checked (see below)
```

## How it works (the method)

The interesting part is how the grey and green families were made: **rather than guessing what
"hand-cut tesserae" or "interlocking shards" should look like, measure them from the real
artwork and generate to those numbers.** (The water family is then designed by hand on top of
the same principle — see the table.)

1. **Segment** the real SFO photo into individual tiles (gradient-watershed; `tools/seg.py`).
2. **Measure** each tile with a shared yardstick (`tools/metrics.py`): area, size-variation
   (`area_cv`), `solidity` (interlocking), `circularity`/right-angle corners (rectangularity),
   colour. These targets live in `tools/sfo_target_metrics.json`.
3. **Generate** three tile families:

| family | model (`src/dc_build.py`) | matched to SFO? (real → gen) |
|---|---|---|
| grey land | deformed quad grid + cluster-merge | yes — right-angle corners 44% → 41%, area-cv 1.03 → 1.10 |
| green parks | warped variable-density Voronoi + merge | yes — solidity 0.79 → 0.73, area-cv 1.96 → 1.5 |
| green/grey size ratio | — | yes — 1.14 → 1.15 |
| **water** | flow-elongated cells along the current | **no — our own, designed by hand** |

4. **Grout** is rendered at the measured width (**8% of tile diameter**) in the measured
   colour (**warm grey, not black**). Land/park/grey colours are sampled per region from the
   SFO palettes; the water is then recoloured a hand-picked blue (a departure from Harvey).

Verify any time:

```bash
python3 tools/compare_metrics.py     # generated vs real, per family
python3 tools/rectangularity.py      # corner-angle distribution (the "squareness" metric)
python3 tools/area_ratio.py          # green/grey area ratio
```

## tools/ — analysis & the journey

Kept, runnable (run from the repo root):

- `seg.py` — mosaic segmentation (grout = gradient ridges, markers = flat tile interiors).
- `metrics.py` — per-tile shape metrics; the shared yardstick.
- `compare_metrics.py`, `rectangularity.py`, `area_ratio.py` — verify generated ≈ real.
- `extract_shapes.py` — vectorise the real tiles into polygons (`tile_shapes.json`).
- `color_model.py` — sample the real per-region palettes → `data/color_model.json`.
  (Run `extract_shapes.py` first; it produces the `tile_shapes.json` this reads.)

`tools/experiments/` — earlier approaches we built and learned from, kept as a record (paths
may point at the old layout). The story of these:

- **`generate.py` (Voronoi), `fracture.py`, `bricks.py`** — generative tile models. We learned
  Voronoi *cannot* reproduce the green shards (too convex, too uniform), which forced the
  fracture model. Bricks/flow-bricks explored rectangular backgrounds.
- **`tune.py`, `tune_green.py`, `fit_all.py`** — parameter searches fitting those models to
  the measured metrics.
- **`segment_real.py`, `discriminate_green.py`, `characterize_shapes.py`** — measuring the real
  tiles, finding which metrics actually separate the families, and a PCA shape model.
- **`compare_visual.py`, `strategies.py`** — side-by-side bake-offs of tiling strategies
  (transplanting real shapes vs generating them). The transplant route was the most faithful
  but couldn't tile seamlessly, which led to the current generate-to-metrics approach.
- **`grout.py`, `debug_seg.py`** — measuring grout width/colour and debugging segmentation.

Also in `web/`: `gallery.html` (a 7-technique mosaic explainer) and `sliders-demo.html` (an
earlier live-tweak DC generator) — standalone, no build step.

## Make a physical version

You can turn the map into a tactile, raised object. Two scripts produce the assets (their
outputs are git-ignored — regenerate any time):

```bash
python3 src/render_height.py 1600         # -> output/dc_greenmap_print.png (colour)
                                          #    output/dc_greenmap_height.png (16-bit height:
                                          #    white = raised tile, black = recessed grout)
python3 tools/make_coaster_mesh.py 100 1.0 3.5   # -> output/coaster/  watertight full-colour
                                                 #    relief mesh (.obj + .mtl + texture)
```

- **Textured UV / "elevated" flatbed print** (best for large, ~30 cm): send a print shop the
  colour image + the height map (the grayscale *is* the elevation channel). Ask specifically
  for *elevated/textured* UV (Canon Touchstone, Arizona PRISMAelevate, or Direct Color TEXTUR3D).
- **Full-colour 3D print** (best for small, fully order-online): upload the coaster mesh to
  craftcloud3d.com or i.materialise; pick a **full-colour** material (e.g. HP MJF Multicolor or
  full-colour sandstone). On upload, set the model's units to **millimetres**.

## Credit

Concept and the original mosaic: **Ellen Harvey, _Green Map_ (2019)**, Grand Hyatt at SFO,
commissioned by the San Francisco Arts Commission.
