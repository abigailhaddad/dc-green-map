"""Render several states' family galleries into ONE browsable site.

    python3 src/build_site.py MI MD FL CO            # a 4-state test batch
    python3 src/build_site.py --all                  # all 50 states + DC (~2 hrs, ~0.4 GB webp)
    python3 src/build_site.py MI MD --height 620 --format webp

Output: output/site/{state}/{land}_{green}_{water}.{fmt} plus output/site/index.html with
four dropdowns (State / Land / Parks / Water). The page is static — open index.html directly
or upload output/site/ to any static host (the images dominate the size, so prefer webp).
"""
import argparse
import json
from pathlib import Path

import yaml

from gallery import COMBOS, DEFAULTS, FILLS, prep, render_combos
import families as F
from state_data import STATE_NAMES, resolve_state

ROOT = Path(__file__).resolve().parents[1]
TEXT = yaml.safe_load(open(ROOT / "data" / "site_text.yaml"))   # editable site copy
FOLDER_TO_NAME = {name.replace(" ", "_"): name.title() for name in STATE_NAMES.values()}
FOLDER_TO_NAME["usa"] = "United States"


def write_index(site_dir, fmt):
    """(Re)write site_dir/index.html from whatever gallery folders exist on disk, so the
    state dropdown always reflects every rendered gallery (per-state + the national 'usa').
    Inlines usa/overlay.json (if present) so the national map's clickable state shapes work
    even over file:// (no fetch)."""
    folders = sorted(p.name for p in site_dir.iterdir()
                     if p.is_dir() and (p / f"quads_shards_flow.{fmt}").exists())
    folders.sort(key=lambda f: (f != "usa", FOLDER_TO_NAME.get(f, f)))  # United States first
    states = [(f, FOLDER_TO_NAME.get(f, f.replace("_", " ").title())) for f in folders]
    ov = site_dir / "usa" / "overlay.json"
    overlay = ov.read_text() if ov.exists() else "null"
    lb = site_dir / "labels.json"
    labels = lb.read_text() if lb.exists() else "null"
    (site_dir / "index.html").write_text(
        site_page(states, overlay, labels).replace("{FMT}", fmt))
    return states


def site_page(states, overlay="null", labels="null"):
    """states: list of (folder, display) in display order. overlay: JSON of the national
    clickable-state shapes. labels: JSON of per-state hover labels (capital marker).
    Four-dropdown browser; the national map loads first and its states are click-to-drill-in."""
    opts = "".join(f'<option value="{n}">{n} — {F.FAMILIES[n]["blurb"]}</option>' for n in FILLS)
    st_opts = "".join(f'<option value="{folder}">{disp}</option>' for folder, disp in states)
    names = {folder: disp for folder, disp in states}
    story_html = "".join(f"<p>{p.strip()}</p>" for p in TEXT["story"].split("\n") if p.strip())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TEXT['title']} — mosaic map explorer</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #1a1610; color: #cfc9ba;
         font: 15px/1.4 system-ui, sans-serif; }}
  header {{ padding: 20px 22px 4px; max-width: 900px; }}
  h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 600; letter-spacing: -.01em; }}
  .intro {{ color: #a59c87; font-size: 13.5px; margin: 0 0 6px; }}
  .sub {{ color: #8c8576; font-size: 12.5px; }}
  a {{ color: #c8a86a; }}
  .links {{ margin: 8px 0 2px; font-size: 14px; }}
  .links a {{ font-weight: 600; }}
  .links .sep {{ color: #5c5546; margin: 0 8px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 14px 22px; }}
  .ctl {{ display: flex; flex-direction: column; gap: 4px; min-width: 210px; }}
  .ctl label {{ font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: #b6ad94; }}
  select {{ background: #2a241b; color: #e7e1d2; border: 1px solid #4a4232;
           border-radius: 6px; padding: 7px 9px; font-size: 14px; }}
  .stage {{ display: flex; flex-direction: column; align-items: center; padding: 6px 22px 24px; }}
  #frame {{ position: relative; display: inline-block; line-height: 0; }}
  #map {{ max-width: 100%; max-height: 76vh; border-radius: 8px; display: block;
        box-shadow: 0 8px 30px #0008; background: #100c08; }}
  #overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  #overlay path {{ fill: transparent; stroke: transparent; stroke-width: 1; cursor: pointer; }}
  #overlay path:hover {{ fill: rgba(255,236,196,.16); stroke: rgba(255,224,170,.85);
        stroke-width: 1.4; }}
  #overlay circle {{ fill: transparent; cursor: pointer; }}
  #overlay circle.cap:hover {{ fill: rgba(255,224,170,.22); stroke: rgba(255,224,170,.85);
        stroke-width: 1; }}
  #overlay circle.park:hover {{ fill: rgba(150,210,140,.20); stroke: rgba(175,225,155,.8);
        stroke-width: 1; }}
  #overlay circle.water:hover {{ fill: rgba(120,165,225,.22); stroke: rgba(150,190,235,.85);
        stroke-width: 1; }}
  #hint {{ color: #8c8576; font-size: 12.5px; margin: 10px 0 0; height: 1.2em; }}
  #back {{ background: none; border: 1px solid #4a4232; color: #c8a86a; cursor: pointer;
        border-radius: 6px; padding: 5px 11px; font-size: 13px; display: none; }}
  code {{ background: #2a241b; padding: 2px 6px; border-radius: 4px; color: #d8cfb6; }}
  .story {{ max-width: 700px; margin: 0 auto; padding: 12px 22px 56px;
           color: #b3aa95; font-size: 14.5px; line-height: 1.62;
           border-top: 1px solid #2c2619; }}
  .story h2 {{ font-size: 16px; color: #d8cfb6; margin: 22px 0 10px; font-weight: 600; }}
  .story p {{ margin: 0 0 13px; }}
</style></head>
<body>
<header>
  <h1>{TEXT['title']}</h1>
  <p class="intro">{TEXT['intro']}</p>
  <div class="sub">{TEXT['sub']}</div>
  <div class="links"><a href="{TEXT['blog_url']}" target="_blank" rel="noopener">My blog</a><span class="sep">·</span><a href="{TEXT['repo_url']}" target="_blank" rel="noopener">Source on GitHub</a></div>
</header>
<div class="controls">
  <div class="ctl"><label>State</label><select id="state">{st_opts}</select></div>
  <div class="ctl"><label>Land</label><select id="land">{opts}</select></div>
  <div class="ctl"><label>Parks</label><select id="green">{opts}</select></div>
  <div class="ctl"><label>Water</label><select id="water">{opts}</select></div>
</div>
<div class="stage">
  <div id="frame">
    <img id="map" alt="mosaic">
    <svg id="overlay" preserveAspectRatio="none"></svg>
  </div>
  <p id="hint"></p>
  <button id="back">&larr; Back to the U.S. map</button>
</div>
<section class="story">{story_html}</section>
<script>
  const D = {json.dumps(DEFAULTS)}, NAMES = {json.dumps(names)};
  const OVERLAY = {overlay}, LABELS = {labels};
  for (const k of ["land","green","water"]) document.getElementById(k).value = D[k];

  function buildUSOverlay() {{                            // clickable state shapes over the US map
    overlay.setAttribute("viewBox", `0 0 ${{OVERLAY.W}} ${{OVERLAY.H}}`);
    overlay.innerHTML = Object.entries(OVERLAY.states).map(([folder, st]) =>
      `<path d="${{st.path}}" data-folder="${{folder}}"><title>${{st.name}}</title></path>`).join("");
    overlay.querySelectorAll("path").forEach(p => {{
      p.onclick = () => {{ state.value = p.dataset.folder; onChange(); }};
      p.onmouseenter = () => {{ hint.textContent = NAMES[p.dataset.folder] || ""; }};
      p.onmouseleave = () => {{ hint.textContent = "Click a state to explore it"; }};
    }});
  }}
  function buildStateLabels(s) {{                         // hover markers: capital + major parks
    const L = LABELS && LABELS[s];
    if (!L) {{ overlay.style.display = "none"; return; }}
    overlay.setAttribute("viewBox", `0 0 ${{L.W}} ${{L.H}}`);
    const markers = [];
    (L.water || []).forEach(p => markers.push({{x: p.x, y: p.y, r: p.r, cls: "water", label: p.name}}));
    (L.parks || []).forEach(p => markers.push({{x: p.x, y: p.y, r: p.r, cls: "park", label: p.name}}));
    if (L.capital) {{
      const cr = Math.max(11, L.H * 0.03);
      markers.push({{x: L.capital.x, y: L.capital.y, r: cr, cls: "cap", label: L.capital.label}});
    }}
    overlay.innerHTML = markers.map((m, i) =>
      `<circle class="${{m.cls}}" cx="${{m.x}}" cy="${{m.y}}" r="${{m.r}}" data-i="${{i}}"></circle>`).join("");
    overlay.querySelectorAll("circle").forEach(el => {{
      el.onmouseenter = () => {{ hint.textContent = markers[el.dataset.i].label; }};
      el.onmouseleave = () => {{ hint.textContent = ""; }};
    }});
    overlay.style.display = markers.length ? "block" : "none";
  }}
  function update() {{
    const s = state.value, l = land.value, g = green.value, w = water.value;
    map.src = `${{s}}/${{l}}_${{g}}_${{w}}.{{FMT}}`;
    map.alt = `${{NAMES[s]}}: land=${{l}} green=${{g}} water=${{w}}`;
    const isUS = s === "usa";
    back.style.display = isUS ? "none" : "inline-block";
    if (isUS && OVERLAY) {{ buildUSOverlay(); overlay.style.display = "block";
      hint.textContent = "Click a state to explore it"; }}
    else {{ buildStateLabels(s); hint.textContent = ""; }}
  }}
  function writeUrl() {{                                  // only on user action; default view stays clean
    const s = state.value, l = land.value, g = green.value, w = water.value;
    const isDefault = s === "usa" && l === D.land && g === D.green && w === D.water;
    const u = isDefault ? location.pathname
      : "#" + new URLSearchParams({{state: s, land: l, parks: g, water: w}});
    try {{ history.replaceState(null, "", u); }} catch (e) {{}}
  }}
  function onChange() {{ update(); writeUrl(); }}
  const setSel = (el, v) => {{ if (v && [...el.options].some(o => o.value === v)) el.value = v; }};
  for (const k of ["state","land","green","water"]) document.getElementById(k).onchange = onChange;
  back.onclick = () => {{ state.value = "usa"; onChange(); }};
  if ([...state.options].some(o => o.value === "usa")) state.value = "usa";  // default: US map
  const q0 = new URLSearchParams(location.hash.slice(1));                     // restore from URL
  setSel(state, q0.get("state")); setSel(land, q0.get("land"));
  setSel(green, q0.get("parks")); setSel(water, q0.get("water"));
  update();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Render a multi-state mosaic site.")
    ap.add_argument("states", nargs="*", help="state names / 2-letter codes")
    ap.add_argument("--all", action="store_true", help="all 50 states + DC")
    ap.add_argument("--height", type=int, default=620)
    ap.add_argument("--format", choices=["png", "webp", "jpeg"], default="webp")
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    codes = sorted(STATE_NAMES) if args.all else [resolve_state(s) for s in args.states]
    if not codes:
        ap.error("pass state codes/names, or --all")

    site = ROOT / "output" / "site"
    site.mkdir(parents=True, exist_ok=True)
    done = []
    for k, ST in enumerate(codes, 1):
        name = STATE_NAMES[ST]
        folder = name.replace(" ", "_")
        print(f"[{k}/{len(codes)}] {name} ({ST}) ...")
        try:
            region, bases, palettes = prep(ST, args.height)
            print(f"  bases: land={bases[1]:.1f} green={bases[2]:.1f} water={bases[3]:.1f}")
            render_combos(region, bases, palettes, site / folder,
                          args.format, args.quality, args.seed, progress=False)
            done.append((folder, name.title()))
        except Exception as e:                       # noqa: BLE001 — skip a bad state, keep going
            print(f"  SKIPPED {name}: {e}")

    page = site_page(done).replace("{FMT}", args.format)
    (site / "index.html").write_text(page)
    print(f"\ndone: {len(done)}/{len(codes)} states -> open {(site / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
