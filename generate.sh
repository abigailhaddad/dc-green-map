#!/usr/bin/env bash
# Generate the DC Green Map: high-res PNG + interactive viewer data.
# Usage: ./generate.sh [height_px]   (default 1600)
set -e
cd "$(dirname "$0")"
H="${1:-1600}"
python3 src/render_map.py "$H"
python3 src/export_tiles.py
echo "done -> output/dc_greenmap.png  +  web/dc_tiles.js  (open web/viewer.html)"
