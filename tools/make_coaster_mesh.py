"""Build a full-colour 3D-printable relief from the aligned colour + height pair:
a watertight OBJ (+MTL +texture) where the tiles stand proud of the grout and the colour is
the rendered map. Upload to Craftcloud / i.materialise / Sculpteo for a full-colour print.

Run `python src/render_height.py` first to make the aligned pair, then this.
Usage:  python tools/make_coaster_mesh.py [long_mm] [relief_mm] [base_mm]   (default 100 1.0 3.5)
"""
import sys, shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COAST = OUT / "coaster"; COAST.mkdir(exist_ok=True)

long_mm  = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0   # coaster ≈ 10 cm
relief_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     # tile-vs-grout height
base_mm  = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5      # solid slab under it

H16 = np.asarray(Image.open(OUT / "dc_greenmap_height.png")).astype(float) / 65535.0
Himg, Wimg = H16.shape
LONG = 320
if Himg >= Wimg:
    gy, gx = LONG, max(2, round(LONG * Wimg / Himg))
    Hmm, Wmm = long_mm, long_mm * Wimg / Himg
else:
    gx, gy = LONG, max(2, round(LONG * Himg / Wimg))
    Wmm, Hmm = long_mm, long_mm * Himg / Wimg

ii = np.linspace(0, Himg - 1, gy).round().astype(int)
jj = np.linspace(0, Wimg - 1, gx).round().astype(int)
Hg = H16[np.ix_(ii, jj)]                                   # height on the mesh grid

dx, dy = Wmm / (gx - 1), Hmm / (gy - 1)
# vertex coords: x right, y up (flip image rows), z up; top domed, bottom flat
J, I = np.meshgrid(np.arange(gx), np.arange(gy))
X = J * dx
Y = (gy - 1 - I) * dy
Ztop = base_mm + relief_mm * Hg
top = np.stack([X, Y, Ztop], -1).reshape(-1, 3)
bot = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
# UVs for the top grid sample the full-res colour texture (crisp tiles)
U = jj[J] / (Wimg - 1)
V = 1 - ii[I] / (Himg - 1)
uv = np.stack([U, V], -1).reshape(-1, 2)

N = gy * gx
def tid(i, j): return i * gx + j + 1                       # top vertex id (1-based)
def bid(i, j): return N + i * gx + j + 1                   # bottom vertex id
VT_DARK = N + 1                                            # one extra uv (corner) for hidden faces

lines = ["mtllib dc_coaster.mtl", "usemtl map", "o dc_coaster"]
lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in top]
lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in bot]
lines += [f"vt {u:.5f} {w:.5f}" for u, w in uv]
lines.append("vt 0 0")
F = []
for i in range(gy - 1):
    for j in range(gx - 1):
        a, b, c, d = tid(i, j), tid(i, j+1), tid(i+1, j+1), tid(i+1, j)   # top, normal +z
        F.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        F.append(f"f {a}/{a} {c}/{c} {d}/{d}")
        a2, b2, c2, d2 = bid(i, j), bid(i, j+1), bid(i+1, j+1), bid(i+1, j)  # bottom, normal -z
        F.append(f"f {a2}/{VT_DARK} {c2}/{VT_DARK} {b2}/{VT_DARK}")
        F.append(f"f {a2}/{VT_DARK} {d2}/{VT_DARK} {c2}/{VT_DARK}")
# side walls (connect top edge to bottom edge) so it's a closed solid
def wall(t1, t2, b1, b2):
    F.append(f"f {t1}/{VT_DARK} {b1}/{VT_DARK} {b2}/{VT_DARK}")
    F.append(f"f {t1}/{VT_DARK} {b2}/{VT_DARK} {t2}/{VT_DARK}")
for j in range(gx - 1):
    wall(tid(0, j), tid(0, j+1), bid(0, j), bid(0, j+1))                  # top edge
    wall(tid(gy-1, j+1), tid(gy-1, j), bid(gy-1, j+1), bid(gy-1, j))      # bottom edge
for i in range(gy - 1):
    wall(tid(i+1, 0), tid(i, 0), bid(i+1, 0), bid(i, 0))                  # left edge
    wall(tid(i, gx-1), tid(i+1, gx-1), bid(i, gx-1), bid(i+1, gx-1))      # right edge
lines += F

(COAST / "dc_coaster.obj").write_text("\n".join(lines) + "\n")
(COAST / "dc_coaster.mtl").write_text(
    "newmtl map\nKa 1 1 1\nKd 1 1 1\nd 1\nillum 1\nmap_Kd dc_coaster_texture.png\n")
shutil.copy(OUT / "dc_greenmap_print.png", COAST / "dc_coaster_texture.png")

print(f"coaster mesh: {Wmm:.0f} x {Hmm:.0f} mm, base {base_mm} + relief {relief_mm} mm")
print(f"  grid {gx}x{gy}  ({N*2} verts, {len(F)} faces)")
print(f"  -> output/coaster/  (dc_coaster.obj + .mtl + texture)")
print("  zip the 3 files and upload to craftcloud3d.com or i.materialise for full-colour print")
