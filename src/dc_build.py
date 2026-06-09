"""DC 'Green Map' generator — inspired by Ellen Harvey's "Green Map" at SFO.

Pipeline (see render() at the bottom):
  1. build_region(S)  — classify the real DC satellite into land / parks / water / Capitol,
                        at DC's true (retroceded) outline and aspect ratio.
  2. build_tiles()    — fill it with ONE seamless tessellation made of three tile families,
                        each generated to match metrics measured from the real SFO mosaic:
                          grey land   = deformed quad grid     -> rectangular tesserae
                          green parks = warped variable Voronoi -> interlocking shards
                          water       = flow-elongated cells    -> tiles along the current
  3. color_tiles()    — colour each tile by sampling its region's real palette.
  4. shade_render()   — bevel/emboss, tonal texture, warm-grey grout at 8% of tile size.

Colour palettes (data/color_model.json) and all the target metrics were derived from the
real SFO photo; see tools/ for the analysis that produced them and README.md for the story.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data"
WALL = np.array([0.10, 0.08, 0.06])              # outside the District (dark wall)
GROUT = np.array([0.34, 0.35, 0.29])             # warm grey, measured from the real mosaic
_CM = json.load(open(DATA / "color_model.json"))
PALETTE = {1: np.array(_CM["gray"]), 2: np.array(_CM["green"]),
           3: np.array(_CM["light"]), 4: np.array(_CM.get("gold", [[0.66, 0.56, 0.21]]))}


# ---------------------------------------------------------------------------
# 1. region map from the real DC satellite
# ---------------------------------------------------------------------------
def build_region(S):
    """0 outside · 1 land · 2 green · 3 water · 4 Capitol, at DC's true aspect ratio.
    The District boundary, rivers, and (major) green space are read from the satellite —
    Maryland is overlaid green, Virginia pink, so the District is everything else."""
    a = np.asarray(Image.open(DATA / "dc_satellite.jpg").convert("RGB"), float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]; bright = (r + g + b) / 3
    md = (g > 1.25 * r) & (g > 1.25 * b) & (g > 90)        # Maryland overlay
    va = (r > g + 18) & (b > g + 8)                         # Virginia overlay
    dc = ndimage.binary_opening(~(md | va), iterations=2)
    lab, n = ndimage.label(dc)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    dcmask = ndimage.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
    water = ndimage.binary_opening(dcmask & (b >= g - 4) & (b >= r - 14) & (bright < 155), iterations=1)
    green = dcmask & ~water & ((g - (r + b) / 2) > 13)     # vegetation index = real green space
    green = ndimage.binary_opening(green, iterations=1)
    green = ndimage.binary_closing(green, iterations=4)    # merge canopy into park blobs
    gl, gn = ndimage.label(green)                          # keep only MAJOR parks, drop speckle
    gs = ndimage.sum(np.ones_like(gl), gl, index=range(1, gn + 1))
    green = np.isin(gl, np.nonzero(gs > 0.0015 * dcmask.sum())[0] + 1)
    reg = np.where(dcmask, 1, 0); reg[green] = 2; reg[water] = 3
    ys, xs = np.where(dcmask)
    reg = reg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H = S; W = max(1, int(round(S * reg.shape[1] / reg.shape[0])))
    reg = np.array(Image.fromarray(reg.astype("uint8")).resize((W, H), Image.NEAREST)).astype(int)
    yy, xx = np.mgrid[0:H, 0:W]                             # gold Capitol disc near the quadrant origin
    reg[(((xx - 0.55 * W) ** 2 + (yy - 0.55 * H) ** 2) < (0.02 * H) ** 2) & (reg > 0)] = 4
    return reg


# ---------------------------------------------------------------------------
# 2. the three tile families
# ---------------------------------------------------------------------------
def _jit_grid(H, W, sp, seed, jit=0.42):
    rng = np.random.default_rng(seed)
    ys = np.arange(sp / 2, H, sp); xs = np.arange(sp / 2, W, sp)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gy = gy + rng.uniform(-jit * sp, jit * sp, gy.shape)
    gx = gx + rng.uniform(-jit * sp, jit * sp, gx.shape)
    return np.column_stack([gy.ravel(), gx.ravel()])


def _noise(H, W, scale, seed):
    rng = np.random.default_rng(seed)
    lo = rng.standard_normal((max(2, H // scale), max(2, W // scale)))
    z = ndimage.zoom(lo, (H / lo.shape[0], W / lo.shape[1]), order=3)[:H, :W]
    z -= z.mean(); m = np.abs(z).max() or 1.0
    return z / m


def _quad_grid(H, W, sp, seed, jit=0.34):
    """Seamless deformed lattice: each tile is a 4-cornered quad sharing edges with its
    neighbours => right-angle-ish corners (rectangular tesserae) with hand-cut variation.
    jit=0.24 matches the real tesserae's ~44% right-angle corners."""
    rng = np.random.default_rng(seed)
    ny, nx = int(H / sp) + 2, int(W / sp) + 2
    vy = np.arange(ny)[:, None] * sp + rng.uniform(-jit * sp, jit * sp, (ny, nx))
    vx = np.arange(nx)[None, :] * sp + rng.uniform(-jit * sp, jit * sp, (ny, nx))
    im = Image.new("I", (W, H), 0); d = ImageDraw.Draw(im)
    lab = 1
    for i in range(ny - 1):
        for j in range(nx - 1):
            d.polygon([(vx[i, j], vy[i, j]), (vx[i, j+1], vy[i, j+1]),
                       (vx[i+1, j+1], vy[i+1, j+1]), (vx[i+1, j], vy[i+1, j])], fill=lab)
            lab += 1
    return np.array(im, dtype=int)


def _var_seeds(H, W, base, seed, var=1.0):
    """Jittered grid with seeds dropped in low-noise clusters => big+small cells (size variance)."""
    rng = np.random.default_rng(seed)
    s = _jit_grid(H, W, base, seed)
    n = _noise(H, W, max(2, int(base * 3)), seed + 10)
    si = np.clip(s[:, 0].astype(int), 0, H - 1); sj = np.clip(s[:, 1].astype(int), 0, W - 1)
    keepp = 0.4 + 0.6 * (n[si, sj] + 1) / 2
    keep = rng.random(len(s)) < (1 - var) + var * keepp
    s = s[keep]
    return s if len(s) >= 4 else _jit_grid(H, W, base, seed)


def _water_tiles(region, base, H, W):
    """Tiles ELONGATED along the river current (flow = along the river's length)."""
    wm = region == 3
    if not wm.any():
        return np.zeros((H, W), int)
    sm = ndimage.gaussian_filter(ndimage.distance_transform_edt(wm).astype(float), 2.5)
    gy, gx = np.gradient(sm)
    flow = np.arctan2(gy, gx) + np.pi / 2           # along the river
    s = _jit_grid(H, W, base * 1.1, seed=7)
    si = np.clip(s[:, 0].astype(int), 0, H - 1); sj = np.clip(s[:, 1].astype(int), 0, W - 1)
    inw = wm[si, sj]; s, si, sj = s[inw], si[inw], sj[inw]
    if len(s) < 2:
        return np.zeros((H, W), int)
    ca, sa = np.cos(flow[si, sj]), np.sin(flow[si, sj])
    wy, wx = np.where(wm)
    bestd = np.full(wy.size, np.inf); bestid = np.zeros(wy.size, int)
    for k in range(len(s)):
        dy = wy - s[k, 0]; dx = wx - s[k, 1]
        u = dx * ca[k] + dy * sa[k]; v = -dx * sa[k] + dy * ca[k]
        d = (u / 3.0) ** 2 + v ** 2                 # elongate 3x along flow
        m = d < bestd; bestd[m] = d[m]; bestid[m] = k
    out = np.zeros((H, W), int); out[wy, wx] = bestid + 1
    return out


def _merge_variance(labels, strength, seed):
    """Merge adjacent tiles into clusters in 'big' (high-noise) regions, leaving others tiny
    => heavy-tailed size distribution (raises area-cv toward the real tiles)."""
    H, W = labels.shape
    n = int(labels.max())
    if n < 2:
        return labels
    parent = np.arange(n + 1)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    au, av = labels[:, :-1].ravel(), labels[:, 1:].ravel(); mh = au != av
    bu, bv = labels[:-1, :].ravel(), labels[1:, :].ravel(); mv = bu != bv
    pu = np.concatenate([au[mh], bu[mv]]); pv = np.concatenate([av[mh], bv[mv]])
    ok = (pu > 0) & (pv > 0); pu, pv = pu[ok], pv[ok]
    key = np.unique(np.minimum(pu, pv).astype(np.int64) * (n + 1) + np.maximum(pu, pv))
    pa, pb = (key // (n + 1)).astype(int), (key % (n + 1)).astype(int)
    ln = ndimage.mean(_noise(H, W, 45, seed + 1), labels, index=np.arange(1, n + 1))
    rng = np.random.default_rng(seed)
    rnd = rng.random(len(pa))
    for i in rng.permutation(len(pa)):
        a, b = pa[i], pb[i]
        if rnd[i] < strength * (ln[a - 1] + 1) / 2:        # merge more where noise is high
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    _, newids = np.unique(roots, return_inverse=True)
    return newids[labels].astype(int)


def build_tiles(region):
    """One seamless tessellation, three distinct families matched to the SFO measurements:
      grey land   right-angle corners ~44%, area-cv ~1.0
      green parks solidity ~0.7 (interlocking), area-cv ~1.5, ~1.1x grey size
      water       elongated along the current"""
    H, W = region.shape
    base = 12.0
    Y, X = np.mgrid[0:H, 0:W]
    grey_id = _quad_grid(H, W, base, seed=1, jit=0.24)
    grey_id = _merge_variance(grey_id, strength=0.35, seed=11)
    amp = base * 0.95
    wy = amp * _noise(H, W, 7, seed=3); wx = amp * _noise(H, W, 7, seed=4)
    gpts = np.column_stack([(Y + wy).ravel(), (X + wx).ravel()])
    green_id = cKDTree(_var_seeds(H, W, base * 1.25, seed=2, var=1.0)).query(gpts)[1].reshape(H, W)
    green_id = _merge_variance(green_id, strength=0.45, seed=12)
    water_id = _water_tiles(region, base, H, W)
    big = grey_id.copy()
    o1 = int(grey_id.max()); big[region == 2] = green_id[region == 2] + o1
    o2 = int(big.max()); big[region == 3] = water_id[region == 3] + o2
    big[region == 4] = int(big.max()) + 1            # Capitol = one tile (smooth disc)
    return big, int(big.max())


# ---------------------------------------------------------------------------
# 3. colour + 4. shading
# ---------------------------------------------------------------------------
def color_tiles(big, nbig, region, seed=3):
    """Colour each tile by sampling its region's real palette, then per-tile brightness jitter."""
    counts = np.zeros((nbig + 1, 5), int)
    np.add.at(counts, (big.ravel(), region.ravel()), 1)
    tile_region = counts.argmax(1)
    rng = np.random.default_rng(seed)
    lut = np.tile(WALL, (nbig + 1, 1))
    for i in range(1, nbig + 1):
        r = tile_region[i]
        if r in PALETTE:
            c = PALETTE[r][rng.integers(len(PALETTE[r]))]
            lum = float(np.mean(c))
            if r == 3:                                   # water: muted blue ramp, keep shimmer
                c = np.clip(np.array([0.33, 0.52, 0.78]) * (0.5 + 0.75 * lum), 0, 1)
            elif r == 1:                                 # land: clean neutral-cool grey
                c = np.clip(0.72 * np.array([lum * 0.97, lum, lum * 1.04]) + 0.28 * c, 0, 1)
            elif r == 2:                                 # parks: boost green saturation
                c = np.clip(np.array([c[0] * 0.85, c[1] * 1.05, c[2] * 0.8]), 0, 1)
            lut[i] = np.clip(c * rng.uniform(0.84, 1.14), 0, 1)
    return lut, tile_region


def grout_px(big, nbig):
    areas = ndimage.sum(np.ones_like(big), big, index=range(1, nbig + 1))
    diam = np.sqrt(4 * np.median([a for a in areas if a > 30]) / np.pi)
    return max(1, int(round(0.08 * diam))), diam       # grout = 8% of tile diameter (measured)


def shade_render(big, region, lut, gpx):
    H, W = big.shape
    out = lut[big].astype(float)
    bound = np.zeros((H, W), bool)
    bound[:, :-1] |= big[:, :-1] != big[:, 1:]
    bound[:-1, :] |= big[:-1, :] != big[1:, :]
    ed = ndimage.distance_transform_edt(~bound)
    g = ndimage.gaussian_filter(ed, 0.7); gy, gx = np.gradient(g)
    emboss = np.clip(-(gx + gy) / 1.4, -1, 1)           # light from top-left => raised tiles
    shade = (1 + 0.14 * emboss) * (0.80 + 0.20 * np.clip(ed / 3.0, 0, 1))
    nz = ndimage.gaussian_filter(np.random.default_rng(2).standard_normal((H, W)), 1.3)
    shade *= 1 + 0.06 * nz                              # within-tile tonal texture
    out *= shade[..., None]
    out[ndimage.binary_dilation(bound, iterations=max(1, gpx // 2))] = GROUT * 0.92
    out[region == 0] = WALL
    # Capitol = one smooth gold disc (a single object, not tiles), with a soft radial sheen
    cap = region == 4
    if cap.any():
        ed4 = ndimage.distance_transform_edt(cap)
        sheen = 0.78 + 0.42 * ed4 / (ed4.max() or 1.0)
        out[cap] = np.clip(np.array([0.82, 0.64, 0.15]) * sheen[..., None], 0, 1)[cap]
    return np.clip(out, 0, 1)


def render(S, seed=3):
    """S = output height in px (width follows DC's real aspect ratio)."""
    region = build_region(S)
    big, nbig = build_tiles(region)
    lut, tile_region = color_tiles(big, nbig, region, seed)
    gpx, diam = grout_px(big, nbig)
    out = shade_render(big, region, lut, gpx)
    return dict(out=out, big=big, region=region, lut=lut, tile_region=tile_region,
                gpx=gpx, diam=diam, nbig=nbig)
