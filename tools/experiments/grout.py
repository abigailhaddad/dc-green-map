"""Measure the real grout width, and render the mosaic with UNIFORM grout at that width
(uniform = constant gap regardless of tile size). Grout width is a real mosaic parameter
and the eye is sensitive to it, so we measure it and reproduce it explicitly."""
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seg import segment

img = np.asarray(Image.open("/tmp/gm3c.png").convert("RGB"), float)
H, W = img.shape[:2]
lum = img @ [0.299, 0.587, 0.114]
labels, n = segment(lum)

# ---- measure grout ---------------------------------------------------------
# grout = genuinely dark pixels relative to the local (tile-scale) mean
local_mean = ndimage.uniform_filter(lum, size=25)
grout_mask = lum < (local_mean - 12)
# watershed centre-lines = grout skeleton; width ≈ grout area / centreline length
bound = np.zeros_like(labels, bool)
bound[:, :-1] |= labels[:, :-1] != labels[:, 1:]
bound[:-1, :] |= labels[:-1, :] != labels[1:, :]
centerline_len = int(bound.sum())
grout_area = int(grout_mask.sum())
grout_w = grout_area / max(centerline_len, 1)

med_area = np.median([s for s in ndimage.sum(np.ones_like(labels), labels, range(1, n + 1)) if s > 60])
med_diam = np.sqrt(4 * med_area / np.pi)
grout_rgb = img[grout_mask].mean(0) / 255 if grout_area else np.array([0.1, 0.07, 0.05])
print(f"grout width ≈ {grout_w:.1f}px   median tile diam ≈ {med_diam:.0f}px   "
      f"ratio ≈ {grout_w/med_diam:.2f}")
print(f"grout colour RGB ≈ ({grout_rgb[0]:.2f},{grout_rgb[1]:.2f},{grout_rgb[2]:.2f})")

# ---- per-label real colour LUT --------------------------------------------
idx = np.arange(1, n + 1)
lut = np.zeros((n + 1, 3))
for ch in range(3):
    lut[1:, ch] = ndimage.mean(img[..., ch], labels, index=idx) / 255


def render(grout_px):
    out = lut[labels]
    out[labels == 0] = grout_rgb
    if grout_px > 0:
        # uniform grout: every pixel within grout_px/2 of a tile boundary => grout
        g = ndimage.binary_dilation(bound, iterations=max(1, int(round(grout_px / 2))))
        out[g] = grout_rgb
    return np.clip(out, 0, 1)


fig, ax = plt.subplots(1, 4, figsize=(20, 6))
ax[0].imshow(img.astype(np.uint8)); ax[0].set_title("REAL photo"); ax[0].axis("off")
for a, gp, t in [(ax[1], 0, "no grout (what I had)"),
                 (ax[2], grout_w, f"measured grout ≈{grout_w:.1f}px"),
                 (ax[3], grout_w * 2, f"2× grout ≈{grout_w*2:.1f}px")]:
    a.imshow(render(gp)); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig("analysis/grout.png", dpi=85)
print("wrote analysis/grout.png")
