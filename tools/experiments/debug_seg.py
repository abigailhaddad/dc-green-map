"""Diagnose segmentation on a single clear crop: show crop | grout mask | colored tiles."""
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = np.asarray(Image.open("/tmp/gm3c.png").convert("RGB"), float)
crop = img[300:760, 1150:1610]          # silver field
lum = crop @ [0.299, 0.587, 0.114]


def segment(lum, grad_pct=45, min_marker=14):
    """Marker = flat (low-gradient) tile interior — works for tiles of ANY brightness.
    Watershed over gradient; ridges (grout) become the tile boundaries."""
    s = ndimage.gaussian_filter(lum, 1.0)
    gx = ndimage.sobel(s, 0); gy = ndimage.sobel(s, 1)
    grad = ndimage.gaussian_filter(np.hypot(gx, gy), 1.0)
    flat = grad < np.percentile(grad, grad_pct)        # tile interiors
    markers, n = ndimage.label(flat)
    sizes = ndimage.sum(np.ones_like(markers), markers, range(1, n + 1))
    small = np.nonzero(sizes < min_marker)[0] + 1      # drop speck markers
    markers[np.isin(markers, small)] = 0
    markers, n = ndimage.label(markers > 0)
    surf = np.clip(grad / grad.max() * 255, 0, 255).astype(np.uint8)
    labels = ndimage.watershed_ift(surf, markers.astype(np.int32))
    return grad, markers, labels, n


fig, ax = plt.subplots(2, 3, figsize=(16, 11))
for row, gp in enumerate([40, 55]):
    grad, markers, labels, n = segment(lum, gp)
    lab2 = labels.copy(); lab2[lab2 < 0] = 0
    rng = np.random.default_rng(1)
    lut = np.concatenate([[[0.1, 0.07, 0.05]], rng.random((lab2.max() + 1, 3))])
    colored = lut[lab2]; colored[labels < 0] = [0.1, 0.07, 0.05]
    ax[row, 0].imshow(crop.astype(np.uint8)); ax[row, 0].set_title("crop")
    ax[row, 1].imshow(grad, cmap="magma"); ax[row, 1].set_title(f"gradient (grout=bright), flat<{gp}%")
    ax[row, 2].imshow(colored); ax[row, 2].set_title(f"{n} watershed tiles")
    for a in ax[row]:
        a.axis("off")
    print(f"grad_pct={gp}: {n} tiles")
fig.tight_layout(); fig.savefig("analysis/debug_seg.png", dpi=85)
print("wrote analysis/debug_seg.png")
