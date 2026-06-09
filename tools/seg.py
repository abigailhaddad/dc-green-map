"""Shared mosaic segmenter: grout = gradient ridges, markers = flat tile interiors,
watershed boundaries land on the grout. Works for tiles of any brightness."""
import numpy as np
from scipy import ndimage


def segment(lum, grad_pct=50, min_marker=16):
    s = ndimage.gaussian_filter(lum, 1.0)
    gx = ndimage.sobel(s, 0); gy = ndimage.sobel(s, 1)
    grad = ndimage.gaussian_filter(np.hypot(gx, gy), 1.0)
    flat = grad < np.percentile(grad, grad_pct)
    markers, n = ndimage.label(flat)
    sizes = ndimage.sum(np.ones_like(markers), markers, range(1, n + 1))
    markers[np.isin(markers, np.nonzero(sizes < min_marker)[0] + 1)] = 0
    markers, n = ndimage.label(markers > 0)
    surf = np.clip(grad / grad.max() * 255, 0, 255).astype(np.uint8)
    labels = ndimage.watershed_ift(surf, markers.astype(np.int32))
    labels[labels < 0] = 0
    return labels, n
