"""Shared segmentation metrics — used by BOTH arms via evaluate.py.

This is a "non-negotiable" shared file (see src/README.md / decisions.md): changes
require agreement from both teammates, logged in decisions.md.

Implements exactly the metric set in configs/eval.yaml: dice, iou, boundary_iou,
hd95, precision, recall. All functions take a single (pred, gt) pair of 2D
binary {0,1} numpy arrays of identical shape and return a Python float.

boundary_iou's dilation width is resolved PER IMAGE as a ratio of that image's
own diagonal (see DEFAULT_DILATION_RATIO below), not a single fixed pixel count
— closed 2026-09-30 in decisions.md. Kvasir-SEG images range from 332x487 to
1920x1072 (diagonal ratio ~3.7x), so a fixed pixel width would be a razor-thin
boundary band on small images and comically strict on large ones. A manual
`dilation_px` override is still accepted by boundary_iou()/compute_all() for a
one-off/debug run, but the default is the ratio-based method.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

# Canonical value from Cheng, Girshick, Dollár & Kirillov, "Boundary IoU:
# Improving Object-Centric Image Segmentation Evaluation" (CVPR 2021), and its
# official implementation (github.com/bowenc0221/boundary-iou-api): dilation
# width = round(dilation_ratio * image_diagonal), minimum 1px. This is the
# paper's own default, not a value this project invented — see decisions.md,
# 2026-09-30 entry, for the worked example at Kvasir-SEG's resolution extremes
# (12px at 332x487, 44px at 1920x1072).
DEFAULT_DILATION_RATIO = 0.02


def resolve_dilation_px(
    mask_shape: tuple[int, int],
    dilation_px: int | None = None,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
) -> int:
    """Pixel width of the boundary band for an image of shape (H, W).

    If `dilation_px` is given explicitly, it's used as-is (manual override —
    e.g. for a one-off debug run at a fixed width). Otherwise the width is
    computed from `dilation_ratio` * this image's own diagonal, per the
    official Boundary IoU implementation, so images of very different
    resolutions (as in Kvasir-SEG) get comparably-sized boundary bands
    relative to their own extent, rather than one band size that's too tight
    for large images and too loose for small ones.
    """
    if dilation_px is not None:
        return dilation_px
    h, w = mask_shape
    img_diag = math.sqrt(h ** 2 + w ** 2)
    return max(1, round(dilation_ratio * img_diag))

# Reviewed and approved 2026-09-30 (decisions.md): every zero-denominator case
# below is already caught by an explicit branch before the division is
# reached, so a smoothing epsilon is unnecessary here (unlike e.g. a soft-Dice
# training loss on continuous values, where it would matter). Confirmed safe
# by exhaustive testing across empty/disjoint/identical/random-overlap mask
# pairs. Kept as a named constant rather than removed outright so it's easy to
# reintroduce with a comment if a future metric variant needs it.
EPS = 0


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sorensen-Dice coefficient (= F1 over pixels). 1.0 if both masks are empty."""
    p, g = _as_bool(pred), _as_bool(gt)
    intersection = np.logical_and(p, g).sum()
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return float(2 * intersection / (p.sum() + g.sum() + EPS))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union. 1.0 if both masks are empty."""
    p, g = _as_bool(pred), _as_bool(gt)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + EPS))


def _mask_boundary(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Boundary band: dilate the mask and its complement, intersect."""
    m = _as_bool(mask)
    struct = np.ones((3, 3), dtype=bool)
    dilated_fg = binary_dilation(m, structure=struct, iterations=dilation_px)
    dilated_bg = binary_dilation(~m, structure=struct, iterations=dilation_px)
    return dilated_fg & dilated_bg


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_px: int | None = None,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to a boundary band
    around each mask's contour. Band width is resolved per-image from
    `dilation_ratio` (default: the paper's own 2% of the image diagonal) unless
    `dilation_px` is given explicitly — see resolve_dilation_px()."""
    p, g = _as_bool(pred), _as_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    resolved_px = resolve_dilation_px(p.shape, dilation_px, dilation_ratio)
    p_boundary = _mask_boundary(p, resolved_px) & p
    g_boundary = _mask_boundary(g, resolved_px) & g
    intersection = np.logical_and(p_boundary, g_boundary).sum()
    union = np.logical_or(p_boundary, g_boundary).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + EPS))


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    """Boundary/contour pixels only: mask minus its 1px erosion."""
    m = _as_bool(mask)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return m & ~eroded


def hd95(pred: np.ndarray, gt: np.ndarray, percentile: float = 95.0) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    Returns:
    - 0.0 if both masks are empty (perfect agreement on "nothing here").
    - np.inf if exactly one mask is empty (undefined distance to a non-existent
      boundary) — evaluate.py must decide how to aggregate this (exclude vs. cap;
      see decisions.md / Methodology checklist — HD95 handling of empty masks is
      NOT yet an open item there and should be added before Week 4 aggregation).

    Fixed 2026-09-30 (decisions.md): distances are computed from CONTOUR pixels
    only (`_mask_edge`), not all foreground pixels. An earlier version used all
    foreground pixels, which is wrong: for any pixel of `p` that lies inside
    `g` (most of the interior, when the masks overlap reasonably well), its
    distance to `g` is exactly 0 — mixing those near-zero interior distances
    into the array before taking the 95th percentile systematically understates
    HD95 (worse for larger/blobbier masks). This matches the standard
    surface-distance definition (e.g. MONAI's compute_hausdorff_distance also
    restricts to boundary/surface points first).
    """
    p, g = _as_bool(pred), _as_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    p_edge, g_edge = _mask_edge(p), _mask_edge(g)

    dt_to_g = distance_transform_edt(~g)
    dt_to_p = distance_transform_edt(~p)

    d_p_to_g = dt_to_g[p_edge]
    d_g_to_p = dt_to_p[g_edge]

    all_d = np.concatenate([d_p_to_g, d_g_to_p])
    return float(np.percentile(all_d, percentile))


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    if tp + fp == 0:
        return 1.0 if g.sum() == 0 else 0.0
    return float(tp / (tp + fp + EPS))


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    tp = np.logical_and(p, g).sum()
    fn = np.logical_and(~p, g).sum()
    if tp + fn == 0:
        return 1.0  # GT is empty; nothing to recall
    return float(tp / (tp + fn + EPS))


METRIC_FUNCS = {
    "dice": dice,
    "iou": iou,
    "boundary_iou": boundary_iou,
    "hd95": hd95,
    "precision": precision,
    "recall": recall,
}


def compute_all(
    pred: np.ndarray,
    gt: np.ndarray,
    metric_names: list[str],
    boundary_dilation_px: int | None = None,
    boundary_dilation_ratio: float = DEFAULT_DILATION_RATIO,
    hd95_percentile: float = 95.0,
) -> dict[str, float]:
    """Compute the requested subset of metrics for one (pred, gt) pair."""
    out = {}
    for name in metric_names:
        if name == "boundary_iou":
            out[name] = boundary_iou(
                pred,
                gt,
                dilation_px=boundary_dilation_px,
                dilation_ratio=boundary_dilation_ratio,
            )
        elif name == "hd95":
            out[name] = hd95(pred, gt, percentile=hd95_percentile)
        else:
            out[name] = METRIC_FUNCS[name](pred, gt)
    return out
