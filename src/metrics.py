"""Shared segmentation metrics — used by BOTH arms via evaluate.py.

This is a "non-negotiable" shared file (see src/README.md / decisions.md): changes
require agreement from both teammates, logged in decisions.md.

Implements exactly the metric set in configs/eval.yaml: dice, iou, boundary_iou,
hd95, precision, recall. All functions take a single (pred, gt) pair of 2D
binary {0,1} numpy arrays of identical shape and return a Python float.

boundary_iou.dilation_px is currently `null` in configs/eval.yaml (open item in
decisions.md) — DEFAULT_DILATION_PX below is a placeholder used only until that
decision is logged; boundary_iou() always accepts an explicit override so
evaluate.py can be pointed at the agreed value without editing this file.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, binary_erosion

# Placeholder only — see configs/eval.yaml boundary_iou.dilation_px (TODO) and
# decisions.md item 2. Do not treat this as the agreed value.
DEFAULT_DILATION_PX = 2

EPS = 0 #1e-7


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
    pred: np.ndarray, gt: np.ndarray, dilation_px: int = DEFAULT_DILATION_PX
) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to a boundary band of
    width `dilation_px` around each mask's contour."""
    p, g = _as_bool(pred), _as_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    p_boundary = _mask_boundary(p, dilation_px) & p
    g_boundary = _mask_boundary(g, dilation_px) & g
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
    """
    p, g = _as_bool(pred), _as_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    # Directed distances: for every foreground pixel of one mask, distance to
    # the nearest foreground pixel of the other. Using all foreground pixels
    # (not just contour pixels) is equivalent for HD95 purposes here since the
    # nearest-point distance for an interior pixel of `p` to `g` is bounded by
    # its distance via `p`'s own boundary — standard practice (e.g. MONAI).
    p_edge, g_edge = _mask_edge(p), _mask_edge(g)


    dt_to_g = distance_transform_edt(~g)
    dt_to_p = distance_transform_edt(~p)

    """d_p_to_g = dt_to_g[p]
    d_g_to_p = dt_to_p[g]"""
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
    boundary_dilation_px: int = DEFAULT_DILATION_PX,
    hd95_percentile: float = 95.0,
) -> dict[str, float]:
    """Compute the requested subset of metrics for one (pred, gt) pair."""
    out = {}
    for name in metric_names:
        if name == "boundary_iou":
            out[name] = boundary_iou(pred, gt, dilation_px=boundary_dilation_px)
        elif name == "hd95":
            out[name] = hd95(pred, gt, percentile=hd95_percentile)
        else:
            out[name] = METRIC_FUNCS[name](pred, gt)
    return out
