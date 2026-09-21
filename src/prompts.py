"""GT-mask -> SAM 3 PVS prompt generation.

Owner: Sebastian (per src/README.md). Not a shared/non-negotiable file, but its
output format is consumed by run_sam3.py, so changes to the return shape should
still be flagged to Balamurali if evaluate.py's expectations change.

Implements exactly the five strategies in configs/sam3_pvs.yaml:
    box-gt   tight GT bounding box, no jitter        (oracle upper bound)
    box-j10  GT bbox with +/-10% jitter on each side
    box-j20  GT bbox with +/-20% jitter on each side
    pt-1     1 point at the mask's max-distance-transform interior point
    pt-3     3 points sampled uniformly at random from the mask interior

All strategies are deterministic given `seed` (configs/sam3_pvs.yaml: seed 42) —
jitter and random-interior sampling use a `np.random.RandomState` seeded from
(base_seed, image_id) so re-running produces identical prompts, and different
images don't share the same jitter draw.

Boxes are returned in xyxy pixel coordinates (matches the box format used by
`model.predict_inst(..., box=...)` in the official sam3_for_sam1_task_example.ipynb
notebook, i.e. NOT the normalized cxcywh format used by Sam3Processor.add_geometric_prompt
in the text-prompt/PCS notebook — see run_sam3.py for which API this project uses
and why).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


def _seed_for_image(base_seed: int, image_id: str) -> int:
    """Deterministic per-image seed derived from (base_seed, image_id), so
    prompt generation is reproducible per-image regardless of iteration order."""
    h = hashlib.sha256(f"{base_seed}:{image_id}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class Prompt:
    strategy_id: str
    image_id: str
    box: Optional[np.ndarray] = None  # xyxy, shape (4,), or None
    points: Optional[np.ndarray] = None  # (N, 2) xy, or None
    point_labels: Optional[np.ndarray] = None  # (N,) in {0,1}, or None


def _gt_bbox_xyxy(mask: np.ndarray) -> np.ndarray:
    """Tight axis-aligned bounding box of a binary mask, in xyxy pixel coords."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask.")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def make_box_prompt(
    mask: np.ndarray, jitter_pct: float, rng: np.random.RandomState
) -> np.ndarray:
    """GT bbox, optionally jittered by +/-jitter_pct of each side's own extent
    (independently per side, per configs/sam3_pvs.yaml). Clipped to image bounds."""
    x0, y0, x1, y1 = _gt_bbox_xyxy(mask)
    h, w = mask.shape
    if jitter_pct > 0:
        box_w, box_h = (x1 - x0), (y1 - y0)
        dx0, dx1 = rng.uniform(-jitter_pct, jitter_pct, size=2) * box_w
        dy0, dy1 = rng.uniform(-jitter_pct, jitter_pct, size=2) * box_h
        x0, x1 = x0 + dx0, x1 + dx1
        y0, y1 = y0 + dy0, y1 + dy1
    x0, x1 = np.clip([x0, x1], 0, w - 1)
    y0, y1 = np.clip([y0, y1], 0, h - 1)
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def make_point_prompt_max_dt(mask: np.ndarray) -> np.ndarray:
    """Single point at the mask's max distance-transform pixel (i.e. the point
    farthest from the boundary) — robust for crescent-shaped polyps where the
    centroid can fall outside the mask (planning doc, §1.4). Deterministic, no
    RNG needed."""
    dt = distance_transform_edt(mask > 0)
    idx = np.unravel_index(np.argmax(dt), dt.shape)
    y, x = idx
    return np.array([[x, y]], dtype=np.float32)


def make_point_prompt_random_interior(
    mask: np.ndarray, n_points: int, rng: np.random.RandomState
) -> np.ndarray:
    """n_points sampled uniformly at random (without replacement) from mask-interior
    pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot sample interior points from an empty mask.")
    n = min(n_points, len(xs))
    chosen = rng.choice(len(xs), size=n, replace=False)
    pts = np.stack([xs[chosen], ys[chosen]], axis=1).astype(np.float32)
    return pts


def generate_prompt(
    strategy: dict, mask: np.ndarray, image_id: str, base_seed: int
) -> Prompt:
    """Generate one Prompt for one (strategy, image) pair. `strategy` is one
    entry from configs/sam3_pvs.yaml: prompts.strategies (already parsed from
    YAML into a dict)."""
    rng = np.random.RandomState(_seed_for_image(base_seed, image_id + strategy["id"]))
    sid = strategy["id"]

    if strategy["type"] == "box":
        box = make_box_prompt(mask, strategy.get("jitter_pct", 0.0), rng)
        return Prompt(strategy_id=sid, image_id=image_id, box=box)

    if strategy["type"] == "point":
        n_points = strategy["n_points"]
        sampling = strategy["sampling"]
        if sampling == "max_distance_transform":
            pts = make_point_prompt_max_dt(mask)
        elif sampling == "random_interior":
            pts = make_point_prompt_random_interior(mask, n_points, rng)
        else:
            raise ValueError(f"Unknown point sampling strategy: {sampling}")
        labels = np.ones(len(pts), dtype=np.int64)  # all foreground points
        return Prompt(strategy_id=sid, image_id=image_id, points=pts, point_labels=labels)

    raise ValueError(f"Unknown prompt strategy type: {strategy['type']}")


def generate_all_prompts(
    strategies: list[dict], mask: np.ndarray, image_id: str, base_seed: int
) -> dict[str, Prompt]:
    """Generate one Prompt per strategy in configs/sam3_pvs.yaml for a single image."""
    return {
        s["id"]: generate_prompt(s, mask, image_id, base_seed) for s in strategies
    }
