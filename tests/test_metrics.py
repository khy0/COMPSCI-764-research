"""Week 1, Task 4 — validate metrics.py on synthetic cases with known answers.

Covers tracker items 4.1-4.4:
  4.1 identical masks    -> Dice = IoU = 1.0
  4.2 disjoint masks     -> Dice = IoU = 0.0
  4.3 partial overlap    -> hand-calculated Dice/IoU, exact match
  4.4 empty-mask edge case -> no crash, sensible values

Also extends the same four scenarios to boundary_iou, hd95, precision, and
recall (not just dice/iou) — the shared-metrics risk register item says
"unit-tested on synthetic masks", not "dice and iou only", and the review that
led to this file's current metrics.py found real bugs specifically in
boundary_iou/hd95, which plain dice/iou tests would never have caught.

Run with:  pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from metrics import boundary_iou, dice, hd95, iou, precision, recall


def square_mask(size: int, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """size x size binary mask; box=(row0, col0, row1, col1) filled with 1s
    (row1/col1 exclusive), everything else 0. box=None -> all-zero mask."""
    m = np.zeros((size, size), dtype=np.uint8)
    if box is not None:
        r0, c0, r1, c1 = box
        m[r0:r1, c0:c1] = 1
    return m


# ---------------------------------------------------------------------------
# 4.1 — identical masks
# ---------------------------------------------------------------------------

class TestIdenticalMasks:
    mask = square_mask(12, (1, 1, 7, 7))  # 6x6 = 36px, away from array edges

    def test_dice_and_iou_equal_one(self):
        assert dice(self.mask, self.mask) == 1.0
        assert iou(self.mask, self.mask) == 1.0

    def test_boundary_iou_equal_one(self):
        assert boundary_iou(self.mask, self.mask, dilation_px=2) == 1.0

    def test_hd95_equal_zero(self):
        assert hd95(self.mask, self.mask) == 0.0

    def test_precision_and_recall_equal_one(self):
        assert precision(self.mask, self.mask) == 1.0
        assert recall(self.mask, self.mask) == 1.0


# ---------------------------------------------------------------------------
# 4.2 — disjoint masks (zero overlap)
# ---------------------------------------------------------------------------

class TestDisjointMasks:
    a = square_mask(12, (0, 0, 3, 3))
    b = square_mask(12, (8, 8, 11, 11))

    def test_dice_and_iou_equal_zero(self):
        assert dice(self.a, self.b) == 0.0
        assert iou(self.a, self.b) == 0.0

    def test_boundary_iou_equal_zero(self):
        assert boundary_iou(self.a, self.b, dilation_px=2) == 0.0

    def test_precision_and_recall_equal_zero(self):
        # no overlap at all -> every predicted pixel is a false positive,
        # every GT pixel is missed
        assert precision(self.a, self.b) == 0.0
        assert recall(self.a, self.b) == 0.0


# ---------------------------------------------------------------------------
# 4.3 — partial overlap, hand-calculated
# ---------------------------------------------------------------------------
#
# A: rows 1-6, cols 1-6   -> 6x6 = 36 px
# B: rows 4-9, cols 4-9   -> 6x6 = 36 px
# overlap: rows 4-6, cols 4-6 -> 3x3 = 9 px
# union = 36 + 36 - 9 = 63
#
# Dice = 2 * overlap / (areaA + areaB) = 2*9 / 72 = 0.25
# IoU  = overlap / union              = 9 / 63    = 1/7 = 0.142857142857...
#
# precision = recall = 0.25 here too, because areaA == areaB == overlap-relative
# symmetric (tp=9, fp=27, fn=27 for both directions).

class TestPartialOverlap:
    a = square_mask(12, (1, 1, 7, 7))
    b = square_mask(12, (4, 4, 10, 10))

    def test_dice_matches_hand_calculation(self):
        assert dice(self.a, self.b) == pytest.approx(0.25, abs=1e-9)

    def test_iou_matches_hand_calculation(self):
        assert iou(self.a, self.b) == pytest.approx(9 / 63, abs=1e-9)

    def test_precision_and_recall_match_hand_calculation(self):
        assert precision(self.a, self.b) == pytest.approx(0.25, abs=1e-9)
        assert recall(self.a, self.b) == pytest.approx(0.25, abs=1e-9)

    def test_boundary_iou_regression(self):
        # Not hand-derivable by a simple area formula (depends on the
        # dilation/erosion geometry), so this is a locked-in regression value
        # computed once from a metrics.py implementation both teammates
        # reviewed and agreed on (see decisions.md). If this ever changes,
        # that's a real behaviour change to discuss, not a test to "fix"
        # by updating the number.
        assert boundary_iou(self.a, self.b, dilation_px=2) == pytest.approx(
            0.12280701754385964, abs=1e-9
        )

    def test_hd95_regression(self):
        assert hd95(self.a, self.b) == pytest.approx(3.637405746046752, abs=1e-6)


# ---------------------------------------------------------------------------
# 4.4 — empty-mask edge cases: confirm no crash, sensible values
# ---------------------------------------------------------------------------

class TestEmptyMasks:
    empty = square_mask(12)
    nonempty = square_mask(12, (1, 1, 7, 7))

    def test_both_empty(self):
        assert dice(self.empty, self.empty) == 1.0
        assert iou(self.empty, self.empty) == 1.0
        assert hd95(self.empty, self.empty) == 0.0
        assert boundary_iou(self.empty, self.empty, dilation_px=2) == 1.0

    def test_pred_empty_gt_nonempty_no_crash(self):
        assert dice(self.empty, self.nonempty) == 0.0
        assert iou(self.empty, self.nonempty) == 0.0
        assert hd95(self.empty, self.nonempty) == float("inf")
        assert precision(self.empty, self.nonempty) == 0.0
        assert recall(self.empty, self.nonempty) == 0.0

    def test_pred_nonempty_gt_empty_no_crash(self):
        assert dice(self.nonempty, self.empty) == 0.0
        assert iou(self.nonempty, self.empty) == 0.0
        assert hd95(self.nonempty, self.empty) == float("inf")
        assert precision(self.nonempty, self.empty) == 0.0
        # GT empty -> "nothing to recall" convention -> trivially 1.0
        assert recall(self.nonempty, self.empty) == 1.0
