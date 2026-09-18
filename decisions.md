# Decisions log

Running record of design decisions and their rationale, per the planning document's
recommendation. Add an entry whenever a choice is made that isn't already fixed by
the planning doc — especially anything that affects reproducibility (P1–P3) or the
Methodology checklist.

Format:

```
## YYYY-MM-DD — Short title
**Decision:**
**Rationale:**
**Owner:**
```

---

## Open decisions (carried over from the planning document — resolve in Week 1)

- [ ] Epochs vs. iterations for U-Net training budget matching (P3). Planning doc
      recommends fixed max iterations + early stopping on validation Dice — confirm
      and set `training.max_iterations` / `training.patience` in
      `configs/unet_base.yaml`.
- [ ] Boundary IoU dilation width (pixels) — set in `configs/eval.yaml` under
      `boundary_iou.dilation_px`.
- [ ] Multiple-comparisons stance for Week 4 statistics (Bonferroni-correct headline
      claims vs. frame secondary comparisons as exploratory) — set in
      `configs/eval.yaml` under `statistics.multiple_comparisons`.
- [ ] SAM 3 vs. SAM 2 pivot trigger date (Week 1 gate) — decide the exact date/time
      this week and record it here once set.
- [ ] U-Net base width: 32 or 64 channels — set in `configs/unet_base.yaml`.

<!-- Add dated entries below as decisions are made. -->
