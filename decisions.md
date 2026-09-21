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
| 2026-09-22 | Kvasir-SEG sanity checks (2.4–2.6) | Source: https://datasets.simula.no/kvasir-seg/ · Downloaded: 2026-09-20 · 1000 images / 1000 masks found in polyp-seg-comparison/data/kvasir-seg | Balamurali |
| 2026-09-22 | 2.4 filename alignment | PASS — 1000/1000 filenames matched 1:1, zero mismatches. | Balamurali |
| 2026-09-22 | 2.5 mask binarity | WARN — sampled 20 masks, 0.77% of pixels (mean) fall outside {0,255}, consistent with JPEG compression noise near edges. dataset.py must threshold (mask > 127), not assume exact 0/255. | Balamurali |
| 2026-09-22 | 2.6 visual spot-check | PASS — visually spot-checked 8 pairs ['cju2hw5gjlr5h0988so2qqres', 'cju5vutu7ll8w0871dfp92n9p', 'cju887ftknop008177nnjt46y', 'cju846ec0kj7z08012o10klrb', 'cju7ez7r22qbc08015xfoz2wb', 'cju1c6yfz42md08550zgoz3pw', 'cju2xjz2ju8pe0993ysv9wg17', 'cju1haab178i70799tk9z8y8x'], all overlays aligned with the visible polyp. | Balamurali |

