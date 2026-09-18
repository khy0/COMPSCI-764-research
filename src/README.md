# `src/` — pipeline code (not yet written)

No implementation exists yet — this file documents the planned layout from the
project's planning document so the structure is visible before the code is.

| File | Owner | Purpose |
|---|---|---|
| `dataset.py` | shared | Data loader used by **both** arms — reads split files, applies resizing/augmentation |
| `metrics.py` | shared | Dice, IoU, Boundary IoU, HD95, precision/recall — one implementation, both arms |
| `evaluate.py` | shared | Evaluation harness — runs `metrics.py` over a model's predictions, writes the per-image results CSV |
| `unet.py` | Balamurali | U-Net architecture definition |
| `train_unet.py` | Balamurali | Training loop — all four brackets × three seeds |
| `prompts.py` | Sebastian | GT-mask → prompt generation (box + point strategies, jitter, interior-point sampling) |
| `run_sam3.py` | Sebastian | SAM 3 PVS-mode zero-shot inference over the test set |

## Non-negotiable

`dataset.py`, `metrics.py`, and `evaluate.py` are shared and written once. Changes
require agreement from both of us — log the change in `decisions.md`.
