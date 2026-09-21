# `src/` — pipeline code

| File | Owner | Status | Purpose |
|---|---|---|---|
| `dataset.py` | shared | done | Data loader used by **both** arms — reads split files, applies resizing/augmentation |
| `metrics.py` | shared | done | Dice, IoU, Boundary IoU, HD95, precision/recall — one implementation, both arms |
| `evaluate.py` | shared | done | Evaluation harness — runs `metrics.py` over a model's predictions, writes the per-image results CSV |
| `make_splits.py` | shared | done | Generates the frozen split files under `data/splits/` — run once, never in the pipeline itself |
| `unet.py` | Balamurali | not started | U-Net architecture definition |
| `train_unet.py` | Balamurali | not started | Training loop — all four brackets × three seeds |
| `prompts.py` | Sebastian | done | GT-mask → prompt generation (box + point strategies, jitter, interior-point sampling) |
| `run_sam3.py` | Sebastian | code done, real weighted GPU run pending | SAM 3 PVS-mode zero-shot inference over the test set |

`dataset.py`, `metrics.py`, `prompts.py`, and `evaluate.py` were integration-tested
together end-to-end against a small synthetic fixture (not committed) before being
wired into `run_sam3.py` — see `decisions.md`, 2026-09-21 entry, for exactly what
was and wasn't verified (in particular: the real `facebook/sam3` gated checkpoint
has not been run yet — that needs a GPU + approved HF access, neither available in
the dev sandbox this was written in).

Run order once real data is downloaded (see `data/kvasir-seg/README.md`):

```
python src/make_splits.py                          # once, then commit data/splits/*.txt
python src/run_sam3.py --device cuda --limit 1      # Week-1 gate: one image, real weights
python src/run_sam3.py --device cuda                # full test-set zero-shot run
python src/evaluate.py --split-file data/splits/test.txt \
    --pred-dir results/sam3_preds/box-gt --arm sam3 --run-id box-gt
```

## Non-negotiable

`dataset.py`, `metrics.py`, and `evaluate.py` are shared and written once. Changes
require agreement from both of us — log the change in `decisions.md`.
