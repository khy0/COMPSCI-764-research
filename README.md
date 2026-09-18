# Zero-Shot Foundation Models vs. Task-Specific Architectures for Niche Medical Image Segmentation

**COMPSCI 764 (Deep Learning) — Research Project**
Sebastian Torres · Balamurali Ragupathi — School of Computer Science, University of Auckland

## Research question

> What volume of annotated domain-specific training data does a task-specific U-Net
> require to match or exceed the zero-shot Promptable Visual Segmentation (PVS)
> performance of SAM 3 on the Kvasir-SEG benchmark?

Full experimental design, rationale, and the six-week roadmap live in the project's
planning document (not tracked in this repo — see the team's shared drive / report
submission). This README covers the repo itself.

## Status

🔧 **Scaffolding stage.** Directory layout, configs, and notebook shells are in place.
No pipeline code has been written yet — see `decisions.md` for what's been decided
so far and what's still open.

## Repository layout

```
polyp-seg-comparison/
├── configs/            # YAML configs — every hyperparameter lives here, not in code
├── data/
│   ├── splits/         # COMMITTED index files: test.txt, val.txt, train_10.txt ...
│   └── kvasir-seg/      # raw dataset — gitignored, see its README for download steps
├── src/                 # pipeline code (not yet written — see src/README.md)
├── notebooks/           # analysis + figure notebooks only — no pipeline logic
├── results/             # per-image result CSVs, committed
├── decisions.md         # running log of every design decision + rationale
└── requirements.txt
```

## The two arms

| | SAM 3 (zero-shot) | U-Net (task-specific) |
|---|---|---|
| Owner | Sebastian | Balamurali |
| Mode | Frozen, Promptable Visual Segmentation (PVS) | Trained from scratch, no pretraining |
| Config | `configs/sam3_pvs.yaml` | `configs/unet_base.yaml` |
| Evaluated on | Identical held-out test set (`data/splits/test.txt`) | Identical held-out test set (`data/splits/test.txt`) |

Both arms are scored by a single shared evaluation harness (`src/evaluate.py`, once
written) using the metrics defined in `configs/eval.yaml`. This is deliberate — see
the non-negotiable note below.

## Non-negotiable

`src/dataset.py`, `src/metrics.py`, and `src/evaluate.py` are **shared and written
once**. Do not fork these per-arm — a subtly different metric implementation between
the two arms invalidates the comparison. Changes to these three files require
agreement between both of us, logged in `decisions.md`.

## Setup

Environment setup instructions will be added once the SAM 3 dependency stack
(CUDA / PyTorch / flash-attention versions) is confirmed working — see the risk
register in the planning document. In the meantime:

```bash
conda create -n polyp-seg python=3.11
conda activate polyp-seg
pip install -r requirements.txt   # placeholder — will be pinned once SAM 3 env is confirmed
```

SAM 3 weights are gated on Hugging Face (`facebook/sam3`) — request access before
Week 1 ends.

## Data

Kvasir-SEG (Jha et al., 2020) — see `data/kvasir-seg/README.md` for download
instructions. The train/val/test split and the four nested training subsets
(10% / 25% / 50% / 100%) are frozen by seed and committed under `data/splits/` once
generated — never regenerated ad hoc.

## Contributing

See `decisions.md` before changing anything in `configs/` or the shared `src/`
files listed above.
