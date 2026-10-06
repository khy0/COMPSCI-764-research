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
- [ ] HD95 handling of images where GT or prediction is empty (`src/metrics.py`
      currently returns `inf` for one-empty, `0.0` for both-empty) — decide whether
      Week 4 aggregation excludes `inf` rows or caps them at some value before
      computing bootstrap CIs. Flagged in `src/metrics.py`'s `hd95()` docstring.

<!-- Add dated entries below as decisions are made. -->

| Date | Decision | Rationale | Decided by | 
|---|---|---|---|
| 2026-09-22 | Kvasir-SEG sanity checks (2.4–2.6) | Source: https://datasets.simula.no/kvasir-seg/ · Downloaded: 2026-09-20 · 1000 images / 1000 masks found in polyp-seg-comparison/data/kvasir-seg | Balamurali |
| 2026-09-22 | 2.4 filename alignment | PASS — 1000/1000 filenames matched 1:1, zero mismatches. | Balamurali |
| 2026-09-22 | 2.5 mask binarity | WARN — sampled 20 masks, 0.77% of pixels (mean) fall outside {0,255}, consistent with JPEG compression noise near edges. dataset.py must threshold (mask > 127), not assume exact 0/255. | Balamurali |
| 2026-09-22 | 2.6 visual spot-check | PASS — visually spot-checked 8 pairs ['cju2hw5gjlr5h0988so2qqres', 'cju5vutu7ll8w0871dfp92n9p', 'cju887ftknop008177nnjt46y', 'cju846ec0kj7z08012o10klrb', 'cju7ez7r22qbc08015xfoz2wb', 'cju1c6yfz42md08550zgoz3pw', 'cju2xjz2ju8pe0993ysv9wg17', 'cju1haab178i70799tk9z8y8x'], all overlays aligned with the visible polyp. | Balamurali |

## 2026-09-21 — SAM 3 PVS API confirmed; Week 1 environment findings

**Decision:** `src/run_sam3.py` uses the "SAM 1 task" / interactive API
(`build_sam3_image_model(..., enable_inst_interactivity=True)` →
`Sam3Processor.set_image()` → `model.predict_inst(state, point_coords=, point_labels=,
box=, multimask_output=)`), confirmed against the official
`facebookresearch/sam3` repo's own `examples/sam3_for_sam1_task_example.ipynb`
notebook — not the text-prompt/PCS API in `sam3_image_predictor_example.ipynb`,
which is out of scope for this project's RQ (visual, not text, prompting).

**Rationale / what was actually verified:** `src/dataset.py`, `src/prompts.py`,
`src/metrics.py`, `src/evaluate.py`, `src/make_splits.py`, and `src/run_sam3.py`
were all written and integration-tested together end-to-end (dataset load → all 5
prompt strategies generated from a GT mask → `run_sam3.predict_one()` → predicted
mask saved → `evaluate.py` scored it with every metric in `configs/eval.yaml` →
per-image CSV written) — but only against a synthetic 3-image fixture, not real
Kvasir-SEG, and only with `run_sam3.py`'s `build_model(checkpoint_path=None,
load_from_hf=False)` path (untrained/random SAM 3 weights on CPU), not the real
gated `facebook/sam3` checkpoint on GPU. Concretely, in the development sandbox:
- No GPU driver present (`torch.cuda.is_available()` is `False`).
- `huggingface.co` and `datasets.simula.no` are both outside that sandbox's
  network allowlist (only PyPI and GitHub are reachable), so neither the gated
  SAM 3 checkpoint nor the real Kvasir-SEG dataset could be downloaded there.
- Forcing the official `sam3` package to build/run on CPU (purely to verify the
  API call chain) required three small local patches to hardcoded
  `device="cuda"` assumptions inside `facebookresearch/sam3` itself
  (`model/position_encoding.py`, `model/decoder.py`, `perflib/fused.py` — the
  last one unconditionally cast activations to bfloat16 assuming a CUDA autocast
  context). None of these patches are part of this repo or needed on a real GPU
  machine — they only change behaviour when `device.type != "cuda"`.

**What this means for the real Week-1 gate:** the *code* in `src/run_sam3.py` is
confirmed correct against the official API and the rest of the shared pipeline.
The actual gate — running it with real trained weights on a real polyp image —
still needs to happen once on a machine with a GPU and HF access approved for
`facebook/sam3` (`--device cuda`, omit `--checkpoint-path` to download from HF,
or point it at a manually-downloaded `sam3.pt`). `requirements.txt` was updated
with the exact dependency set that turned out to be needed (`einops` and
`pycocotools` in particular are silently required by `import sam3` but are NOT
in `facebookresearch/sam3`'s own base `pyproject.toml` dependencies — only in its
"notebooks"/"dev" extras — so a bare `pip install git+https://github.com/
facebookresearch/sam3.git` is not enough on its own).

**Owner:** Sebastian.

## macOS import fixes for vendored `sam3/` (2026-09)

Getting `import sam3` to work on Apple Silicon (env `cs764`, Python 3.11) needed:
- `pip install "setuptools<81"` — `sam3/model_builder.py` imports `pkg_resources`,
  which newer setuptools no longer ships.
- `pip install psutil` — imported by `sam3/model/sam3_video_predictor.py`.
- **Local patch** in `sam3/sam3/model/sam3_tracker_utils.py`: the top-level
  `from sam3.model.edt import edt_triton` was moved inside
  `sample_one_point_from_error_center`. `triton` has no macOS wheels, and that
  function is training-time only (RITM-style click sampling), so inference via
  `src/run_sam3.py` never touches it. On a CUDA Linux machine this patch is
  harmless. It is a local diff against upstream `facebookresearch/sam3`.
- **More local patches to make `--device mps` work** (all no-ops on CUDA):
  - `model/position_encoding.py`, `model/decoder.py`: precomputed pos-enc / box-RPB
    coordinate caches were built with a hardcoded `device="cuda"`. They now use CUDA
    if available, else CPU, and are moved to the input's device on use (they are
    plain dicts, so `model.to(device)` doesn't move them).
  - `model_builder.py::_setup_device_and_mode`: upstream only moved the model when
    `device == "cuda"`, so `mps` left weights on CPU ("MPSFloatType vs
    torch.FloatTensor" error). Now moves for any non-CPU device.
  - `perflib/fused.py::addmm_act`: `aten._addmm_activation` is a CUDA-fused bf16
    kernel, not implemented on MPS. Non-CUDA tensors now use unfused
    `linear` + `relu`/`gelu` in the input dtype.
- `pip install scikit-image` — needed by `sam1_task_predictor.py` for mask
  post-processing (skipped with a warning otherwise).
- Result: `python src/run_sam3.py --device mps --limit 1` runs end to end with the real
  `facebook/sam3` weights and writes all 5 strategy masks (plausible foreground
  fractions, not yet scored). MPS runs fp32 while CUDA uses bf16 + fused kernels, so
  outputs will differ slightly from a CUDA run; the official Week-1 gate should
  still be repeated on a GPU machine.
- Note: run import checks from `src/` (or anywhere but the repo root), because
  from the root `import sam3` resolves to the `./sam3` checkout folder.



| Date | Decision | Rationale | Decided by | 
|---|---|---|---|
| 2026-09-25 | Week 1 Task 5 — U-Net smoke test | WARN — no CUDA device, ran on mps (Apple Silicon (mps)), torch 2.11.0. requirements.txt versions are not yet pinned (open TODO in the file itself). | Balamurali |
| 2026-09-25 | 5.2 U-Net architecture | PASS — UNet(depth=4, base_channels=32) in src/unet.py, 7,763,041 params, output shape torch.Size([2, 1, 256, 256]) matches mask shape, no pretrained weights. | Balamurali |
| 2026-09-25 | 5.3 debug-subset wiring | PASS — 20-image debug split written to data/splits/debug_20.txt (first 20 IDs of train_10.txt), DataLoader batches of 4 confirmed working via a custom collate_fn (dataset.py's KvasirSample dataclass isn't batchable by the default collate — flagged for Sebastian, worked around locally, dataset.py untouched). | Balamurali |
| 2026-09-25 | 5.4 loss + optimizer | PASS — combined BCE + soft-Dice loss, Adam optimizer at lr=0.001, both matching configs/unet_base.yaml (training.loss=dice_bce, training.optimizer=adam). | Balamurali |
| 2026-09-25 | 5.5 training loop (2 epochs) | PASS — 2 epochs over 20 images completed with no runtime errors, losses=[1.3389, 1.2746], total wall-clock=5.5s. | Balamurali |
| 2026-09-25 | 5.6 loss decreased | PASS — loss dropped 1.3389 -> 1.2746 across the 2 epochs. | Balamurali |
| 2026-09-25 | 5.7 checkpoint save/reload | PASS — checkpoint saved to checkpoints/unet_smoke_test.pt and reloaded into a fresh UNet instance, forward pass on a real batch ran with no errors. | Balamurali |
| 2026-09-25 | 5.8 runtime/memory/bugs | Device=mps (Apple Silicon (mps)), total wall-clock=5.5s over 2 epochs / 20 images, 2165.9 MB MPS driver-allocated memory (approximate, no per-run peak reset on MPS). Bugs hit: none. | Balamurali |
| 2026-09-27 | Week 2 Task 1 — U-Net training loop | PASS — visually spot-checked augmented image+mask pairs across several re-runs (flip/rotate/scale-jitter/colour-jitter via albumentations), mask stayed aligned with the image in every case. | Balamurali |
| 2026-09-27 | 1.2 early stopping | PASS — trained 10pct/seed0 for 14 epochs (early-stopped, patience=10), best val_dice=0.3766 (src/metrics.py's dice(), same fn evaluate.py uses). | Balamurali |
| 2026-09-27 | 1.3 checkpoint save/reload | PASS — saved checkpoints/unet_10pct_seed0_last.pt and checkpoints/unet_10pct_seed0_best.pt (model+optimizer state, epoch, best_val_dice, seed, bracket); reloaded into a fresh model+optimizer, re-evaluated val_dice=0.3355, matches last epoch's in-memory score: True. | Balamurali |
| 2026-09-27 | 1.4 seeding | PASS — set_seed(0) seeds random/numpy/torch (+ CUDA determinism flags when a GPU is present), called before any dataset/model/DataLoader construction below. | Balamurali |
| 2026-09-27 | Open TODO reminder | training.patience used 10 as a default (not yet a confirmed decision); training.max_iterations still null — both still need a joint decision logged separately. | Balamurali |

**Note (2026-09-28):** "env `cs764`" in the entry above refers to Sebastian's laptop
environment specifically — Balamurali's Mac mini uses a separate `base` conda
environment with the same requirements.txt packages installed. Env *names* differ
per machine; only the package versions need to match.| 2026-09-28 | Week 2 Task 2 — first full 100%-bracket run | PASS — trained 100pct/seed1 for 98 epochs, best val_dice=0.8319, total wall-clock=3474.5s on mps, no crashes. | Balamurali |
| 2026-09-28 | 2.2 loss curve shape | PASS — train_loss 1.1810 -> 0.2826, val_dice 0.3874 -> 0.8222 (best 0.8319) over 98 epochs. Curve shape looks like real learning. | Balamurali |
| 2026-09-28 | 2.3 test-set Dice sanity check | PASS — test-set Dice (via src/evaluate.py, dilation_px override=2) mean=0.8293 std=0.1819 over 200 images. plausible for a first from-scratch U-Net attempt. | Balamurali |
| 2026-09-28 | 2.4 open TODO reminder | boundary_iou.dilation_px used 2 as a one-off --dilation-px override for this run only (configs/eval.yaml still has it as null) — still needs a real joint decision with Sebastian before Week 3's grid. | Balamurali |
