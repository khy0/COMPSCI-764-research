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
