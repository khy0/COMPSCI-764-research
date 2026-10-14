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

- [ ] Epochs vs. iterations for U-Net training budget matching (P3) — STILL OPEN,
      see 2026-10-10 entry below. `training.max_epochs` / `training.patience` are
      now wired from `configs/unet_base.yaml` (fixed 2026-10-09, PR #4), but that
      only fixed a dead-config-key bug — it does NOT make the budget matched
      across brackets, since brackets differ ~10x in images-per-epoch. Needs a
      joint decision: step-based budget, or a logged, justified epoch-based
      deviation.
- [x] ~~Boundary IoU dilation width (pixels)~~ — RESOLVED 2026-09-30, see dated
      entry below. Not a fixed pixel count: `configs/eval.yaml:
      boundary_iou.dilation_ratio` (0.02, the paper's own default), resolved
      per-image.
- [ ] Multiple-comparisons stance for Week 4 statistics (Bonferroni-correct headline
      claims vs. frame secondary comparisons as exploratory) — set in
      `configs/eval.yaml` under `statistics.multiple_comparisons`.
- [ ] SAM 3 vs. SAM 2 pivot trigger date (Week 1 gate) — decide the exact date/time
      this week and record it here once set. **Worth explicitly closing now rather
      than leaving open indefinitely:** the real Week 1 gate has since succeeded
      (macOS/mps run below) and the real Week 2 box-gt full-test-set run (see
      2026-09-30 entry below) is strong (median Dice 0.948), so the pivot condition
      has been overtaken by events either way — but that's a call for the two of
      you to make and log explicitly, not something to leave silently implied.
- [ ] U-Net base width: 32 or 64 channels — set in `configs/unet_base.yaml`.
      **Worth explicitly closing now:** Balamurali's Week 1 Task 5 smoke test
      (below) already implemented and ran `base_channels=32`
      (7,763,041 params) — if that's the team's final choice, log it as a real
      decision rather than leaving the checklist item open under an
      already-shipped default.
- [ ] HD95 handling of images where GT or prediction is empty (`src/metrics.py`
      currently returns `inf` for one-empty, `0.0` for both-empty) — decide whether
      Week 4 aggregation excludes `inf` rows or caps them at some value before
      computing bootstrap CIs. Flagged in `src/metrics.py`'s `hd95()` docstring.
      Note (2026-09-30): not yet triggered in practice — none of the 200 real
      box-gt results came back `inf` (Kvasir-SEG's GT is never empty, and SAM3's
      worst box-gt failures were wrong-location, not empty, predictions — see
      dated entry below). Still worth deciding before a prompt strategy more
      likely to produce a genuinely empty mask (e.g. `pt-1`) is scored.

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

## 2026-09-30 — metrics.py: EPS→0 and HD95 contour fix, reviewed and applied

**Decision:** Approved Balamurali's `src/metrics.py` changes (EPS 1e-7→0 in
dice/iou/boundary_iou/precision/recall; `hd95()` restricted to contour pixels
via a new `_mask_edge()` helper instead of all foreground pixels).

**Rationale:** EPS was redundant — every zero-denominator case in all five
functions is already caught by an explicit branch before the division, and
this was confirmed by exhaustive testing (820 calls across empty/disjoint/
identical/random-overlap mask pairs, zero NaN/Inf with EPS=0). The HD95 change
is a real correctness fix, not cosmetic: the original implementation included
every foreground pixel of each mask, not just its contour, which mixes in
near-zero interior-to-other-mask distances whenever the masks overlap and
systematically understates HD95 (more so for larger/blobbier masks) — matches
MONAI's actual surface-distance implementation, which the removed comment had
incorrectly cited to justify the original (wrong) shortcut. Also removed a
leftover dead triple-quoted comment-out block in `hd95()`.

**Owner:** Sebastian (review) / Balamurali (original change).

## 2026-09-30 — Week 2: full SAM3 inference, compute profiling, set_image bug fix

**Decision:** `src/run_sam3.py` now runs all 5 prompt strategies over the full
test split (the loop structure was already there from Week 1 — `--limit` is
purely a debug knob now), optionally auto-invokes the shared `evaluate.py`
harness per strategy via `--evaluate` (same scoring code, no duplication, just
saves 5 manual CLI calls), and records per-image compute profiling
(`results/sam3_compute_profile.csv`) via a new `src/profiling.py` utility.

**Bug found and fixed while adding profiling:** the Week 1 version of
`predict_one()` called `processor.set_image()` — the expensive image-encoder
pass — *inside itself*, so it re-ran once per prompt strategy instead of once
per image. Across 5 strategies that's ~5x wasted encoder compute on every
real run over the full test set. It was invisible in the Week 1 "one image"
gate check because the redundancy was small in absolute terms there; it only
became obvious once profiling was added and a 3-image / 5-strategy smoke test
took 6+ minutes and timed out. Fixed by hoisting `set_image()` out of
`predict_one()` into `run()`'s per-image loop (called once, result reused
across all 5 strategies) — confirmed by re-running the same smoke test, which
then completed in 108s instead of timing out past 400s. The profiler now
measures the one-time embed step and each strategy's decode step as *separate*
rows (`run_id="embed"` vs. `run_id=<strategy_id>`), which is both the fix and
a more informative breakdown than a single per-prompt number would have been.

**Profiling methodology (see `src/profiling.py` for full detail):** latency is
wall-clock per call; parameter count is a one-off `sum(p.numel() ...)` after
model build (860,055,224 total for the full SAM 3 image model, 0 trainable —
correct for a frozen zero-shot model); memory is measured differently on CPU
(`resource.getrusage().ru_maxrss`, a process-lifetime peak, NOT call-scoped)
vs. CUDA (`torch.cuda.max_memory_allocated()`, a true per-call peak) — these
are not directly comparable to each other. Note: same caveat likely applies to
MPS — check what `src/profiling.py`'s device check resolves to on `mps` before
trusting a Mac-collected number against a CUDA one.

**Owner:** Sebastian.

## 2026-09-30 — Boundary IoU dilation width: resolved as a ratio, not a fixed pixel count

**Decision:** `configs/eval.yaml: boundary_iou.dilation_px` (was `null`, the
open item this closes) is replaced as the primary mechanism by
`boundary_iou.dilation_ratio: 0.02`. The boundary-band width used by
`boundary_iou()` is now resolved PER IMAGE as `round(dilation_ratio *
sqrt(h**2 + w**2))` (min 1px) — see `metrics.resolve_dilation_px()`. A fixed
`dilation_px` is still accepted as a manual override (for a one-off/debug run
at a constant width) but is no longer the normal path, and `evaluate.py` no
longer raises if it's left `null` — it just uses the ratio.

**Rationale:** Kvasir-SEG images range from 332x487 to 1920x1072 pixels
(confirmed on the dataset's own page, datasets.simula.no/kvasir-seg) — a
~3.7x range in diagonal length. A single fixed `dilation_px` (the placeholder
had been 2) would be an inconsistent metric across the dataset: a
razor-thin boundary band relative to object size on the small end, and far
too strict relative to object size on the large end — the exact scale problem
Boundary IoU as a metric family exists to be robust to, if implemented
correctly. 0.02 is not a value picked for this project — it's the default
from the metric's origin paper (Cheng, Girshick, Dollár & Kirillov, "Boundary
IoU: Improving Object-Centric Image Segmentation Evaluation," CVPR 2021,
https://arxiv.org/abs/2103.16562) and its official reference implementation
(github.com/bowenc0221/boundary-iou-api), which computes the dilation the
same way: `dilation = round(dilation_ratio * img_diag)`, `dilation_ratio =
0.02` by default. Worked example at Kvasir-SEG's resolution extremes:
- Smallest image (332x487): diagonal = 589.4px → 2% → 12px boundary band.
- Largest image (1920x1072): diagonal = 2199.0px → 2% → 44px boundary band.

**Why this is the better choice over just picking one fixed number:** citing
the paper's own default is easy to defend in the report and to explain to
anyone reviewing the methodology — it's "the standard way to do this,"
not an internal team guess that needs its own justification. It also removes
an entire axis of arbitrariness from the results before any real evaluation
pass is run, rather than after.

**Verified:** this was the fix for the real crash Sebastian's teammate hit
running the full test set (`evaluate_run()` raising `ValueError:
boundary_iou.dilation_px is still null`) — see the 200-row box-gt results
entry directly below, which is the first real evaluation pass run after this
fix, with no `--dilation-px` override needed.

**Owner:** Sebastian.

## 2026-09-30 — First real Week 2 results: box-gt, full 200-image test set

**Finding:** `src/run_sam3.py --evaluate` was run for real (GPU/mps, real
`facebook/sam3` weights, real Kvasir-SEG test split, 200/200 images, no
missing predictions) — this is the first real end-to-end confirmation the
whole pipeline works, not just the Week 1/2 smoke tests. Predictions and
compute profiling completed for **all 5 prompt strategies** (1200 profiling
rows = 200 images × (1 embed + 5 strategies)) before the (now-fixed)
`dilation_px` crash stopped scoring after strategy 1. `box-gt` was then
re-scored standalone after the fix; the other 4 strategies' predictions
already exist on disk and only need `evaluate.py` run against them — **no
inference needs to be re-run.**

`box-gt` (oracle GT-derived box — the upper-bound condition per
`configs/sam3_pvs.yaml`'s own comment) results, n=200:

| metric | mean | median |
|---|---|---|
| dice | 0.894 | 0.948 |
| iou | 0.837 | 0.902 |
| boundary_iou | 0.588 | 0.655 |
| hd95 | 34.7 px | 8.0 px |
| precision | 0.908 | 0.991 |
| recall | 0.900 | 0.931 |

**Mean vs. median:** HD95's mean is >4x its median (34.7 vs 8.0px) — heavily
right-skewed by 9/200 images (4.5%) with dice < 0.5, which explains almost all
of the mean/median gap across every metric. The other 95.5% of images are
excellent (75th pct dice 0.967, 90th pct 0.975). **Report median or the
bootstrap CI for HD95 in particular, not the raw mean** — the mean alone
overstates typical boundary error by describing the tail, not the center.

**Boundary IoU vs. IoU gap is real, not just the failures:** boundary_iou sits
~0.25 below plain IoU even at the median (0.655 vs 0.902), so this isn't only
the 9 bad images — SAM3's zero-shot masks capture gross polyp extent well but
are comparatively imprecise at the fine boundary. Worth keeping as a headline
qualitative finding.

**9 failure images (dice < 0.5), worth a manual visual check before trusting
aggregate stats fully:** two are near-total misses even under an *oracle* box
prompt — `cju3518w2d838079939fqztbc` (dice=0.0, hd95=223.9px — finite, not
`inf`, meaning the prediction was NOT empty, it was non-empty and
**disjoint from the GT polyp**, i.e. SAM3 confidently segmented something at
the wrong location) and `cju2ysg748ru80878sp6j0gm0` (dice≈0.00003,
hd95=133.8px, same pattern). Polyp size does not explain the failures
(dice vs. `gt_area_pct` correlation is weak, r=-0.17; the 9 failures range
8-30% of frame, unremarkable sizes). Good candidates to seed
`notebooks/04_failure_analysis.ipynb`.

**Next step:** run `evaluate.py` for the remaining 4 strategies
(`box-j10`, `box-j20`, `pt-1`, `pt-3`) against the predictions already on
disk — fast, no GPU/model needed, just scoring. `box-gt` alone is the oracle
ceiling per its own config comment, not "SAM3's zero-shot performance" — the
RQ comparison needs all 5.

**Owner:** Sebastian.

## 2026-09-30 — All 5 SAM3 prompt strategies scored, full 200-image test set

**Finding:** the remaining 4 strategies (`box-j10`, `box-j20`, `pt-1`, `pt-3`)
were scored against the predictions already on disk from the earlier full run
(no re-inference needed, as expected). `results/per_image_results.csv` now
has all 1000 rows (200 images x 5 strategies). This closes out the "full SAM3
inference + scoring" part of Week 2.

| strategy | dice mean | dice median | boundary_iou median | hd95 median (px) | dice<0.5 (failures) | dice<0.01 (total miss) |
|---|---|---|---|---|---|---|
| box-gt | 0.894 | 0.948 | 0.655 | 8.0 | 9/200 (4.5%) | 3 |
| box-j10 | 0.846 | 0.934 | 0.551 | 12.5 | 17/200 (8.5%) | 4 |
| box-j20 | 0.744 | 0.877 | 0.418 | 27.2 | 34/200 (17.0%) | 14 |
| pt-1 | 0.820 | 0.925 | 0.551 | 18.9 | 21/200 (10.5%) | 2 |
| pt-3 | 0.865 | 0.933 | 0.589 | 12.9 | 12/200 (6.0%) | 0 |

**Ordering is sane and consistent across every metric (dice, IoU,
boundary_iou, precision, recall) at both mean and median:** `box-gt` >
`pt-3` > `box-j10` > `pt-1` > `box-j20`. Matches expectations — oracle box is
the ceiling, larger box jitter hurts more than smaller jitter, 3 points beat
1 point. `pt-3` edging out `box-j10` (3 points carrying more shape signal
than a box already offset 10% from the true extent) holds across every
metric together, so it reads as a real effect rather than noise from one
metric.

**Mean/median gap from the `box-gt` entry above is not a one-off — it holds
for all 5 strategies** (e.g. dice mean 0.744 vs median 0.877 for `box-j20`):
every strategy has a tail of bad cases dragging its mean down well below its
typical (median) performance. Reinforces the existing `configs/eval.yaml`
choice of bootstrap CIs + Wilcoxon (median-based, nonparametric) as the
headline stats — a plain mean +/- SD would materially misrepresent all 5
runs, not just `box-gt`.

**Reportable trade-off, not just "boxes beat points":** `pt-3` has ZERO
total-miss cases (dice < 0.01) across all 200 images — fewer than even the
oracle `box-gt` (3) — while `box-j20` has 14. Box prompts give the best
*typical* accuracy but degrade catastrophically more often as the box gets
noisier; point prompts (especially 3-point) are lower-ceiling but more
failure-resistant. Good candidate for a headline qualitative finding
alongside the boundary_iou-vs-IoU gap noted in the `box-gt` entry.

**No pipeline red flags:** ordering is consistent across every metric,
failure counts scale sensibly with prompt degradation (more jitter -> more
failures), and the re-derived `box-gt` mean/median here match the standalone
`box-gt` run exactly. Ready for the actual paired Wilcoxon/bootstrap
comparison once the multiple-comparisons stance (still open above) is
settled.

**Owner:** Sebastian.

| 2026-10-06 | 3.1 box-gt data received | Received Sebastian's SAM 3 box-gt results (200 images). | Balamurali |
| 2026-10-06 | 3.2 Mansoori et al. reported range | Mansoori et al., "Polyp SAM 2" (ICHMS 2025 / arXiv:2408.05892), Table 2: box-gt mDice=0.939. Verified against the PDF directly. | Balamurali |
| 2026-10-06 | 3.3 box-gt cross-check | Our SAM3 box-gt mean Dice=0.8943 (std=0.1669, n=200) vs Mansoori's 0.9390 (SAM 2). Gap=+0.0447. Verdict: PASS. | Balamurali |

| 2026-10-06 | Week 2 Task 4 — compute profiling hooks (U-Net) | PASS (with caveat) — reused Sebastian's src/profiling.py ComputeProfiler/count_params, same methodology as SAM3 side. 10pct/seed0 test run: 7,763,041 trainable params, total training wall-clock=321.9s, inference latency (steady-state, excl. first-call MPS cold-start of 619ms) mean=2.72ms std=0.41ms over 200 test images. Peak memory NOT separable between training/inference on MPS — ru_maxrss is a process-lifetime high-water mark (612.4MB for both), a documented limitation of src/profiling.py's CPU/MPS path, not a per-call peak like the CUDA path gives. | Balamurali |

## Week 3 Task 1 — U-Net grid (4 brackets x 3 seeds = 12 runs)
Date: 2026-10-07
Status: COMPLETE

Resolved config/unet_base.yaml TODOs: max_iterations=100, patience=10.

Best val_dice per run:
  10pct:  seed0=0.5327  seed1=0.5107  seed2=0.3716
  25pct:  seed0=0.5425  seed1=0.4947  seed2=0.3709
  50pct:  seed0=0.7927  seed1=0.5187  seed2=0.7916
  100pct: seed0=0.8249  seed1=0.8029  seed2=0.8327

Note: high seed variance at 10pct/25pct (~0.16-0.17 spread); 50pct_seed1
is a clear outlier vs. seed0/seed2 — flagged for follow-up.
100pct_seed2 ran the full 100-epoch cap without early stopping
(best val_dice=0.8327 @ epoch 97); the other 11 runs all stopped earlier.

## 2026-10-10 — Training budget matching (P3): status correction + rationale

**Status: OPEN (was incorrectly implied closed by the 2026-10-07 entry above).**
Flagged by Sebastian in PR #4 review — no rationale or joint sign-off was ever
logged for closing this, and he's right that the earlier fix (wiring
`max_epochs`/`patience` from config, PR #4) only fixed a *different* bug: those
keys were previously dead (ignored by `train_unet.py`, which used hardcoded
argparse defaults instead). That fix is real and still correct, but it does not
make the training budget *matched* across brackets as P3 requires.

**The actual problem:** budgeting by epoch count is not comparable across
brackets of very different size. Confirmed from `logs/train_*.log`'s
`train_images=` line:

| bracket | train_images | batch_size | steps/epoch (approx) |
|---|---|---|---|
| 10pct  | ~70  | 8 | ~9  |
| 25pct  | 175  | 8 | ~22 |
| 50pct  | 350  | 8 | ~44 |
| 100pct | 700  | 8 | ~88 |

"100 epochs, patience 10" is a ~10x-different gradient-update budget depending
on bracket — this directly confounds the RQ.

**Evidence several low-bracket runs plausibly stopped before real learning
happened**, from total-epochs vs. best-epoch in `logs/train_*.log`:

| run | total epochs | best epoch | best val_dice |
|---|---|---|---|
| 10pct_seed1 | 40 | 30 | 0.5107 |
| 10pct_seed2 | 11 | 1  | 0.3716 |
| 25pct_seed0 | 52 | 42 | 0.5425 |
| 25pct_seed1 | 32 | 22 | 0.4947 |
| 25pct_seed2 | 13 | 3  | 0.3709 |
| 50pct_seed1 | 38 | 28 | 0.5187 |

`10pct_seed2` and `25pct_seed2` effectively never trained past their first few
epochs before patience ran out — low Dice there is very likely a stopping
artifact, not purely data scarcity. (`10pct_seed0` has no log file — that run
reused an older pre-grid checkpoint and was never re-executed by the Week 3
grid script; it should be re-run from scratch.)

**Decision needed (joint, not yet made):** (a) switch to a step-based budget —
same `max_steps` across brackets, validate every K steps, patience counted in
evaluations — or (b) keep epoch-based budgeting with an explicit, logged
justification. Either way, add a minimum-epochs-before-early-stopping guard.
Also open: `training.lr_schedule: cosine` is set in `configs/unet_base.yaml`
but is not implemented anywhere in `src/train_unet.py` — decide whether to
implement it or remove the key and log it as unused.

**Owner:** Sebastian + Balamurali (joint sign-off required before re-closing
the checklist item above).

## Week 3 Task 1.4/1.5 — U-Net test-set evaluation (all 12 checkpoints)
Date: 2026-10-07
Status: COMPLETE

Ran src/evaluate_unet_grid.py: inference on frozen test.txt (200 images) for all
12 checkpoints, scored via shared src/evaluate.py (real methodology — dilation_ratio
=0.02, not the earlier --dilation-px=2 one-off override), appended to
results/per_image_results.csv (now 3400 rows: 1000 sam3 + 2400 unet, no dupes/missing).

Status: PROVISIONAL — see 2026-10-10 entries below (training-budget fairness
still open; evaluation mask-resizing fix pending; all 12 runs should be
re-evaluated once that's fixed).

Test-set Dice per bracket: 10pct=0.479  25pct=0.484  50pct=0.705  100pct=0.819.
10pct->25pct (0.479->0.484) is NOT a confirmed trend — it's well inside the
observed seed spread at those brackets (~0.15-0.17) and should be read as "no
detectable difference," not "monotonic increase confirmed." 50pct->100pct is a
real, seed-spread-exceeding jump.

50pct_seed1 outlier (0.542 vs 0.794/0.779 for seeds 0/2) confirmed at test-set
level too — not a validation artifact, still unexplained, flagged for Sebastian.

8/2400 unet rows have non-finite hd95 (re-counted after the bilinear-upsampling
fix above — was 7 before that fix; the fix changed which borderline
predictions end up empty). All 8 have dice=0.0 on that image, i.e. the
PREDICTED mask was empty (GT is never empty for Kvasir-SEG, so "empty GT"
cannot occur here — correcting the earlier, inaccurate "empty predicted or GT
mask" wording). The 8 rows, by run_id/image_id:
  25pct_seed0  / cju1fuoa4wmc50835qfd11sp9
  50pct_seed0  / cju15l5ubz9yh0855b3ivdpse
  50pct_seed2  / cju15l5ubz9yh0855b3ivdpse
  50pct_seed2  / cju6vvb8svhed0801jjcquh5e
  50pct_seed2  / cju89z6pqpqfx0817mfv8ixjc
  100pct_seed0 / cju89z6pqpqfx0817mfv8ixjc
  100pct_seed1 / cju15l5ubz9yh0855b3ivdpse
  100pct_seed1 / cjyzjzssvd8pq0838f4nolj5l
Excluded from hd95 means by evaluate.py's own filtering, expected behaviour.
cju15l5ubz9yh0855b3ivdpse now shows up empty in THREE separate runs
(50pct_seed0, 50pct_seed2, 100pct_seed1) and cju89z6pqpqfx0817mfv8ixjc in two
(50pct_seed2, 100pct_seed0) — both have low gt_area_pct (3.48%, 1.49%). Worth
a quick visual check; reads as genuinely hard small polyps that the model
is marginally under-confident on, not run-specific noise.

Deleted stale results/unet_100pct_seed1_per_image.csv (old checkpoint, old
dilation override, superseded by this run).

## 2026-10-12 — Week 3 grid re-run: full retrain (unintended), early-stopping failure found

**What happened:** run_week3_grid.sh's skip-check pointed at a stale path for
the profiling CSV (pre-dates PR #4's move to results/compute_profiles/), so
it silently retrained all 12 configs instead of just the intended
10pct_seed0 fix. All 12 checkpoints are now fresh (2026-10-09/10).

**New best val_dice per run:**
  10pct:  seed0=0.4992  seed1=0.4739  seed2=0.4953
  25pct:  seed0=0.5194  seed1=0.7077  seed2=0.5171
  50pct:  seed0=0.8166  seed1=0.5295  seed2=0.5446
  100pct: seed0=0.5259  seed1=0.8010  seed2=0.8338

**Critical finding — 100pct_seed0 (val_dice=0.5259, stopped at epoch 19) is
not a valid result.** Every other 100pct run reaches 0.80+. Its log shows
early stopping triggered after a noisy dip (epoch 6: val_dice=0.2154) burned
through patience while the run was still mid-climb. This demonstrates the
early-stopping criterion itself is noise-sensitive independent of bracket
size or step-count budget — it just killed the largest, most data-rich
bracket by bad luck. 25pct_seed1 shows the mirror failure: hit the 100-epoch
cap still improving (0.7077* at epoch 99), right-censored rather than
converged.

**This escalates the open 2026-10-10 training-budget entry above:** whatever
step-vs-epoch decision gets made, the stopping rule also needs smoothing
(e.g. patience measured against a moving average of val_dice, not the raw
single-epoch value) or higher patience — this would have broken 100pct_seed0
regardless of the budget-matching fix.

**Status: current per_image_results.csv numbers from this grid should be
treated as provisional pending a stopping-criterion fix and a targeted
re-run of 100pct_seed0 at minimum.**

**Owner:** Sebastian + Balamurali (same joint call as the budget decision —
recommend covering both in one sync, not two).

| Date | Decision | Rationale | Decided by |
|---|---|---|---|
| 2026-10-13 | Replace epoch-based early stopping with: (1) fixed 8,800-step training budget for all four data brackets, validated every 88 steps (100 validation checkpoints per run, regardless of bracket size); (2) checkpoint selection on a 3-validation moving average of val Dice instead of single-point best-vs-current; (3) cosine LR schedule with ~5% warmup, applied uniformly across brackets (replaces the dead `lr_schedule` config key); (4) explicit seeding of the augmentation pipeline and DataLoader. Rollout order: quick repro test (25pct, seed 99, identical config run twice, 15 epochs, compare per-epoch val Dice) → implement the above plus a fix to the grid script's stale resume/skip-path check → pilot on 100pct_seed0, 25pct_seed1, 50pct_seed2 (~3h) → full 12-run grid only if the pilot recovers. Both the original and pilot grids are retained in this log as evidence. U-Net result rows remain provisional until the full grid re-run completes. | Corrects two compounding issues found in the Week 3 grid: (a) training length was not comparable across brackets — 10% got ~9 updates/epoch vs. 100%'s ~88, nearly 10x, despite a nominally shared "100 epochs, patience 10" budget; (b) single-point early stopping is too sensitive to one noisy validation read. Verified directly against `logs/train_100pct_seed0.log`: epoch 6's val_dice crash (0.2154) was not accompanied by any train-loss spike (0.9825, in line with the surrounding trend), and epochs 10–19 — rather than burning patience on that one bad round — oscillated between 0.35 and 0.52 for ten straight epochs without ever beating epoch 9's peak (0.5259). A moving-average selection criterion would have carried the run through that noise instead of stopping it at epoch 19. | Balamurali Ragupathi — pending Sebastian Torres sign-off |
| 2026-10-14 | Confirmed fix for the reproducibility gap: pass `seed=<run seed>` to `A.Compose(...)` in `build_train_transform()` (`src/train_unet.py`). albumentations 2.0.8's `Compose.__init__` exposes `seed: int or None = None`; individual transforms (`HorizontalFlip`, etc.) do not take their own seed, so the fix lives at the `Compose` call site, not per-transform. | Repro test (25pct bracket, seed 99, identical config, 15 epochs, `--patience 15` so no early stop cuts either run short) run twice before this fix produced non-identical val_dice from epoch 1 onward and different final bests (0.4324 vs 0.4237 — see `logs/repro_25pct_seed99_run1.log` / `run2.log`). The same test run twice after adding `seed=` to `Compose` produced byte-identical per-epoch train_loss and val_dice across all 15 epochs (`logs/repro_25pct_seed99_run3.log` / `run4.log`, empty diff). Residual epoch-to-epoch volatility (e.g. epoch 6 dipping to 0.2191, epochs 12–13 dipping to 0.28/0.26) persists after the fix and is therefore a real training-dynamics issue, not a seeding artifact — supports the case for the cosine+warmup schedule and moving-average checkpoint selection in the 2026-10-13 entry above. | Balamurali Ragupathi |
| 2026-10-14 | Pilot (100pct_seed0, 25pct_seed1, 50pct_seed2) completed under the new scheme (run_week3_pilot.sh, v2_ run tag, fixed 8,800-step budget / val every 88 steps / 3-validation moving-average selection / cosine+5% warmup / seeded Compose+DataLoader). All three runs recovered cleanly, no failures: 100pct_seed0 0.5259→0.8291, 25pct_seed1 0.7077→0.7974, 50pct_seed2 0.5446→0.8383. ~56 min/run (3,280–3,380s wall clock, device=mps). Checkpoints and profiles under checkpoints/unet_v2_* / results/compute_profiles/unet_compute_profile_v2_*.csv. | Clears the gate set in the 2026-10-13 entry — every piloted bracket, including the one that previously failed outright, now converges under the fixed scheme with no regression. Known open item: the run_week3_grid.sh resume/skip-path bug from 2026-10-12 was sidestepped for the pilot via --run-tag v2_, not actually fixed — needs resolving (or consistently applying the same tag) before the full 12-run grid, or it risks the same silent-retrain/skip failure mode again. | Balamurali Ragupathi — pending Sebastian Torres sign-off |
| 2026-10-15 | Full 12-run grid (run_week3_grid.sh, v2_ run tag) completed — all 12 configs trained + evaluated on the frozen test set via evaluate_unet_grid.py --run-tag v2_ --force (old Oct 7 rows for all 12 run_ids correctly superseded, 3,400 total rows, 2,400 unet / 1,000 sam3, no duplicates). Test-set Dice per bracket: 10pct=0.768, 25pct=0.805, 50pct=0.829, 100pct=0.846 — monotonically increasing (confirmed True, vs. the old grid where 10pct->25pct, 0.479->0.484, was indistinguishable from seed noise). Per-run test Dice, boundary_iou, hd95, precision, recall all recorded in per_image_results.csv. | Clears the "provisional" status set in the 2026-10-13 entry — U-Net result rows are now final under the fixed scheme and safe to use for Week 4 analysis. The data-efficiency curve is now clean and usable for the RQ, which it wasn't under the broken grid. Also validates the training-dynamics fix end to end: the three pilot configs' val_dice (0.8291/0.7974/0.8383) and their test Dice (0.843/0.816/0.839) are consistent with each other and with the other seeds in their brackets — no sign the pilot numbers were a fluke. | Balamurali Ragupathi |

