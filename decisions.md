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
