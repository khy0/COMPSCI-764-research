"""SAM 3 zero-shot PVS-mode inference over the (frozen) test split.

Owner: Sebastian (per src/README.md).

Uses the "SAM 1 task" / interactive-instance-segmentation API confirmed from the
official facebookresearch/sam3 repo (examples/sam3_for_sam1_task_example.ipynb),
NOT the text-prompt (PCS) API in sam3_image_predictor_example.ipynb — this
project's RQ is specifically about promptable *visual* (point/box) segmentation,
so PCS's `set_text_prompt` is out of scope entirely (see references.bib / lit
review, Related Work D).

Confirmed call chain (verified in this repo, see decisions.md Week-1 entry):

    model = build_sam3_image_model(bpe_path=..., enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    state = processor.set_image(pil_image)
    masks, scores, logits = model.predict_inst(
        state, point_coords=..., point_labels=..., box=..., multimask_output=...
    )
    # masks: (n_masks, H, W) bool array at native image resolution

For a box prompt we use multimask_output=False (a box is much less ambiguous than
a single point — matches the official notebook's own guidance). For point
prompts we use multimask_output=True and keep the highest-scoring mask, exactly
as the official notebook recommends for ambiguous prompts.

Environment note (Week 1, logged in decisions.md)
--------------------------------------------------
Running this script for real requires (a) a GPU, and (b) HuggingFace access
approved for the gated `facebook/sam3` checkpoint. Neither was available in the
sandbox this was developed in (no GPU driver; huggingface.co and
datasets.simula.no are both outside that sandbox's network allowlist). This
script was still verified end-to-end there by building the model with random
(untrained) weights (`checkpoint_path=None, load_from_HF=False`) on CPU and
confirming the full call chain above runs and returns correctly-shaped masks
for both a box and a point prompt against a real image — i.e. the *code* is
confirmed correct against the real API; only weight-loaded, GPU-executed,
real-polyp-image inference remains to be run once on a machine that has both.
That run (this script, unmodified, with --device cuda and real HF access) is
the actual Week-1 gate deliverable and should happen on Sebastian's or the
university's GPU machine.

Week 2 additions (logged in decisions.md)
------------------------------------------
- Full test-set x all-5-strategies loop (this was already structurally here
  from Week 1 — `--limit` was only ever a debug knob; omit it for the real run).
- Compute-profiling hooks (src/profiling.py): per-image latency + peak memory,
  plus a one-off model parameter count, written to
  results/sam3_compute_profile.csv. See src/profiling.py's docstring for the
  CPU-vs-CUDA memory-measurement caveat before quoting these numbers in the
  report — they mean different things on the two devices.
- `--evaluate` now optionally calls the shared `evaluate.py` harness directly
  after inference for each strategy, so one command produces both the
  predicted masks and the appended rows in results/per_image_results.csv (the
  same file `evaluate.py` run standalone would produce — no duplicated scoring
  logic, this just saves a second manual CLI step per strategy).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import KvasirSegDataset  # noqa: E402
from prompts import generate_all_prompts  # noqa: E402
from profiling import ComputeProfiler, count_params  # noqa: E402


def build_model(
    checkpoint_path: str | None, device: str, load_from_hf: bool = True
):
    """Thin wrapper so the rest of this file doesn't need to import sam3 unless
    this function is actually called (keeps `python -m pytest src/` importable
    even where sam3/torch aren't installed, e.g. Balamurali's machine before
    Week 4)."""
    from sam3 import build_sam3_image_model

    bpe_path = _find_bpe_path()
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf and checkpoint_path is None,
        enable_inst_interactivity=True,
    )
    return model


def _find_bpe_path() -> str:
    import sam3

    root = os.path.dirname(sam3.__file__)
    candidates = [
        os.path.join(root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Could not locate bpe_simple_vocab_16e6.txt.gz inside the installed sam3 "
        f"package (checked: {candidates}). Is sam3 installed from source "
        "(pip install git+https://github.com/facebookresearch/sam3.git)?"
    )


def predict_one(model, state, prompt) -> np.ndarray:
    """Run predict_inst for a single Prompt (from prompts.py) against an
    already-embedded image `state`, and return the single best binary mask
    (H, W) uint8 {0,1}, at native image resolution.

    IMPORTANT: `state` must come from a single `processor.set_image(image)`
    call made ONCE per image by the caller (see `run()`), not once per prompt.
    Found during Week 2 compute-profiling: an earlier version of this function
    called `set_image()` internally, so it re-ran the (expensive) image
    encoder on every one of the 5 prompt-strategy calls per image instead of
    once — ~5x wasted encoder compute on the full test set, and it also
    corrupted what the per-strategy latency numbers actually meant (the
    profiled time was ~97% redundant re-encoding, ~3% the actual
    prompt-conditioned decode step the profiling was supposed to measure).
    `run()` now profiles the one-time `set_image()` embedding cost separately
    (run_id="embed") from each strategy's `predict_inst()` decode cost."""
    if prompt.box is not None:
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=None,
            point_labels=None,
            box=prompt.box[None, :],
            multimask_output=False,
        )
        best = masks[0]
    else:
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=prompt.points,
            point_labels=prompt.point_labels,
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))]

    return (best > 0).astype(np.uint8)


def run(
    data_root: str,
    test_split_file: str,
    sam3_config_path: str,
    out_dir: str,
    device: str,
    checkpoint_path: str | None,
    limit: int | None = None,
    profile_csv: str | None = "results/sam3_compute_profile.csv",
    evaluate_after: bool = False,
    eval_config_path: str = "configs/eval.yaml",
    dilation_px: int | None = None,
    per_image_csv: str | None = None,
) -> None:
    with open(sam3_config_path) as f:
        sam3_cfg = yaml.safe_load(f)

    strategies = sam3_cfg["prompts"]["strategies"]
    base_seed = sam3_cfg["prompts"]["seed"]

    dataset = KvasirSegDataset(
        data_root=data_root, split_file=test_split_file, resize=None, transform=None
    )
    if limit is not None:
        dataset.image_ids = dataset.image_ids[:limit]

    print(f"Building SAM 3 model on device={device} "
          f"(checkpoint={checkpoint_path or 'download from HF: facebook/sam3'})...")
    model = build_model(checkpoint_path, device=device)
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model, device=device)
    print("Model ready.")

    param_counts = count_params(model)
    print(f"Model params: {param_counts['n_params_total']:,} total "
          f"({param_counts['n_params_trainable']:,} trainable — should be 0, "
          f"this is a frozen zero-shot model).")
    profiler = ComputeProfiler(arm="sam3", device=device, param_counts=param_counts)

    for strategy in strategies:
        os.makedirs(os.path.join(out_dir, strategy["id"]), exist_ok=True)

    for i, sample in enumerate(dataset):
        # Encode the image ONCE per image (this is the expensive step) and
        # reuse the resulting `state` for all 5 prompt strategies below — see
        # predict_one()'s docstring for why this matters. Profiled separately
        # (run_id="embed") from each strategy's decode cost so the per-strategy
        # latency numbers reflect only the prompt-conditioned decode step.
        with profiler.measure(run_id="embed", image_id=sample.image_id):
            state = processor.set_image(sample.image)

        prompts = generate_all_prompts(strategies, sample.mask, sample.image_id, base_seed)
        for strategy_id, prompt in prompts.items():
            with profiler.measure(run_id=strategy_id, image_id=sample.image_id):
                pred = predict_one(model, state, prompt)
            out_path = os.path.join(out_dir, strategy_id, sample.image_id + ".png")
            Image.fromarray(pred * 255).save(out_path)
        print(f"[{i + 1}/{len(dataset)}] {sample.image_id}: "
              f"{len(prompts)} strategies done")

    print(f"Done. Predictions written under {out_dir}/<strategy_id>/<image_id>.png")

    if profile_csv:
        profiler.write_csv(profile_csv)
        summary = profiler.summary()
        print(f"Profiling summary: {summary}")
        print("NOTE (configs/eval.yaml compute_profiling.same_hardware_required): "
              "these numbers are only comparable against the U-Net arm's profiling "
              "if both were collected on the same machine.")

    if evaluate_after:
        # Reuses the SAME shared harness (evaluate.py) that a standalone
        # `python src/evaluate.py ...` call would use — no scoring logic is
        # duplicated here, this just automates calling it once per strategy.
        import evaluate as evaluate_module

        with open(eval_config_path) as f:
            eval_cfg = yaml.safe_load(f)
        if dilation_px is not None:
            eval_cfg["boundary_iou"]["dilation_px"] = dilation_px
        out_csv = per_image_csv or eval_cfg["output"]["per_image_csv"]

        for strategy in strategies:
            strategy_id = strategy["id"]
            pred_dir = os.path.join(out_dir, strategy_id)
            print(f"Evaluating strategy '{strategy_id}' via evaluate.py -> {out_csv} ...")
            rows = evaluate_module.evaluate_run(
                data_root=data_root,
                split_file=test_split_file,
                pred_dir=pred_dir,
                arm="sam3",
                run_id=strategy_id,
                eval_config=eval_cfg,
            )
            evaluate_module.write_csv(rows, out_csv, append=True)
    else:
        print("Next: src/evaluate.py --pred-dir "
              f"{out_dir}/<strategy_id> --arm sam3 --run-id <strategy_id> "
              "(or re-run with --evaluate to do this automatically for all 5 strategies)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kvasir-seg")
    parser.add_argument("--sam3-config", default="configs/sam3_pvs.yaml")
    parser.add_argument("--out-dir", default="results/sam3_preds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default=None,
                         help="Local path to a downloaded sam3.pt. If omitted, "
                              "downloads from the gated facebook/sam3 HF repo "
                              "(requires prior HF access approval + login).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N test images (debug/smoke run "
                              "— use --limit 1 for the Week-1 gate 'one image' check; "
                              "omit entirely for the real Week-2 full-test-set run).")
    parser.add_argument("--profile-csv", default="results/sam3_compute_profile.csv",
                         help="Where to write per-image latency/memory/param-count "
                              "rows. Pass an empty string to disable profiling.")
    parser.add_argument("--evaluate", action="store_true",
                         help="Also call the shared evaluate.py harness for each "
                              "strategy right after its predictions are written, "
                              "appending to results/per_image_results.csv.")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--dilation-px", type=int, default=None,
                         help="Force a FIXED boundary-band width in pixels for "
                              "--evaluate runs, instead of the default per-image "
                              "ratio-based width (configs/eval.yaml: "
                              "boundary_iou.dilation_ratio, 0.02 by default). "
                              "Normally leave this unset.")
    parser.add_argument("--per-image-csv", default=None,
                         help="Override configs/eval.yaml output.per_image_csv "
                              "for --evaluate runs.")
    args = parser.parse_args()

    with open(args.sam3_config) as f:
        sam3_cfg = yaml.safe_load(f)
    test_split_file = sam3_cfg["inference"]["test_split_file"]

    run(
        data_root=args.data_root,
        test_split_file=test_split_file,
        sam3_config_path=args.sam3_config,
        out_dir=args.out_dir,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        limit=args.limit,
        profile_csv=args.profile_csv or None,
        evaluate_after=args.evaluate,
        eval_config_path=args.eval_config,
        dilation_px=args.dilation_px,
        per_image_csv=args.per_image_csv,
    )


if __name__ == "__main__":
    main()
