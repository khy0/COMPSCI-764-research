"""Week 3 Task 1.4 / 1.5 — evaluate all 12 U-Net checkpoints on the frozen test set.

For each of the 12 (bracket, seed) configs from the grid (configs/unet_base.yaml's
seeds: [0, 1, 2] x brackets: 10pct/25pct/50pct/100pct):

  1) Load the BEST checkpoint (checkpoints/unet_<run_name>_best.pt).
  2) Run inference over data/splits/test.txt (the frozen, untouched test set —
     never used for training or early stopping), same pattern as the Week 2
     Task 2 notebook (05_week2_task2_100pct_run.ipynb, cells 8-9): predictions
     are resized from the 256x256 training resolution back to each image's
     NATIVE resolution before saving, so scoring happens in the same "space"
     Sebastian's SAM 3 arm uses (this IS Week 3 Task 2.2, done per-run as a
     byproduct rather than deferred).
  3) Save predicted masks to results/unet_preds/<run_name>/<image_id>.png.
  4) Hand off to the SHARED src/evaluate.py (same harness Sebastian's SAM 3 arm
     uses) to score against ground truth and APPEND the rows to
     results/per_image_results.csv (configs/eval.yaml's single source of
     truth), with arm=unet, run_id=<bracket>_seed<seed>.

Deliberately NOT using --dilation-px like the old Week 2 Task 2 notebook did —
that was a one-off override because configs/eval.yaml's boundary_iou.dilation_px
was still null at the time. It's since been decided (dilation_ratio=0.02, closed
2026-09-30, see decisions.md) and src/evaluate.py reads that from the config by
default now, so every run here uses the real, agreed methodology — this also
means these 12 U-Net boundary_iou numbers will NOT be directly comparable to the
old scratch results/unet_100pct_seed1_per_image.csv file (which used the
--dilation-px 2 override on an older checkpoint) — that file is now stale
(superseded by this run's 100pct_seed1, trained fresh by the Week 3 grid script)
and safe to delete/ignore.

Idempotent: before evaluating a run, checks whether results/per_image_results.csv
already has rows for (arm=unet, run_id=<run_name>) and skips it (so re-running
this script after a partial run doesn't duplicate rows — matters for Week 3 Task
3.2's "no duplicate rows" sanity check). Pass --force to re-evaluate specific
runs anyway (old rows for that run_id are removed first, then replaced).

Usage (from repo root):
    python src/evaluate_unet_grid.py
    python src/evaluate_unet_grid.py --force                  # re-run all 12
    python src/evaluate_unet_grid.py --only 100pct_seed1       # re-run just one
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset import KvasirSegDataset  # noqa: E402
from train_unet import (  # noqa: E402 — reuse the SAME building blocks as training,
    build_eval_transform,  # deliberately not reimplemented here.
    collate_kvasir_samples,
    load_checkpoint,
)
from unet import UNet  # noqa: E402

BRACKETS = ["10pct", "25pct", "50pct", "100pct"]
SEEDS = [0, 1, 2]

PER_IMAGE_CSV = REPO_ROOT / "results" / "per_image_results.csv"


def existing_run_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "arm" not in df.columns:
        return set()
    return set(df.loc[df["arm"] == "unet", "run_id"].unique())


def drop_run_id(csv_path: Path, run_id: str) -> None:
    """Remove any existing arm=unet rows for this run_id (used with --force)."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[~((df["arm"] == "unet") & (df["run_id"] == run_id))]
    removed = before - len(df)
    if removed:
        df.to_csv(csv_path, index=False)
        print(f"  (--force) removed {removed} existing rows for run_id={run_id}")


def run_one(bracket: str, seed: int, data_root: Path, cfg: dict, device: torch.device, force: bool) -> bool:
    run_name = f"{bracket}_seed{seed}"
    ckpt_path = REPO_ROOT / "checkpoints" / f"unet_{run_name}_best.pt"
    if not ckpt_path.exists():
        print(f"[{run_name}] SKIP — no checkpoint at {ckpt_path.relative_to(REPO_ROOT)}")
        return False

    already_done = run_name in existing_run_ids(PER_IMAGE_CSV)
    if already_done and not force:
        print(f"[{run_name}] SKIP — already has rows in results/per_image_results.csv (use --force to redo)")
        return True
    if already_done and force:
        drop_run_id(PER_IMAGE_CSV, run_name)

    print(f"[{run_name}] loading checkpoint and running inference on test set...")
    resize = tuple(cfg["input"]["resize"])
    model = UNet(
        in_channels=3, out_channels=1,
        base_channels=cfg["model"]["base_channels"], depth=cfg["model"]["depth"],
    ).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
    model.eval()

    test_split = REPO_ROOT / "data" / "splits" / "test.txt"
    test_dataset = KvasirSegDataset(str(data_root), str(test_split), resize=resize, transform=build_eval_transform())
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_kvasir_samples)

    pred_dir = REPO_ROOT / "results" / "unet_preds" / run_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
            for image_id, prob in zip(batch["image_id"], probs):
                gt_path = data_root / "masks" / f"{image_id}.jpg"
                gt_native_shape = np.array(Image.open(gt_path).convert("L")).shape
                # Upsample the CONTINUOUS probability map bilinearly to native resolution
                # FIRST, then threshold — avoids blocky NEAREST edges from thresholding at
                # 256px. Per Sebastian's PR #4 review comment.
                prob_img = Image.fromarray(prob).resize(
                    (gt_native_shape[1], gt_native_shape[0]), Image.BILINEAR
                )
                pred_native = (np.array(prob_img) > 0.5).astype(np.uint8) * 255
                pred_img = Image.fromarray(pred_native)
                pred_img.save(pred_dir / f"{image_id}.png")
                n_written += 1
    print(f"[{run_name}] wrote {n_written} predicted masks -> {pred_dir.relative_to(REPO_ROOT)}")

    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "evaluate.py"),
        "--data-root", str(data_root),
        "--split-file", str(test_split),
        "--pred-dir", str(pred_dir),
        "--arm", "unet",
        "--run-id", run_name,
        "--eval-config", str(REPO_ROOT / "configs" / "eval.yaml"),
        # No --dilation-px override and no --no-append / --out-csv: use the real,
        # decided methodology (dilation_ratio=0.02) and append to the shared
        # results/per_image_results.csv, same as Sebastian's SAM3 rows.
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"[{run_name}] evaluate.py FAILED — see stderr above.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if rows already exist.")
    parser.add_argument("--only", default=None, help="Run just one, e.g. --only 100pct_seed1")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "unet_base.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root_candidates = [REPO_ROOT / "data" / "kvasir-seg", REPO_ROOT / "data" / "kvasir-seg" / "Kvasir-SEG"]
    data_root = next((c for c in data_root_candidates if (c / "images").is_dir()), None)
    if data_root is None:
        raise FileNotFoundError(f"Couldn't find data/kvasir-seg/images under any of {data_root_candidates}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    if args.only:
        bracket, seed_part = args.only.split("_seed")
        combos = [(bracket, int(seed_part))]
    else:
        combos = [(b, s) for b in BRACKETS for s in SEEDS]

    results_ok = []
    for bracket, seed in combos:
        ok = run_one(bracket, seed, data_root, cfg, device, force=args.force)
        results_ok.append((f"{bracket}_seed{seed}", ok))
        print()

    failed = [r for r, ok in results_ok if not ok]
    print("=" * 70)
    if failed:
        print(f"{len(failed)} run(s) did not complete: {failed}")
    else:
        print("All requested runs evaluated (or already were).")

    # Task 1.5 — tabulate mean test-set Dice per bracket, sanity-check the trend.
    if PER_IMAGE_CSV.exists():
        df = pd.read_csv(PER_IMAGE_CSV)
        unet_df = df[df["arm"] == "unet"].copy()
        unet_df["bracket"] = unet_df["run_id"].str.split("_seed").str[0]
        unet_df["seed"] = unet_df["run_id"].str.split("_seed").str[1]

        print("\nTest-set Dice per run (mean over images in that run):")
        per_run = unet_df.groupby("run_id")["dice"].agg(["mean", "std", "count"])
        print(per_run.to_string())

        print("\nTest-set Dice per bracket (mean over all seeds+images in that bracket):")
        per_bracket = unet_df.groupby("bracket")["dice"].mean().reindex(BRACKETS)
        print(per_bracket.to_string())

        increasing = per_bracket.is_monotonic_increasing
        print(f"\nMonotonic increase 10pct -> 100pct: {increasing}")
        if not increasing:
            print("WARNING: bracket ordering by Dice is NOT strictly increasing — "
                  "expected (per Week 3 Task 1.5), eyeball before trusting downstream analysis.")
    else:
        print("results/per_image_results.csv not found — nothing to summarize.")


if __name__ == "__main__":
    main()
