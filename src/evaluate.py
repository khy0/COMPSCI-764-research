"""Shared evaluation harness — used by BOTH arms.

This is a "non-negotiable" shared file (see src/README.md / decisions.md): changes
require agreement from both teammates, logged in decisions.md.

Takes a directory of predicted masks (one file per test image, produced by either
run_sam3.py or train_unet.py's inference pass) and the frozen test split, scores
every prediction against ground truth using metrics.py, and writes ONE per-image
CSV that both arms' analysis notebooks read from (configs/eval.yaml:
output.per_image_csv). All aggregate statistics (bootstrap CIs, Wilcoxon tests,
stratification) are computed downstream from this CSV in notebooks/02_results_analysis.ipynb
— evaluate.py itself only produces the per-image rows, deliberately, so there is a
single source of truth (configs/eval.yaml's own comment: "all aggregate stats
derive from this").

Usage
-----
    python src/evaluate.py \\
        --data-root data/kvasir-seg \\
        --split-file data/splits/test.txt \\
        --pred-dir results/sam3_preds/box-gt \\
        --arm sam3 --run-id box-gt \\
        --eval-config configs/eval.yaml

Predicted masks are expected as one file per image, named "<image_id>.png",
single-channel, {0, 255} (or any threshold — see --pred-threshold).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import read_split_file  # noqa: E402
from metrics import compute_all  # noqa: E402


def load_mask(path: str, threshold: int = 127) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def load_gt_mask(data_root: str, image_id: str, mask_ext: str = ".jpg") -> np.ndarray:
    path = os.path.join(data_root, "masks", image_id + mask_ext)
    return load_mask(path, threshold=127)


def evaluate_run(
    data_root: str,
    split_file: str,
    pred_dir: str,
    arm: str,
    run_id: str,
    eval_config: dict,
    pred_threshold: int = 127,
    pred_ext: str = ".png",
) -> list[dict]:
    """Score every prediction in `pred_dir` against GT for images in `split_file`.
    Returns a list of per-image result dicts (also the rows written to the CSV)."""
    metric_names = eval_config["metrics"]
    dilation_px = eval_config["boundary_iou"]["dilation_px"]
    if dilation_px is None:
        raise ValueError(
            "configs/eval.yaml: boundary_iou.dilation_px is still null (TODO). "
            "This is an open item in decisions.md — agree a value with Balamurali "
            "before running a real evaluation pass. Pass --dilation-px to override "
            "for a one-off/debug run without editing the config."
        )
    hd95_percentile = eval_config["hd95"]["percentile"]

    image_ids = read_split_file(split_file)
    rows = []
    missing = []
    for image_id in image_ids:
        pred_path = os.path.join(pred_dir, image_id + pred_ext)
        if not os.path.exists(pred_path):
            missing.append(image_id)
            continue
        pred = load_mask(pred_path, threshold=pred_threshold)
        gt = load_gt_mask(data_root, image_id)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred * 255).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST
                )
            )
            pred = (pred > 127).astype(np.uint8)

        m = compute_all(
            pred,
            gt,
            metric_names,
            boundary_dilation_px=dilation_px,
            hd95_percentile=hd95_percentile,
        )
        gt_area_pct = 100.0 * gt.sum() / gt.size
        row = {
            "arm": arm,
            "run_id": run_id,
            "image_id": image_id,
            "gt_area_pct": round(gt_area_pct, 4),
            **{k: round(v, 6) for k, v in m.items()},
        }
        rows.append(row)

    if missing:
        print(
            f"WARNING: {len(missing)}/{len(image_ids)} predictions missing in "
            f"{pred_dir} (e.g. {missing[:5]}) — skipped, not scored as zero.",
            file=sys.stderr,
        )
    return rows


def write_csv(rows: list[dict], out_path: str, append: bool = True) -> None:
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(out_path)
    mode = "a" if (append and file_exists) else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path} (mode={mode}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/kvasir-seg")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--arm", required=True, choices=["sam3", "unet"])
    parser.add_argument(
        "--run-id",
        required=True,
        help="e.g. a SAM3 prompt strategy id (box-gt) or a U-Net bracket+seed (10pct-seed0)",
    )
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--pred-threshold", type=int, default=127)
    parser.add_argument("--pred-ext", default=".png")
    parser.add_argument("--dilation-px", type=int, default=None,
                         help="Override configs/eval.yaml boundary_iou.dilation_px "
                              "for a one-off run (does not edit the config file).")
    parser.add_argument("--out-csv", default=None,
                         help="Defaults to configs/eval.yaml output.per_image_csv")
    parser.add_argument("--no-append", action="store_true",
                         help="Overwrite the output CSV instead of appending.")
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_config = yaml.safe_load(f)

    if args.dilation_px is not None:
        eval_config["boundary_iou"]["dilation_px"] = args.dilation_px

    out_csv = args.out_csv or eval_config["output"]["per_image_csv"]

    rows = evaluate_run(
        data_root=args.data_root,
        split_file=args.split_file,
        pred_dir=args.pred_dir,
        arm=args.arm,
        run_id=args.run_id,
        eval_config=eval_config,
        pred_threshold=args.pred_threshold,
        pred_ext=args.pred_ext,
    )
    write_csv(rows, out_csv, append=not args.no_append)

    if rows:
        for name in eval_config["metrics"]:
            vals = [r[name] for r in rows if np.isfinite(r[name])]
            if vals:
                print(f"{name:>14s}: mean={np.mean(vals):.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
