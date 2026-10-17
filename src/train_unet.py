"""U-Net training loop — Week 2 Task 1 (augmentation, early stopping, checkpointing,
seeding) + Week 2 Task 4 (compute profiling hooks). Also the script Week 3's grid
loops over 12 times (4 brackets x 3 seeds), per the tracker's own note under Week 3
Task 1.2 ("launch train_unet.py for each (bracket, seed) pair").

Usage:
    python src/train_unet.py --bracket 100pct --seed 0
    python src/train_unet.py --bracket 10pct --seed 1 --max-epochs 150 --patience 10

Config is read from configs/unet_base.yaml (model/input/training/augmentation/
brackets) — CLI flags override only the things that are genuinely per-run
(bracket, seed) or still open TODOs in that config (max-epochs, patience), so all
future Week 3 runs stay comparable (P3 — matched training budget across brackets).

Compute profiling (Week 2 Task 4): reuses Sebastian's src/profiling.py
ComputeProfiler/count_params (written arm-agnostic for exactly this — see that
file's own docstring), so U-Net and SAM3 numbers are captured the same way
(configs/eval.yaml: compute_profiling.same_hardware_required — profile both arms
on the same machine, same session, nothing else competing for CPU/GPU at the time).
One row covers the WHOLE training run (run_id="train_<bracket>_seed<seed>",
total wall-clock + peak memory + param counts — matches Task 4.1's literal ask for
a single post-training number, not per-epoch). A second block of rows, one per test
image (run_id="infer_<bracket>_seed<seed>"), covers per-image inference latency
(Task 4.2's "mean ± std" ask is computed from these afterward).

IMPORTANT gotcha (flag for Sebastian, Task 4.3's "confirm... aligned with
Sebastian's"): ComputeProfiler.write_csv() always opens its output in "w" mode —
it does NOT append like evaluate.py's write_csv does. Call it once per run with
the SAME path and every earlier run's profiling rows are silently gone. Rather
than touch src/profiling.py unilaterally (it's flagged in decisions.md as "a
shared proposal, not a shared fact" until both arms have used it), this script
works around it the same way Week 1 worked around the dataset.py collate gap:
locally, without editing the shared file. Each run gets ITS OWN profiling CSV,
named by bracket+seed (results/unet_compute_profile_<bracket>_seed<seed>.csv) —
combine them (and with results/sam3_compute_profile.csv) in Week 4's
compute/amortisation notebook, the same way Week 3's "assemble master results
CSV" task already does for per-image scores. Mention the append gap to Sebastian
before Week 3's 12-run grid, in case the fix belongs in profiling.py itself.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
import math

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset import KvasirSegDataset  # noqa: E402
from metrics import dice as dice_metric  # noqa: E402 — reuse the SAME shared metric
from profiling import ComputeProfiler, count_params  # noqa: E402 — Sebastian's, reused as-is
from unet import UNet  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Task 1.4 — Seeding
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Seed everything that introduces randomness into a run: shuffling, weight
    init, augmentation choices. Same seed -> same result twice (best-effort on
    GPU; a few CUDA ops stay nondeterministic even with these flags set)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Task 1.1 — Mask-aware augmentation pipeline
# --------------------------------------------------------------------------- #

def build_train_transform(seed: int) -> A.Compose:
    """Flips, rotation, scale jitter, colour jitter — identical across all 4
    brackets (configs/unet_base.yaml augmentation section, P3: do not vary by
    bracket). Geometric ops (flip/rotate/affine) move image AND mask together;
    ColorJitter/Normalize are image-only in albumentations by design.
    `seed` makes this Compose's own RNG (and the child transforms it draws
    augmentation params from) deterministic — closes the reproducibility gap
    from Phase 3/decisions.md: previously only random/np.random/torch were
    seeded, not albumentations' own generator."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=35, p=0.5),
            A.Affine(scale=(0.85, 1.15), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        seed=seed,
    )

def build_eval_transform() -> A.Compose:
    """No augmentation — used for validation and test-set inference/profiling."""
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


# --------------------------------------------------------------------------- #
# DataLoader plumbing — same gap/fix as the Week 1 Task 5 smoke test:
# dataset.py's KvasirSample dataclass isn't batchable by the default collate_fn.
# --------------------------------------------------------------------------- #

def collate_kvasir_samples(batch: list) -> dict:
    images = torch.stack([s.image for s in batch], dim=0)
    masks = torch.stack([s.mask for s in batch], dim=0)
    image_ids = [s.image_id for s in batch]
    return {"image": images, "mask": masks, "image_id": image_ids}


# --------------------------------------------------------------------------- #
# Loss (same combined BCE + soft-Dice as the Week 1 Task 5 smoke test)
# --------------------------------------------------------------------------- #

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(1)
    target_flat = target.flatten(1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1.0 - ((2 * intersection + eps) / (denom + eps)).mean()


bce_loss = nn.BCEWithLogitsLoss()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return bce_loss(logits, target) + dice_loss(logits, target)


# --------------------------------------------------------------------------- #
# Task 1.2 — Validation Dice, using src/metrics.py's dice() directly — the SAME
# function evaluate.py uses for the real scoring, so "how's training going" and
# "how good is the final model" are never measured by two different formulas.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_val_dice(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dice_scores = []
    for batch in val_loader:
        images = batch["image"].to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
        gts = batch["mask"].numpy().astype(np.uint8)
        for pred, gt in zip(preds, gts):
            dice_scores.append(dice_metric(pred, gt))
    model.train()
    return float(np.mean(dice_scores))


# --------------------------------------------------------------------------- #
# Task 1.3 — Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(path: Path, model, optimizer, step: int, best_val_dice: float, seed: int, bracket: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_dice": best_val_dice,
            "seed": seed,
            "bracket": bracket,
        },
        path,
    )

def load_checkpoint(path: Path, model, optimizer=None, map_location=None) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# --------------------------------------------------------------------------- #
# Task 4.2 — Inference-time profiling: one profiled forward pass per test image.
# Batch size 1 deliberately, for a clean per-image latency number (mirrors how
# src/run_sam3.py profiles SAM3 one image at a time).
# --------------------------------------------------------------------------- #

@torch.no_grad()
def profile_inference(
    model: nn.Module,
    test_dataset,
    device: torch.device,
    profiler: ComputeProfiler,
    run_id: str,
) -> None:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_kvasir_samples)
    for batch in loader:
        image = batch["image"].to(device)
        image_id = batch["image_id"][0]
        with profiler.measure(run_id=run_id, image_id=image_id):
            _ = model(image)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train the U-Net arm on one (bracket, seed) config.")
    parser.add_argument("--bracket", required=True, choices=["10pct", "25pct", "50pct", "100pct"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "unet_base.yaml"))
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Total training steps. Defaults to configs/unet_base.yaml's training.total_steps.",
    )
    parser.add_argument(
        "--val-every-steps", type=int, default=None,
        help="Validate every N steps. Defaults to configs/unet_base.yaml's training.val_every_steps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--test-split", default=str(REPO_ROOT / "data" / "splits" / "test.txt"),
        help="Used only for Task 4.2 inference-time profiling, not for scoring.",
    )
    parser.add_argument(
        "--profile-csv", default=None,
        help="Defaults to results/compute_profiles/unet_compute_profile_<tag><bracket>_seed<seed>.csv "
             "(one file per run — see module docstring on why it's not a single "
             "shared file: ComputeProfiler.write_csv() overwrites, it doesn't append).",
    )
    parser.add_argument(
        "--skip-inference-profile", action="store_true",
        help="Skip the Task 4.2 test-set inference profiling pass (training-time "
             "profiling from Task 4.1 always runs). Useful for a fast debug run.",
    )
    parser.add_argument(
        "--run-tag", default="",
        help="Prefix for checkpoint/log/profile filenames (e.g. 'v2_'), so this "
             "scheme's runs can't collide with the old grid's artifacts under the "
             "same (bracket, seed) and get silently skipped by the resume check.",
    )
    args = parser.parse_args()

    set_seed(args.seed)  # Task 1.4 — before anything else touches randomness

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total_steps = args.total_steps if args.total_steps is not None else cfg["training"]["total_steps"]
    val_every = args.val_every_steps if args.val_every_steps is not None else cfg["training"]["val_every_steps"]
    window = cfg["training"]["selection_window"]
    warmup_steps = int(cfg["training"]["warmup_pct"] * total_steps)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.device:
        device = torch.device(args.device)

    data_root_candidates = [REPO_ROOT / "data" / "kvasir-seg", REPO_ROOT / "data" / "kvasir-seg" / "Kvasir-SEG"]
    data_root = next((c for c in data_root_candidates if (c / "images").is_dir()), None)
    if data_root is None:
        raise FileNotFoundError(f"Couldn't find data/kvasir-seg/images under any of {data_root_candidates}")

    bracket_cfg = next(b for b in cfg["brackets"] if b["name"] == args.bracket)
    train_split = REPO_ROOT / bracket_cfg["split_file"]
    val_split = REPO_ROOT / "data" / "splits" / "val.txt"  # configs/data.yaml splits.val.file — frozen, same every run

    resize = tuple(cfg["input"]["resize"])

    train_dataset = KvasirSegDataset(str(data_root), str(train_split), resize=resize,
                                      transform=build_train_transform(args.seed))
    val_dataset = KvasirSegDataset(str(data_root), str(val_split), resize=resize, transform=build_eval_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_kvasir_samples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_kvasir_samples)

    model = UNet(in_channels=3, out_channels=1,
                 base_channels=cfg["model"]["base_channels"], depth=cfg["model"]["depth"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_name = f"{args.run_tag}{args.bracket}_seed{args.seed}"
    checkpoints_dir = REPO_ROOT / "checkpoints"
    best_ckpt_path = checkpoints_dir / f"unet_{run_name}_best.pt"
    last_ckpt_path = checkpoints_dir / f"unet_{run_name}_last.pt"

    profile_csv = args.profile_csv or str(REPO_ROOT / "results" / "compute_profiles" / f"unet_compute_profile_{run_name}.csv")

    # Task 4.1 — trainable parameter count, via Sebastian's count_params() so the
    # number is computed identically on both arms (same function, same definition
    # of "trainable").
    param_counts = count_params(model)
    print(f"Params: total={param_counts['n_params_total']:,}  trainable={param_counts['n_params_trainable']:,}")

    profiler = ComputeProfiler(arm="unet", device=str(device), param_counts=param_counts)

    val_history = []          # list of (step, val_dice)
    best_moving_avg = -1.0
    running_loss = 0.0
    n_batches_since_val = 0
    train_iter = iter(train_loader)

    print(f"Run: bracket={args.bracket} seed={args.seed} device={device} "
          f"train_images={len(train_dataset)} val_images={len(val_dataset)} "
          f"total_steps={total_steps} val_every={val_every}")

    # Task 4.1 — wrap the WHOLE training run in one profiler call: total
    # wall-clock + peak memory for the run, not per-epoch (profiler.measure()
    # resets CUDA's peak-memory counter on entry and reads it back on exit, so
    # this is a true peak-over-the-whole-run number on CUDA; on CPU/MPS it's the
    # process-lifetime peak RSS — see src/profiling.py's own docstring caveat).
    with profiler.measure(run_id=f"train_{run_name}", image_id="all_steps"):
        t0 = time.perf_counter()
        for step in range(1, total_steps + 1):
            model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images = batch["image"].to(device)
            targets = batch["mask"].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches_since_val += 1

            if step % val_every == 0:
                train_loss = running_loss / n_batches_since_val
                val_dice = evaluate_val_dice(model, val_loader, device)
                val_history.append((step, val_dice))
                moving_avg = float(np.mean([d for _, d in val_history[-window:]]))
                interval_time = time.perf_counter() - t0

                improved = moving_avg > best_moving_avg
                if improved:
                    best_moving_avg = moving_avg
                    save_checkpoint(best_ckpt_path, model, optimizer, step, best_moving_avg, args.seed, args.bracket)
                save_checkpoint(last_ckpt_path, model, optimizer, step, best_moving_avg, args.seed, args.bracket)

                marker = "*" if improved else " "
                print(f"Step {step:5d}/{total_steps}  train_loss={train_loss:.4f}  "
                      f"val_dice={val_dice:.4f}  moving_avg({window})={moving_avg:.4f}{marker}  "
                      f"best_avg={best_moving_avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"time={interval_time:.1f}s")

                running_loss = 0.0
                n_batches_since_val = 0
                t0 = time.perf_counter()

    # profiler.records has exactly one row at this point (the "train_..." row
    # from the `with` block just above) — read it directly rather than via
    # profiler.summary(), which aggregates ALL records and would silently
    # include the per-image inference rows too once those are added below.
    train_record = profiler.records[-1]
    print(f"\nDone. Best moving-avg val_dice={best_moving_avg:.4f} -> {best_ckpt_path.relative_to(REPO_ROOT)}")
    print(f"Training profile: total_wall_clock_s={train_record.latency_ms / 1000:.1f}  "
          f"peak_mem_mb={train_record.peak_mem_mb:.1f}  "
          f"(device={device}; see src/profiling.py docstring for the CPU/MPS peak-memory caveat)")

    # Task 4.2 — per-image inference-time profiling on the test set, using the
    # BEST checkpoint (the one that'll actually get used for evaluation/results),
    # reloaded fresh rather than whatever's left in memory after early stopping.
    if not args.skip_inference_profile:
        test_dataset = KvasirSegDataset(str(data_root), args.test_split, resize=resize, transform=build_eval_transform())
        load_checkpoint(best_ckpt_path, model, optimizer=None, map_location=device)
        infer_run_id = f"infer_{run_name}"
        profile_inference(model, test_dataset, device, profiler, run_id=infer_run_id)

        infer_latencies = [r.latency_ms for r in profiler.records if r.run_id == infer_run_id]
        infer_mems = [r.peak_mem_mb for r in profiler.records if r.run_id == infer_run_id]
        print(f"Inference profile ({len(infer_latencies)} test images): "
              f"latency_ms mean={np.mean(infer_latencies):.2f} std={np.std(infer_latencies):.2f}  "
              f"peak_mem_mb mean={np.mean(infer_mems):.1f}")
    else:
        print("Skipped Task 4.2 inference profiling (--skip-inference-profile).")

    profiler.write_csv(profile_csv)
    print(f"Wrote compute profile to {Path(profile_csv).relative_to(REPO_ROOT)} "
          f"(this run only — see module docstring: write_csv() does not append).")

    return val_history, best_moving_avg


if __name__ == "__main__":
    main()