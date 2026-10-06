"""U-Net training loop — Week 2 Task 1 (augmentation, early stopping, checkpointing,
seeding). Also the script Week 3's grid loops over 12 times (4 brackets x 3 seeds),
per the tracker's own note under Week 3 Task 1.2 ("launch train_unet.py for each
(bracket, seed) pair").

Usage:
    python src/train_unet.py --bracket 100pct --seed 0
    python src/train_unet.py --bracket 10pct --seed 1 --max-epochs 150 --patience 10

Config is read from configs/unet_base.yaml (model/input/training/augmentation/
brackets) — CLI flags override only the things that are genuinely per-run
(bracket, seed) or still open TODOs in that config (max-epochs, patience), so all
future Week 3 runs stay comparable (P3 — matched training budget across brackets).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

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

def build_train_transform() -> A.Compose:
    """Flips, rotation, scale jitter, colour jitter — identical across all 4
    brackets (configs/unet_base.yaml augmentation section, P3: do not vary by
    bracket). Geometric ops (flip/rotate/affine) move image AND mask together;
    ColorJitter/Normalize are image-only in albumentations by design."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=35, p=0.5),
            A.Affine(scale=(0.85, 1.15), p=0.5),  # scale jitter, output size unchanged
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform() -> A.Compose:
    """No augmentation — used for validation (and later, test-set scoring)."""
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

def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_dice: float, seed: int, bracket: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
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
# Main training loop
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train the U-Net arm on one (bracket, seed) config.")
    parser.add_argument("--bracket", required=True, choices=["10pct", "25pct", "50pct", "100pct"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "unet_base.yaml"))
    parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Safety cap on epochs. configs/unet_base.yaml's training.max_iterations "
             "is still an open TODO (decisions.md) — this flag lets you run/observe "
             "before that number is fixed (Week 2 Task 2).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs with no val-Dice improvement. "
             "configs/unet_base.yaml's training.patience is still null (open TODO) — "
             "10 is a common default; confirm/change it and log your choice in "
             "decisions.md yourself (Task 1.2 asks you to 'pick and document N').",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)  # Task 1.4 — before anything else touches randomness

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    train_dataset = KvasirSegDataset(str(data_root), str(train_split), resize=resize, transform=build_train_transform())
    val_dataset = KvasirSegDataset(str(data_root), str(val_split), resize=resize, transform=build_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_kvasir_samples)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_kvasir_samples)

    model = UNet(in_channels=3, out_channels=1,
                 base_channels=cfg["model"]["base_channels"], depth=cfg["model"]["depth"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    run_name = f"{args.bracket}_seed{args.seed}"
    checkpoints_dir = REPO_ROOT / "checkpoints"
    best_ckpt_path = checkpoints_dir / f"unet_{run_name}_best.pt"
    last_ckpt_path = checkpoints_dir / f"unet_{run_name}_last.pt"

    best_val_dice = -1.0
    epochs_without_improvement = 0
    history = []

    print(f"Run: bracket={args.bracket} seed={args.seed} device={device} "
          f"train_images={len(train_dataset)} val_images={len(val_dataset)}")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        val_dice = evaluate_val_dice(model, val_loader, device)
        epoch_time = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice, "time_s": epoch_time})

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_dice, args.seed, args.bracket)
        else:
            epochs_without_improvement += 1

        # Always overwrite the "last" checkpoint too — a resume-from-crash safety
        # net for Colab disconnects (Task 1.3), separate from the best-so-far one.
        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_dice, args.seed, args.bracket)

        marker = "*" if improved else " "
        print(f"Epoch {epoch:3d}/{args.max_epochs}  train_loss={train_loss:.4f}  "
              f"val_dice={val_dice:.4f}{marker}  best={best_val_dice:.4f}  "
              f"no_improve={epochs_without_improvement}  time={epoch_time:.1f}s")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val_dice improvement for {args.patience} epochs.")
            break

    print(f"\nDone. Best val_dice={best_val_dice:.4f} -> {best_ckpt_path.relative_to(REPO_ROOT)}")
    return history, best_val_dice


if __name__ == "__main__":
    main()
