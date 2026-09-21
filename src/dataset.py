"""Shared Kvasir-SEG dataset loader — used by BOTH the SAM 3 arm and the U-Net arm.

This is a "non-negotiable" shared file (see src/README.md / decisions.md): changes
require agreement from both teammates.

Design notes
------------
- Reads split membership from the frozen index files under data/splits/ (one image
  ID per line, no extension). Never re-derives a split by re-shuffling — that is
  `make_splits.py`'s job, run exactly once (planning doc, P1).
- Returns full-resolution PIL images + numpy masks by default (`resize=None`),
  because the SAM 3 arm needs the original resolution to build point/box prompts in
  image coordinates and to score predictions against full-resolution ground truth.
  The U-Net arm asks for a fixed `resize` (see configs/unet_base.yaml, input.resize)
  and gets resized/augmented tensors instead — same underlying image/mask pairs,
  same split files, two different `__getitem__` return shapes controlled by the
  constructor arguments below.
- Augmentation (flip/rotation/scale-jitter/color-jitter) is applied only when a
  `transform` is supplied. SAM 3 zero-shot inference must NEVER receive an
  augmented image (there is nothing to train, and the GT mask used to build the
  oracle prompt must stay pixel-aligned with the image) — call sites for the SAM 3
  arm (run_sam3.py) must construct this dataset with transform=None.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class KvasirSample:
    image_id: str
    image: object  # PIL.Image (transform=None) or torch.Tensor (transform given)
    mask: object  # np.ndarray[H,W] uint8 {0,1} (transform=None) or torch.Tensor


def read_split_file(split_path: str) -> list[str]:
    """Read a frozen split index file: one image ID per line, blank lines/comments ignored."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Frozen splits are generated once by src/make_splits.py and committed "
            f"under data/splits/ — run that script first (see data/splits/README.md)."
        )
    ids = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


class KvasirSegDataset(Dataset):
    """Kvasir-SEG image/mask pairs restricted to a given frozen split file.

    Parameters
    ----------
    data_root : str
        Path to data/kvasir-seg/ (expects images/ and masks/ subdirectories,
        see data/kvasir-seg/README.md for the expected layout and filenames).
    split_file : str
        Path to a frozen split index file under data/splits/ (e.g. "test.txt",
        "train_25.txt").
    image_ext, mask_ext : str
        File extensions for images/masks. Kvasir-SEG ships both as .jpg.
    resize : Optional[tuple[int, int]]
        If given, (H, W) to resize image+mask to before any transform is applied.
        Leave None for the SAM 3 arm (needs native resolution).
    transform : Optional[Callable]
        An albumentations-style callable taking `image=np.ndarray, mask=np.ndarray`
        and returning a dict with the same keys, already resized/augmented/tensored.
        Leave None for zero-shot SAM 3 inference (no augmentation, ever).
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        image_ext: str = ".jpg",
        mask_ext: str = ".jpg",
        resize: Optional[tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.resize = resize
        self.transform = transform

        self.image_ids = read_split_file(split_file)
        if not self.image_ids:
            raise ValueError(f"Split file {split_file} is empty.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_pair(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        img_path = os.path.join(self.images_dir, image_id + self.image_ext)
        mask_path = os.path.join(self.masks_dir, image_id + self.mask_ext)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image {img_path} not found — is data/kvasir-seg/ populated? "
                f"See data/kvasir-seg/README.md."
            )
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask {mask_path} not found for image {image_id}.")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        # Kvasir-SEG masks are near-binary JPEGs (compression artifacts introduce
        # a few intermediate gray values) — threshold at the midpoint, not >0.
        mask = (mask_raw > 127).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx: int) -> KvasirSample:
        image_id = self.image_ids[idx]
        image, mask = self._load_pair(image_id)

        if self.resize is not None:
            h, w = self.resize
            image = np.array(Image.fromarray(image).resize((w, h), Image.BILINEAR))
            mask = np.array(
                Image.fromarray(mask * 255).resize((w, h), Image.NEAREST)
            )
            mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            return KvasirSample(image_id=image_id, image=out["image"], mask=out["mask"])

        # No transform: hand back full-resolution PIL image + numpy mask, exactly
        # what run_sam3.py / prompts.py expect.
        return KvasirSample(image_id=image_id, image=Image.fromarray(image), mask=mask)
