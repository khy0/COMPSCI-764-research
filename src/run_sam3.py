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


def predict_one(model, processor, image: Image.Image, prompt) -> np.ndarray:
    """Run predict_inst for a single Prompt (from prompts.py) and return the
    single best binary mask (H, W) uint8 {0,1}, at native image resolution."""
    state = processor.set_image(image)

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

    for strategy in strategies:
        os.makedirs(os.path.join(out_dir, strategy["id"]), exist_ok=True)

    for i, sample in enumerate(dataset):
        prompts = generate_all_prompts(strategies, sample.mask, sample.image_id, base_seed)
        for strategy_id, prompt in prompts.items():
            pred = predict_one(model, processor, sample.image, prompt)
            out_path = os.path.join(out_dir, strategy_id, sample.image_id + ".png")
            Image.fromarray(pred * 255).save(out_path)
        print(f"[{i + 1}/{len(dataset)}] {sample.image_id}: "
              f"{len(prompts)} strategies done")

    print(f"Done. Predictions written under {out_dir}/<strategy_id>/<image_id>.png")
    print("Next: src/evaluate.py --pred-dir "
          f"{out_dir}/<strategy_id> --arm sam3 --run-id <strategy_id>")


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
                              "— use --limit 1 for the Week-1 gate 'one image' check).")
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
    )


if __name__ == "__main__":
    main()
