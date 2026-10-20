"""Render the qualitative panel for the cases chosen by select_qualitative_cases.py.

One row per case; columns: image | ground truth | SAM 3 | U-Net (small bracket) | U-Net (large bracket).
Prediction = translucent fill in the arm colour; ground-truth outline drawn on every cell.
Needs the images, GT masks and saved prediction PNGs (run LOCALLY / on the cluster):

    python src/render_qualitative_panel.py \
        --cases results/analysis/qualitative_cases.csv \
        --data-root data/kvasir-seg \
        --sam3-pred-dir results/sam3_preds/box-j10 \
        --unet-pred-root results/unet_preds \
        --out results/figures/qualitative

--unet-pred-root holds <bracket>_seed<k>/ folders and auto-picks the middle seed per bracket.
Or name the two folders yourself:  --unet-pred-dirs 10pct=<dir> 100pct=<dir>

U-Net prediction PNGs are the same {0,255} files evaluate_unet_grid.py scores. One seed per bracket is shown:
with --unet-pred-root the seed whose mean Dice is closest to the bracket mean is chosen automatically.
Use --draft to watermark test renders.
"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from skimage import measure

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
C_SAM, C_UNET_S, C_UNET_L = "#eb6834", "#2a78d6", "#2a78d6"   # arm colour; U-Net brackets share the arm hue
REASONS = {"sam3_wins": "SAM 3 ahead", "unet_wins": "U-Net ahead", "both_fail": "Both struggle",
           "typical": "Typical", "small_polyp": "Small polyp", "boundary_gap": "Similar Dice, boundary differs"}


def rd(path, thr=127):
    return np.array(Image.open(path).convert("L")) > thr


def dice(p, g):
    return 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)


def overlay(ax, img, gt, pred, color, text):
    ax.imshow(img)
    if pred is not None:
        rgba = np.zeros(pred.shape + (4,)); rgba[pred] = matplotlib.colors.to_rgba(color, 0.45)
        ax.imshow(rgba)
    for c in measure.find_contours(gt.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color="white", lw=2.6); ax.plot(c[:, 1], c[:, 0], color=INK, lw=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if text:
        ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=8.5, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none", alpha=0.75))


def pick_seed_dir(root, bracket, csv_path):
    """Representative seed = the one whose test-set mean Dice is closest to the bracket's
    mean over seeds (the middle seed for 3 seeds). Avoids showing a best/worst seed."""
    d = pd.read_csv(csv_path)
    d = d[(d.arm == "unet") & d.run_id.str.match(rf"^{bracket}[_-]seed\d+$")]
    per = d.groupby("run_id").dice.mean()
    run = (per - per.mean()).abs().idxmin()
    print(f"[{bracket}] seed dice means {per.round(4).to_dict()} -> showing {run}")
    path = os.path.join(root, run)
    if not os.path.isdir(path):
        alt = os.path.join(root, run.replace("-seed", "_seed"))
        path = alt if os.path.isdir(alt) else path
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="results/analysis/qualitative_cases.csv")
    ap.add_argument("--data-root", default="data/kvasir-seg")
    ap.add_argument("--sam3-pred-dir", required=True)
    ap.add_argument("--unet-pred-dirs", nargs=2, metavar="BRACKET=DIR",
                    help="explicit folders, one per column (use this OR --unet-pred-root)")
    ap.add_argument("--unet-pred-root", help="folder holding <bracket>_seed<k>/ subfolders (e.g. unet_preds/)")
    ap.add_argument("--brackets", nargs=2, default=["10pct", "100pct"])
    ap.add_argument("--csv", default="results/per_image_results.csv",
                    help="per-image CSV, used only to pick the representative seed")
    ap.add_argument("--image-ext", default=".jpg"); ap.add_argument("--mask-ext", default=".jpg")
    ap.add_argument("--out", default="results/figures/qualitative")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    cases = pd.read_csv(a.cases)
    if a.unet_pred_dirs:
        unets = [x.split("=", 1) for x in a.unet_pred_dirs]
    elif a.unet_pred_root:
        unets = [(b, pick_seed_dir(a.unet_pred_root, b, a.csv)) for b in a.brackets]
    else:
        ap.error("give --unet-pred-dirs or --unet-pred-root")
    cols = [("Image", None, None)] + [("Ground truth", None, None),
            (f"SAM 3 ({cases.sam3_strategy[0]})", a.sam3_pred_dir, C_SAM)] + \
           [(f"U-Net, {b} data", d, C_UNET_S) for b, d in unets]
    n = len(cases)
    fig, axs = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.1 * n + 0.9), facecolor=SURFACE, squeeze=False)
    for i, r in cases.iterrows():
        img = np.array(Image.open(os.path.join(a.data_root, "images", r.image_id + a.image_ext)).convert("RGB"))
        gt = rd(os.path.join(a.data_root, "masks", r.image_id + a.mask_ext))
        for j, (title, d, col) in enumerate(cols):
            ax = axs[i, j]
            if i == 0: ax.set_title(title, fontsize=10.5, color=INK, fontweight="semibold", loc="left")
            if j == 0:
                overlay(ax, img, gt, None, None, None)
                ax.set_ylabel(f"{REASONS.get(r.reason, r.reason)}\n{r.gt_area_pct:.1f}% area", rotation=0, ha="right",
                              va="center", fontsize=9, color=INK2, labelpad=8)
                for c in ax.lines: c.set_visible(False)       # raw image column: no outline
            elif j == 1:
                overlay(ax, img, gt, gt, "#898781", None)
            else:
                p = rd(os.path.join(d, r.image_id + ".png"))
                if p.shape != gt.shape:
                    p = np.array(Image.fromarray((p * 255).astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 127
                overlay(ax, img, gt, p, col, f"Dice {dice(p, gt):.2f}")
    fig.text(0.01, 0.995, "Qualitative cases (white/black outline = ground truth; fill = prediction)", va="top",
             fontsize=11.5, fontweight="semibold", color=INK)
    if a.draft:
        fig.text(0.5, 0.5, "PROVISIONAL / SYNTHETIC", ha="center", va="center", fontsize=44, color=INK, alpha=0.08, rotation=20)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=170, facecolor=SURFACE); fig.savefig(a.out + ".pdf", facecolor=SURFACE)
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
