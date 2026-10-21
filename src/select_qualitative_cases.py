"""Deterministic selection of 6-8 qualitative cases from the per-image CSV (no images needed).

Rule (so the selection is defensible, not cherry-picked) - pick, in this order, skipping
images already chosen:
  1. sam3_wins x2      largest  (SAM3 - U-Net) Dice          [SAM3 best case]
  2. unet_wins x2      largest  (U-Net - SAM3) Dice          [U-Net best case]
  3. both_fail x1      lowest   max(SAM3, U-Net) Dice        [shared failure]
  4. typical x1        closest to the median of the mean Dice of the two arms
  5. small_polyp x1    gt_area_pct < small_cut, largest |Dice gap|
  6. boundary_gap x1   |Dice gap| small (<0.05) but largest |Boundary IoU gap|
SAM3 = the strategy in --sam3-strategy (default box-j10, the realistic prompt);
U-Net = seed-mean at --unet-bracket (default 100pct). Reasons are written to the CSV.

    python src/select_qualitative_cases.py --csv results/per_image_results.csv \
        --out results/analysis/qualitative_cases.csv
"""
import argparse, re
import numpy as np, pandas as pd, yaml


def select(df, sam3, bracket, n_cases=8, small_cut=5.0):
    s = df[(df.arm == "sam3") & (df.run_id == sam3)].set_index("image_id")
    u = df[(df.arm == "unet")].copy()
    u["b"] = u.run_id.str.extract(r"^(.+?)[_-]seed\d+$")[0]
    u = u[u.b == bracket].groupby("image_id")[["dice", "boundary_iou", "hd95"]].mean()
    m = s[["gt_area_pct", "dice", "boundary_iou", "hd95"]].join(u, lsuffix="_sam3", rsuffix="_unet", how="inner")
    m["gap"] = m.dice_unet - m.dice_sam3                    # >0 U-Net better
    m["bgap"] = m.boundary_iou_unet - m.boundary_iou_sam3
    picks, used = [], set()

    def take(order, reason, k):
        c = 0
        for iid in order:
            if iid in used: continue
            picks.append((iid, reason)); used.add(iid); c += 1
            if c == k: break

    take(m.gap.sort_values().index, "sam3_wins", 2)                       # most negative gap
    take(m.gap.sort_values(ascending=False).index, "unet_wins", 2)
    take(np.maximum(m.dice_sam3, m.dice_unet).sort_values().index, "both_fail", 1)
    mid = ((m.dice_sam3 + m.dice_unet) / 2); med = mid.median()
    take((mid - med).abs().sort_values().index, "typical", 1)
    sm = m[m.gt_area_pct < small_cut]
    take(sm.gap.abs().sort_values(ascending=False).index, "small_polyp", 1)
    close = m[m.gap.abs() < 0.05]
    take(close.bgap.abs().sort_values(ascending=False).index, "boundary_gap", 1)
    picks = picks[:n_cases]
    out = m.loc[[p[0] for p in picks]].copy()
    out.insert(0, "reason", [p[1] for p in picks])
    out.insert(1, "sam3_strategy", sam3); out.insert(2, "unet_bracket", bracket)
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--sam3-strategy", default="box-j10")
    ap.add_argument("--unet-bracket", default=None)
    ap.add_argument("--out", default="results/analysis/qualitative_cases.csv")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); q = cfg["qualitative"]
    df = pd.read_csv(a.csv or cfg["input"]["per_image_csv"])
    for c in ("dice", "boundary_iou", "hd95"):
        df.loc[~np.isfinite(df[c]), c] = np.nan if c == "hd95" else df.loc[np.isfinite(df[c]), c].min()
    out = select(df, a.sam3_strategy, a.unet_bracket or q["ref_unet_bracket"], q["n_cases"], q["small_cut_pct"])
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))
