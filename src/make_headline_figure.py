"""Headline multi-panel figure (planning doc §3.4).

Top row: one panel per primary metric (Dice, Boundary IoU, HD95) - U-Net learning curve
(seed-mean line, two-stage bootstrap 95% band, individual seeds as dots) against SAM3
zero-shot reference lines (each with its own 95% image-bootstrap band).
Bottom panel: crossover interval n* per SAM3 strategy and metric (point + 95% interval;
an arrow means the interval is censored at the largest bracket).

Reads the tables written by stats_analysis.py and the per-image CSV (for the seed dots).

    python src/make_headline_figure.py --analysis-dir results/analysis \
        --csv results/per_image_results.csv --out results/figures/headline
    # add --draft to stamp a "PROVISIONAL / SYNTHETIC" watermark (used for test renders)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# palette.md light mode, fixed slot order (validated: node scripts/validate_palette.js)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_UNET, C_ORACLE, C_REAL = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3
SAM_LINES = [("box-gt", C_ORACLE, "SAM 3 oracle box"),
             ("box-j10", C_REAL, "SAM 3 box, 10% jitter")]
MARKERS = {"dice": "o", "boundary_iou": "s", "hd95": "D"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1)


def fmt_n(v):
    return f"{int(round(v))}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", default="results/analysis")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--out", default="results/figures/headline")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    csv = a.csv or cfg["input"]["per_image_csv"]
    summ = pd.read_csv(os.path.join(a.analysis_dir, "summary_ci.csv"))
    cross = pd.read_csv(os.path.join(a.analysis_dir, "crossover.csv"))
    prim = cfg["primary_metrics"]
    brs = cfg["unet"]["brackets"]
    nmax = max(brs.values())

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig = plt.figure(figsize=(12.5, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, len(prim), height_ratios=[1.45, 1], hspace=0.42, wspace=0.26,
                          left=0.065, right=0.975, top=0.79, bottom=0.085)

    # ---- top row: learning curves
    df = pd.read_csv(csv)
    for j, m in enumerate(prim):
        ax = fig.add_subplot(gs[0, j]); style_axes(ax)
        mc = cfg["metrics"][m]
        u = summ[(summ.metric == m) & (summ.arm == "unet")].sort_values("n_train")
        x = u.n_train.to_numpy(float)
        # SAM3 reference lines + image-bootstrap bands
        for sid, col, _ in SAM_LINES:
            r = summ[(summ.metric == m) & (summ.arm == "sam3") & (summ.condition == sid)]
            if r.empty: continue
            r = r.iloc[0]
            ax.axhspan(r.ci_image_lo, r.ci_image_hi, color=col, alpha=0.10, linewidth=0)
            ax.axhline(r.estimate, color=col, linewidth=2, solid_capstyle="round")
        # U-Net: band, seeds, line
        ax.fill_between(x, u.ci_two_stage_lo, u.ci_two_stage_hi, color=C_UNET, alpha=0.12, linewidth=0)
        # individual seeds (per-seed summary over images)
        ud = df[df.arm == "unet"].copy()
        if not ud.empty:
            ud[["b", "seed"]] = ud.run_id.str.extract(r"^(.+?)[_-]seed(\d+)$")
            ud = ud[np.isfinite(ud[m])] if mc["summary"] == "mean" else ud
            g = ud.groupby(["b", "seed"])[m].agg("median" if mc["summary"] == "median" else "mean").reset_index()
            g["n"] = g.b.map(brs)
            ax.scatter(g.n, g[m], s=22, color=C_UNET, alpha=0.45, linewidths=0, zorder=3)
        ax.plot(x, u.estimate, color=C_UNET, linewidth=2, solid_capstyle="round", zorder=4)
        ax.scatter(x, u.estimate, s=64, color=C_UNET, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([fmt_n(v) for v in x])
        ax.minorticks_off(); ax.set_xlim(x.min() * 0.8, nmax * 1.5)
        ax.set_xlabel("Training images (log scale)", color=INK2, fontsize=9.5)
        ax.set_ylabel(mc["label"] + ("  - lower is better" if mc["direction"] < 0 else ""), color=INK2, fontsize=9.5)
        ax.set_title(f"{mc['label'].split(' (')[0]}", loc="left", fontsize=11.5, fontweight="semibold", color=INK, pad=8)
        if mc["direction"] > 0: ax.set_ylim(min(0.0, ax.get_ylim()[0]) if False else ax.get_ylim()[0], min(1.0, ax.get_ylim()[1]))
        else: ax.set_ylim(0, None)
        # selective direct labels on the right edge for the SAM3 lines
        for sid, col, lab in SAM_LINES:
            r = summ[(summ.metric == m) & (summ.arm == "sam3") & (summ.condition == sid)]
            if r.empty: continue
            ax.annotate(sid, (x.min() * 0.8, r.iloc[0].estimate), xytext=(4, 4), textcoords="offset points",
                        ha="left", va="bottom", fontsize=8.5, color=INK2)

    # ---- bottom: crossover intervals
    axc = fig.add_subplot(gs[1, :]); style_axes(axc)
    axc.grid(axis="y", visible=False); axc.grid(axis="x", color=GRID, linewidth=1)
    strategies = [s for s in cfg["sam3"]["strategies"] if s in set(cross.sam3_strategy)]
    off = {m: (i - (len(prim) - 1) / 2) * 0.22 for i, m in enumerate(prim)}
    lo_n, hi_n = min(brs.values()), nmax
    axc.set_xscale("log")
    for yi, s in enumerate(strategies):
        for m in prim:
            r = cross[(cross.metric == m) & (cross.sam3_strategy == s)]
            if r.empty: continue
            r = r.iloc[0]; y = yi + off[m]
            lo, hi = r.n_star_lo, r.n_star_hi
            if not np.isfinite(lo):          # every replicate censored: no crossing observed
                axc.scatter([hi_n * 1.12], [y], s=22, marker=MARKERS[m], color=MUTED, linewidths=0)
                axc.text(hi_n * 1.22, y, "no crossing ≤ %d" % hi_n, va="center", ha="left", fontsize=7.5, color=INK2)
                continue
            cens = not np.isfinite(hi)
            hi_draw = hi_n * 1.3 if cens else hi
            axc.plot([lo, hi_draw], [y, y], color=INK2, linewidth=2, solid_capstyle="round", zorder=2)
            if cens:
                axc.annotate("", (hi_n * 1.5, y), (hi_n * 1.25, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
            if r.point_status == "interp":
                axc.scatter([r.n_star_point], [y], s=60, marker=MARKERS[m], color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
    for b, n in brs.items():
        axc.axvline(n, color=GRID, linewidth=1, zorder=0)
    axc.set_yticks(range(len(strategies))); axc.set_yticklabels(strategies, fontsize=9.5, color=INK2)
    axc.invert_yaxis()
    axc.set_xlim(lo_n * 0.8, hi_n * 2.6)
    axc.set_xticks(list(brs.values())); axc.set_xticklabels([fmt_n(v) for v in brs.values()])
    axc.minorticks_off()
    axc.set_xlabel("Training images needed for the U-Net to match SAM 3 (log scale)", color=INK2, fontsize=9.5)
    axc.set_title("Crossover interval n*", loc="left", fontsize=11.5, fontweight="semibold", color=INK, pad=8)
    axc.tick_params(axis="y", length=0)

    # ---- legends (>=2 series -> always present)
    h1 = [Line2D([0], [0], color=C_UNET, lw=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="U-Net (seed mean; band = 95% CI; dots = seeds)")]
    h1 += [Line2D([0], [0], color=c, lw=2, label=l + " (band = 95% CI)") for _, c, l in SAM_LINES]
    fig.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.06, 0.905), ncol=3, columnspacing=2.2, frameon=False,
               fontsize=9.5, labelcolor=INK2, handlelength=2.4)
    h2 = [Line2D([0], [0], color=INK2, lw=0, marker=MARKERS[m], markersize=8, label=cfg["metrics"][m]["label"].split(" (")[0]) for m in prim]
    axc.legend(handles=h2, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    axc.text(0.19, 1.035, "point = n*  ·  line = 95% interval  ·  arrow = open-ended (>%d)" % hi_n if False else f"point = n*  ·  line = 95% interval  ·  arrow = open-ended (>{hi_n})",
             transform=axc.transAxes, fontsize=8.5, color=MUTED, va="bottom")

    fig.text(0.065, 0.975, "How much labelled data does a from-scratch U-Net need to match zero-shot SAM 3?",
             ha="left", va="top", fontsize=13.5, fontweight="semibold", color=INK)
    fig.text(0.065, 0.943, "Kvasir-SEG, 200 held-out test images; 3 seeds per U-Net bracket", ha="left", va="top", fontsize=9.5, color=INK2)
    if a.draft:
        fig.text(0.5, 0.5, "PROVISIONAL / SYNTHETIC U-NET DATA", ha="center", va="center", fontsize=40,
                 color=INK, alpha=0.08, rotation=20, fontweight="bold")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=200, facecolor=SURFACE)
    fig.savefig(a.out + ".pdf", facecolor=SURFACE)
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
