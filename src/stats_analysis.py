"""Week 4 statistics: bootstrap CIs, paired Wilcoxon tests, crossover interval.

Reads the single per-image CSV written by evaluate.py (both arms) and writes tidy
CSVs under results/analysis/. Nothing here recomputes metrics.

Usage
-----
    python src/stats_analysis.py \
        --csv results/per_image_results.csv \
        --config configs/analysis.yaml --eval-config configs/eval.yaml \
        --out-dir results/analysis

Outputs
-------
    summary_ci.csv        per condition x metric: estimate, image-only CI, two-stage CI,
                          per-seed mean/sd/min/max (U-Net)
    wilcoxon.csv          U-Net bracket vs each SAM3 strategy, per metric: paired Wilcoxon,
                          rank-biserial r, median paired difference, raw/Holm/Bonferroni p
    crossover.csv         per metric x SAM3 strategy: point crossover n*, bootstrap interval,
                          censoring fractions
    crossover_boot.npz    raw bootstrap draws (curves + n*) for the figure script
    data_report.txt       n images, n inf, n dropped, policy used

Conventions
-----------
* Unit of analysis is the test image. The U-Net score of an image in a bracket is the
  mean over seeds (paired against the single SAM3 score for the same image).
* Two-stage bootstrap: each replicate resamples test images (shared across all
  conditions so comparisons stay paired) AND, per U-Net bracket, resamples seeds with
  replacement. With only 3 seeds the seed stage is coarse; the image-only CI is always
  reported next to it so the reader can see how much the seed stage widens it.
* Summary statistic per metric comes from configs/analysis.yaml (mean; median for HD95).
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy import stats

_RUN_RE = re.compile(r"^(?P<bracket>.+?)[_-]seed(?P<seed>\d+)$")


# --------------------------------------------------------------------------- loading
def load_results(csv_path: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"arm", "run_id", "image_id"} | set(cfg["metrics"])
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df["arm"] = df["arm"].astype(str)
    df["run_id"] = df["run_id"].astype(str)
    df["bracket"] = None
    df["seed"] = np.nan
    is_unet = df["arm"] == "unet"
    parsed = df.loc[is_unet, "run_id"].str.extract(_RUN_RE)
    if parsed["bracket"].isna().any():
        bad = df.loc[is_unet, "run_id"][parsed["bracket"].isna()].unique()[:5]
        raise ValueError(f"U-Net run_id not like '<bracket>_seed<k>': {list(bad)}")
    df.loc[is_unet, "bracket"] = parsed["bracket"].values
    df.loc[is_unet, "seed"] = parsed["seed"].astype(int).values
    dup = df.duplicated(["arm", "run_id", "image_id"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (arm, run_id, image_id) rows - the CSV was "
                         "appended twice? Re-run evaluate.py with --no-append.")
    return df


def apply_hd95_policy(df: pd.DataFrame, cfg: dict):
    """Returns (df, report dict). `inf` HD95 = empty prediction."""
    pol = cfg["hd95_inf_policy"]
    df = df.copy()
    inf_mask = np.isinf(df["hd95"].to_numpy(float))
    report = {"mode": pol["mode"], "n_inf_rows": int(inf_mask.sum()),
              "n_inf_by_condition": df.loc[inf_mask].groupby(["arm", "run_id"]).size().to_dict()}
    if pol["mode"] == "cap":
        fin = df.loc[~inf_mask, "hd95"]
        cap = pol.get("cap_value") or (float(fin.max()) if len(fin) else 0.0)
        df.loc[inf_mask, "hd95"] = cap
        report["cap_value"] = cap
    elif pol["mode"] == "exclude":
        bad_imgs = set(df.loc[inf_mask, "image_id"])
        df = df[~df["image_id"].isin(bad_imgs)].copy()
        report["n_images_dropped"] = len(bad_imgs)
    else:
        raise ValueError("hd95_inf_policy.mode must be 'cap' or 'exclude'")
    return df, report


def build_conditions(df: pd.DataFrame, cfg: dict, metric: str):
    """-> (image_ids, {cond_name: array[n_img, n_seeds]}, meta).
    Only images present in EVERY condition are kept (paired design)."""
    conds = {}
    for s in cfg["sam3"]["strategies"]:
        sub = df[(df.arm == "sam3") & (df.run_id == s)]
        if len(sub):
            conds[f"sam3:{s}"] = sub.pivot(index="image_id", columns="run_id", values=metric)
    for b in cfg["unet"]["brackets"]:
        sub = df[(df.arm == "unet") & (df.bracket == b)]
        if len(sub):
            conds[f"unet:{b}"] = sub.pivot(index="image_id", columns="seed", values=metric)
    if not conds:
        raise ValueError("No conditions found in CSV")
    ids = None
    for p in conds.values():
        ids = p.dropna().index if ids is None else ids.intersection(p.dropna().index)
    ids = sorted(ids)
    out = {k: p.loc[ids].to_numpy(float) for k, p in conds.items()}
    return ids, out


# ---------------------------------------------------------------------- bootstrap
def _summ(x: np.ndarray, how: str, axis: int):
    return np.median(x, axis=axis) if how == "median" else np.mean(x, axis=axis)


def bootstrap_conditions(V: dict, how: str, n_boot: int, rng, two_stage: bool, chunk=1000):
    """Paired bootstrap. Returns {cond: array[n_boot]} of the summary statistic.
    Image draws are SHARED across conditions (keeps paired comparisons paired)."""
    n_img = next(iter(V.values())).shape[0]
    out = {k: np.empty(n_boot) for k in V}
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        I = rng.integers(0, n_img, size=(b, n_img))
        for k, a in V.items():
            ns = a.shape[1]
            if two_stage and ns > 1:
                S = rng.integers(0, ns, size=(b, ns))
                M = a[:, S].mean(axis=2).T            # (b, n_img): seed-resampled image means
            else:
                M = np.tile(a.mean(axis=1), (b, 1))   # seed-averaged image scores
            sel = np.take_along_axis(M, I, axis=1)
            out[k][lo:lo + b] = _summ(sel, how, axis=1)
    return out


def ci(draws: np.ndarray, level: float):
    a = (1 - level) / 2
    return np.quantile(draws, [a, 1 - a])


# ------------------------------------------------------------------------ crossover
def crossover_n(curve: np.ndarray, ns: np.ndarray, ref: float, direction: int):
    """curve: values per bracket (ascending n). Returns (n_star, status).
    status in {'interp','below_min','never'}; below_min -> n* < ns[0] (already matched)."""
    d = direction * (curve - ref)       # >= 0 means U-Net at least as good
    if d[0] >= 0:
        return ns[0], "below_min"
    for i in range(1, len(ns)):
        if d[i] >= 0:
            frac = (0 - d[i - 1]) / (d[i] - d[i - 1])
            return float(np.exp(np.log(ns[i - 1]) + frac * (np.log(ns[i]) - np.log(ns[i - 1])))), "interp"
    return np.nan, "never"


def crossover_vec(curves: np.ndarray, ns: np.ndarray, ref: np.ndarray, direction: int):
    """curves: (B, n_brackets); ref: (B,). -> n_star (B,), status array."""
    n_star = np.empty(len(ref))
    status = np.empty(len(ref), dtype=object)
    for b in range(len(ref)):
        n_star[b], status[b] = crossover_n(curves[b], ns, ref[b], direction)
    return n_star, status


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------- driver
def run(csv, cfg, eval_cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(csv, cfg)
    df, hd_report = apply_hd95_policy(df, cfg)
    n_boot = int(eval_cfg["statistics"]["bootstrap_resamples"])
    level = float(eval_cfg["statistics"]["ci"])
    rng_seed = cfg["bootstrap"]["seed"]
    br = cfg["unet"]["brackets"]
    ns_all = np.array(list(br.values()), float)
    report = [f"csv: {csv}", f"hd95 policy: {hd_report}"]

    summary_rows, wil_rows, cross_rows = [], [], []
    boot_store = {}

    for metric, mc in cfg["metrics"].items():
        how, direction = mc["summary"], mc["direction"]
        ids, V = build_conditions(df, cfg, metric)
        report.append(f"[{metric}] paired images: {len(ids)}; conditions: {list(V)}")

        # --- CIs: image-only and two-stage (independent streams, reproducible)
        d_img = bootstrap_conditions(V, how, n_boot, np.random.default_rng(rng_seed), False)
        d_two = bootstrap_conditions(V, how, n_boot, np.random.default_rng(rng_seed + 1), True)
        for k, a in V.items():
            arm, name = k.split(":")
            est = float(_summ(a.mean(axis=1), how, axis=0))
            lo_i, hi_i = ci(d_img[k], level)
            lo_t, hi_t = ci(d_two[k], level)
            row = dict(metric=metric, arm=arm, condition=name,
                       n_train=br.get(name) if arm == "unet" else np.nan,
                       n_images=len(ids), n_seeds=a.shape[1], summary=how, estimate=est,
                       ci_image_lo=lo_i, ci_image_hi=hi_i, ci_two_stage_lo=lo_t, ci_two_stage_hi=hi_t)
            if arm == "unet":
                per_seed = np.array([_summ(a[:, j], how, axis=0) for j in range(a.shape[1])])
                row.update(seed_mean=per_seed.mean(), seed_sd=per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
                           seed_min=per_seed.min(), seed_max=per_seed.max())
            summary_rows.append(row)

        # --- paired Wilcoxon: every U-Net bracket vs every SAM3 strategy
        fam = []
        for ub in [b for b in br if f"unet:{b}" in V]:
            u = V[f"unet:{ub}"].mean(axis=1)
            for s in [s for s in cfg["sam3"]["strategies"] if f"sam3:{s}" in V]:
                z = V[f"sam3:{s}"][:, 0]
                diff = direction * (u - z)                  # >0 means U-Net better
                nz = diff != 0
                if nz.sum() < 2:
                    stat_, p, r = np.nan, 1.0, 0.0
                else:
                    res = stats.wilcoxon(u, z, alternative=cfg["wilcoxon"]["alternative"],
                                         zero_method=cfg["wilcoxon"]["zero_method"])
                    stat_, p = float(res.statistic), float(res.pvalue)
                    # rank-biserial from signed ranks of the direction-adjusted differences
                    ranks = stats.rankdata(np.abs(diff[nz]))
                    pos, neg = ranks[diff[nz] > 0].sum(), ranks[diff[nz] < 0].sum()
                    r = float((pos - neg) / (pos + neg))
                fam.append(dict(metric=metric, unet_bracket=ub, n_train=br[ub], sam3_strategy=s,
                                n_pairs=len(ids), n_ties=int((~nz).sum()), W=stat_,
                                p_raw=p, rank_biserial=r,                 # + favours U-Net
                                median_diff=float(np.median(u - z)),      # raw units, unet - sam3
                                frac_images_unet_better=float(np.mean(diff > 0))))
        if fam:
            pr = np.array([f["p_raw"] for f in fam])
            for f, h, bf in zip(fam, holm(pr), np.minimum(1.0, pr * len(pr))):
                f.update(p_holm=h, p_bonferroni=bf, family_size=len(pr))
            wil_rows += fam

        # --- crossover: brackets present, ascending n
        ubs = [b for b in br if f"unet:{b}" in V]
        if len(ubs) >= 2:
            ns = np.array([br[b] for b in ubs], float)
            curves = np.stack([d_two[f"unet:{b}"] for b in ubs], axis=1)
            curves_img = np.stack([d_img[f"unet:{b}"] for b in ubs], axis=1)
            pt_curve = np.array([_summ(V[f"unet:{b}"].mean(axis=1), how, 0) for b in ubs])
            boot_store[f"{metric}__curves"] = curves
            boot_store[f"{metric}__ns"] = ns
            for s in [s for s in cfg["sam3"]["strategies"] if f"sam3:{s}" in V]:
                ref_pt = float(_summ(V[f"sam3:{s}"][:, 0], how, 0))
                n_pt, st_pt = crossover_n(pt_curve, ns, ref_pt, direction)
                ref_b = d_two[f"sam3:{s}"]
                n_b, st_b = crossover_vec(curves, ns, ref_b, direction)
                n_bi, st_bi = crossover_vec(curves_img, ns, d_img[f"sam3:{s}"], direction)
                boot_store[f"{metric}__{s}__nstar"] = n_b
                boot_store[f"{metric}__{s}__ref"] = ref_b

                def pack(nb, sb):
                    # Censored replicates stay in the distribution: "never crossed" = +inf,
                    # so an upper bound that reaches the censored mass is reported as ">max n"
                    # instead of being silently pulled down by dropping those replicates.
                    below = float(np.mean(sb == "below_min")); never = float(np.mean(sb == "never"))
                    full = np.where(np.isnan(nb), np.inf, nb)
                    a_ = (1 - level) / 2
                    lo, hi = np.quantile(full, [a_, 1 - a_], method="nearest")
                    return float(lo), float(hi), below, never

                lo, hi, below, never = pack(n_b, st_b)
                lo_i2, hi_i2, below_i, never_i = pack(n_bi, st_bi)
                label = ("<%d" % ns[0]) if st_pt == "below_min" else (">%d" % ns[-1]) if st_pt == "never" else f"{n_pt:.0f}"
                cross_rows.append(dict(
                    metric=metric, sam3_strategy=s, sam3_ref=ref_pt, n_star_point=n_pt,
                    n_star_label=label, point_status=st_pt,
                    n_star_lo=lo, n_star_hi=hi, frac_never=never, frac_already_at_min=below,
                    n_star_image_only_lo=lo_i2, n_star_image_only_hi=hi_i2,
                    frac_never_image_only=never_i, frac_already_at_min_image_only=below_i,
                    interval_note="percentile interval over ALL replicates; inf = never crossed within the "
                                  "largest bracket; value at the smallest n = already matched at the smallest bracket"))

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary_ci.csv"), index=False)
    pd.DataFrame(wil_rows).to_csv(os.path.join(out_dir, "wilcoxon.csv"), index=False)
    pd.DataFrame(cross_rows).to_csv(os.path.join(out_dir, "crossover.csv"), index=False)
    if boot_store:
        np.savez_compressed(os.path.join(out_dir, "crossover_boot.npz"), **boot_store)
    with open(os.path.join(out_dir, "data_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"Wrote analysis tables to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--out-dir", default="results/analysis")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ev = yaml.safe_load(open(a.eval_config))
    run(a.csv or cfg["input"]["per_image_csv"], cfg, ev, a.out_dir)


if __name__ == "__main__":
    main()
