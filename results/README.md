# Results (committed)

Per-image result CSVs land here, produced by `src/evaluate.py` once it exists.
Committed to git — these are small (one row per image per condition) and are the
single source of truth every statistic in the report derives from.

Expected: `per_image_results.csv`, one row per (image, arm, bracket/prompt) combination,
columns for each metric in `configs/eval.yaml`.
