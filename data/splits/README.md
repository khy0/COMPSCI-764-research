# Split index files (committed, frozen)

This directory holds the **frozen** train/val/test split and the four nested
training subsets, as plain-text files — one image ID per line.

Generated once from `configs/data.yaml`, seed 42, then committed here. **Never
regenerate these in code** — a library version bump can silently change a shuffle,
which would invalidate every result (planning doc, P1).

Expected files, once generated in Week 1:

| File | Contents |
|---|---|
| `test.txt` | 200 images — held out, never trained on, identical for both arms |
| `val.txt` | 100 images — U-Net model selection only |
| `train_pool.txt` | 700 images — source pool for the nested subsets below |
| `train_10.txt` | 70 images (10% of train pool) |
| `train_25.txt` | 175 images (25%) — superset of `train_10.txt` |
| `train_50.txt` | 350 images (50%) — superset of `train_25.txt` |
| `train_100.txt` | 700 images (100%) — superset of `train_50.txt` |
