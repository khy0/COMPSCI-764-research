# Kvasir-SEG dataset (not committed)

This directory holds the raw Kvasir-SEG dataset (Jha et al., 2020) and is
gitignored — do not commit images.

**Download:** https://datasets.simula.no/kvasir-seg/

Expected layout after extraction:

```
data/kvasir-seg/
├── images/     # 1000 colonoscopy images (.jpg)
└── masks/      # 1000 corresponding binary polyp masks (.jpg)
```

Run the sanity check (Week 1 task) before anything else: confirm 1000 image/mask
pairs, masks are binary, and every image has a matching mask filename.
