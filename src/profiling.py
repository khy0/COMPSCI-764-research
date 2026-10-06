"""Compute-profiling helpers — latency / memory / param-count instrumentation.

Per configs/eval.yaml `compute_profiling` (track_latency, track_memory,
track_params, same_hardware_required). This is NOT (yet) one of the
"non-negotiable" shared files listed in src/README.md — it was written for the
SAM 3 arm (Week 2) but deliberately arm-agnostic, so the U-Net arm can reuse it
for symmetric profiling later without duplicating logic. Flagged in decisions.md
for Balamurali's awareness; treat it as a shared proposal, not a shared fact,
until he's used it too.

IMPORTANT — same_hardware_required (configs/eval.yaml): these numbers are only
comparable between the two arms if both were profiled on the *same machine* in
the *same run session* (no other heavy process competing for CPU/RAM/GPU at the
time). Latency/memory are NOT portable across machines the way segmentation
metrics are — don't let SAM3 numbers collected here get compared against U-Net
numbers collected on a different machine.

Memory-measurement caveat (read before trusting a number): CPU and CUDA are
measured differently, and are NOT directly comparable to each other:
  - CUDA: `torch.cuda.max_memory_allocated()` after
    `torch.cuda.reset_peak_memory_stats()` — a true, call-scoped peak. Reliable.
  - CPU: `resource.getrusage().ru_maxrss` sampled right after the call. This is
    the process's peak RSS *since process start*, monotonically non-decreasing —
    NOT resettable per-call. Reported values are only useful as a coarse
    "footprint so far" signal (e.g. to catch obvious leaks/regressions across a
    run), not as a precise per-image peak. If per-image CPU peak isolation
    becomes something the report needs, that requires `psutil` sampling in a
    background thread or per-image subprocess isolation — neither is done here.
    `ru_maxrss`'s *unit* also differs by OS: KB on Linux, bytes on macOS —
    see `_ru_maxrss_mb()` below. This also covers `--device mps`, which has no
    CUDA-style reset/peak API and falls back to this same CPU measurement.
"""

from __future__ import annotations

import csv
import os
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


def _ru_maxrss_mb() -> float:
    """`ru_maxrss` is in KB on Linux but bytes on macOS (Darwin) — same struct
    field, different unit, and getrusage(2) itself doesn't say which. Get this
    wrong and every CPU/MPS memory number here is off by ~1024x."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024 ** 2) if sys.platform == "darwin" else 1024
    return raw / divisor


def count_params(model) -> dict:
    """Total and trainable parameter counts for a torch.nn.Module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_params_total": total, "n_params_trainable": trainable}


@dataclass
class ProfileRecord:
    arm: str
    run_id: str
    image_id: str
    device: str
    latency_ms: float
    peak_mem_mb: float
    n_params_total: int
    n_params_trainable: int


class ComputeProfiler:
    """Accumulates one ProfileRecord per (image, strategy) call via `measure()`,
    then writes them all out with `write_csv()`.

    Usage:
        profiler = ComputeProfiler(arm="sam3", device="cuda", param_counts=count_params(model))
        with profiler.measure(run_id="box-gt", image_id="cju0...") :
            pred = predict_one(model, processor, image, prompt)
        ...
        profiler.write_csv("results/sam3_compute_profile.csv")
    """

    def __init__(self, arm: str, device: str, param_counts: dict):
        self.arm = arm
        self.device = device
        self.param_counts = param_counts
        self.records: list[ProfileRecord] = []
        self._is_cuda = device.startswith("cuda")

    @contextmanager
    def measure(self, run_id: str, image_id: str):
        if self._is_cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._is_cuda:
                import torch

                torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                # See module docstring: this is process-lifetime peak RSS, not a
                # true call-scoped peak, and the raw unit is OS-dependent.
                peak_mem_mb = _ru_maxrss_mb()
            latency_ms = (time.perf_counter() - t0) * 1000

            self.records.append(
                ProfileRecord(
                    arm=self.arm,
                    run_id=run_id,
                    image_id=image_id,
                    device=self.device,
                    latency_ms=round(latency_ms, 3),
                    peak_mem_mb=round(peak_mem_mb, 3),
                    n_params_total=self.param_counts["n_params_total"],
                    n_params_trainable=self.param_counts["n_params_trainable"],
                )
            )

    def write_csv(self, out_path: str) -> None:
        if not self.records:
            return
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fieldnames = list(self.records[0].__dataclass_fields__.keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.__dict__)
        print(f"Wrote {len(self.records)} profiling rows to {out_path}.")

    def summary(self) -> dict:
        """Quick aggregate for console/log output — not a substitute for the
        per-image CSV, which is what any reported statistic should derive from."""
        if not self.records:
            return {}
        latencies = [r.latency_ms for r in self.records]
        mems = [r.peak_mem_mb for r in self.records]
        return {
            "n_calls": len(self.records),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "mean_peak_mem_mb": round(sum(mems) / len(mems), 2),
            "max_peak_mem_mb": round(max(mems), 2),
            "n_params_total": self.records[0].n_params_total,
        }
