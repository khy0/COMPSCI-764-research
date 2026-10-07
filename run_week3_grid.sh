#!/usr/bin/env bash
# Week 3 Task 1.2 — queue and run all 12 U-Net configs (4 brackets x 3 seeds),
# using configs/unet_base.yaml's own seeds: [0, 1, 2].
#
# Run this from the repo root:
#   chmod +x run_week3_grid.sh
#   nohup caffeinate -i ./run_week3_grid.sh > logs/grid_overall.log 2>&1 &
#   disown
#
# - `caffeinate -i` stops macOS from sleeping mid-run (a full grid is several
#   hours — see the time estimate in chat).
# - `nohup ... &` + `disown` keeps it running even if you close the terminal
#   window; check progress any time with:
#     tail -f logs/grid_overall.log
#   or see which per-run logs exist yet: ls logs/
#
# Resumable: if interrupted (crash, sleep, you killed it), just re-run the same
# command — any (bracket, seed) that already has BOTH a best checkpoint AND a
# profiling CSV is skipped, everything else (re)runs from scratch. A checkpoint
# with no matching profiling CSV (e.g. 100pct/seed1 from the earlier Task 2 run,
# trained before profiling hooks existed) is NOT considered complete and will be
# retrained — this keeps every run's profiling data captured the same way.

set -uo pipefail

mkdir -p logs

BRACKETS=(10pct 25pct 50pct 100pct)
SEEDS=(0 1 2)

TOTAL=$(( ${#BRACKETS[@]} * ${#SEEDS[@]} ))
I=0
FAILED=()

for bracket in "${BRACKETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    I=$((I + 1))
    run_name="${bracket}_seed${seed}"
    ckpt="checkpoints/unet_${run_name}_best.pt"
    profile_csv="results/unet_compute_profile_${run_name}.csv"
    log_file="logs/train_${run_name}.log"

    if [[ -f "$ckpt" && -f "$profile_csv" ]]; then
      echo "[$I/$TOTAL] $run_name — already complete (checkpoint + profiling CSV present), skipping."
      continue
    fi

    echo "[$I/$TOTAL] $run_name — starting $(date)"
    python src/train_unet.py --bracket "$bracket" --seed "$seed" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    if [[ $status -ne 0 ]]; then
      echo "[$I/$TOTAL] $run_name — FAILED (exit $status), see $log_file. Continuing to next run."
      FAILED+=("$run_name")
    else
      echo "[$I/$TOTAL] $run_name — done $(date)"
    fi
  done
done

echo ""
echo "Grid run finished. $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED runs (${#FAILED[@]}): ${FAILED[*]} — check their logs in logs/, then re-run this script to retry just those."
else
  echo "All 12 runs completed successfully."
fi
