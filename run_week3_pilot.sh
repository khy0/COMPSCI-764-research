#!/usr/bin/env bash
# Pilot for the fixed-step / moving-average / cosine+warmup / seeded scheme
# (decisions.md, 2026-10-13 and 2026-10-14 entries) — the 3 configs Sebastian
# proposed before committing to the full 12-run grid.
#
#   chmod +x run_week3_pilot.sh
#   nohup caffeinate -i ./run_week3_pilot.sh > logs/pilot_overall.log 2>&1 &
#   disown
#
# RUN_TAG keeps these runs' artifacts from colliding with (or being silently
# skipped because of) the original grid's files already on disk.

set -uo pipefail
mkdir -p logs

RUN_TAG="v2_"
CONFIGS=("100pct 0" "25pct 1" "50pct 2")

TOTAL=${#CONFIGS[@]}
I=0
FAILED=()

for cfg in "${CONFIGS[@]}"; do
  read -r bracket seed <<< "$cfg"
  I=$((I + 1))
  run_name="${RUN_TAG}${bracket}_seed${seed}"
  ckpt="checkpoints/unet_${run_name}_best.pt"
  profile_csv="results/compute_profiles/unet_compute_profile_${run_name}.csv"
  log_file="logs/train_${run_name}.log"

  if [[ -f "$ckpt" && -f "$profile_csv" ]]; then
    echo "[$I/$TOTAL] $run_name — already complete, skipping."
    continue
  fi

  echo "[$I/$TOTAL] $run_name — starting $(date)"
  python -u src/train_unet.py --bracket "$bracket" --seed "$seed" --run-tag "$RUN_TAG" 2>&1 | tee "$log_file"
  
  status=${PIPESTATUS[0]}

  if [[ $status -ne 0 ]]; then
    echo "[$I/$TOTAL] $run_name — FAILED (exit $status), see $log_file."
    FAILED+=("$run_name")
  else
    echo "[$I/$TOTAL] $run_name — done $(date)"
  fi
done

echo ""
echo "Pilot run finished. $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED runs (${#FAILED[@]}): ${FAILED[*]}"
else
  echo "All $TOTAL pilot runs completed successfully."
fi