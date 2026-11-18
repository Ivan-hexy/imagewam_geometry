#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mode="${1:-}"; [[ $# -gt 0 ]] && shift
case "$mode" in baseline|only_pose|only_depth|both|geometry) ;; *) echo 'Usage: bash scripts/train.sh baseline|only_pose|only_depth|both [Hydra overrides]' >&2; exit 2;; esac
gpus="${GPU_PER_NODE:-4}"
if [[ "${DRY_RUN:-false}" != true ]]; then
  python scripts/check_setup.py --mode "$mode" --require-weights -- "$@"
fi
cmd=(python -m accelerate.commands.launch
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --num_processes "$gpus" --num_machines 1 --machine_rank 0
  --main_process_port "${MASTER_PORT:-29500}"
  scripts/train.py "task=imagewam_$mode"
  "output_dir=./runs/imagewam_${mode}_$(date +%Y%m%d_%H%M%S)" "$@")
if [[ "${DRY_RUN:-false}" == true ]]; then printf '%q ' "${cmd[@]}"; printf '
'; else exec "${cmd[@]}"; fi
