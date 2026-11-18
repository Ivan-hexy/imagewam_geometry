#!/usr/bin/env bash
set -euo pipefail
export IMAGEWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$IMAGEWAM_ROOT"
if [[ -f .env.local ]]; then source .env.local; fi
export SIMPLE_RGBD_ROOT="${SIMPLE_RGBD_ROOT:-$IMAGEWAM_ROOT/data/simple_rgbd_1aa44615/SIMPLE-RGBD-Teleop}"
export SIMPLE_RGBD_CACHE="${SIMPLE_RGBD_CACHE:-$IMAGEWAM_ROOT/data/simple_rgbd_1aa44615/SIMPLE-RGBD-Teleop-cache}"
export FLUX2_SRC="${FLUX2_SRC:-$IMAGEWAM_ROOT/third_party/flux2}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-$IMAGEWAM_ROOT/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-$IMAGEWAM_ROOT/checkpoints/flux2/FLUX.2-dev/ae.safetensors}"
export SIMPLE_ACTION_INIT="${SIMPLE_ACTION_INIT:-$IMAGEWAM_ROOT/checkpoints/action_dit_flux2_4b_simple9_36d_init.pt}"
export DEPTH_PRO_SRC="${DEPTH_PRO_SRC:-$IMAGEWAM_ROOT/third_party/ml-depth-pro}"
export DEPTH_PRO_CHECKPOINT="${DEPTH_PRO_CHECKPOINT:-$IMAGEWAM_ROOT/checkpoints/depth_pro/depth_pro.pt}"
export PYTHONPATH="$IMAGEWAM_ROOT/src:$FLUX2_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
