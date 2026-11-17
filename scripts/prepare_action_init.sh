#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if [[ -f "$SIMPLE_ACTION_INIT" ]]; then echo "Already exists: $SIMPLE_ACTION_INIT"; exit 0; fi
python scripts/flux2/preprocess_action_dit_flux2.py   --model-config configs/model/imagewam_flux2_klein_4b_base.yaml   --flux2-src-path "$FLUX2_SRC" --flux2-model-path "$FLUX2_MODEL_PATH"   --variant klein-base-4b --action-dim 36 --use-gradient-checkpointing   --output "$SIMPLE_ACTION_INIT" --apply-alpha-scaling true
