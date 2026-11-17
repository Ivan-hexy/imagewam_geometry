#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MS_HUB="${MS_HUB:-.download-venv/bin/ms-hub}"
"$MS_HUB" download ivanhe666/simple_rgbd --repo-type dataset   --revision a36b289eb6d6d7c6b36e8ee11f7c8a03ac3730f6   --local-dir "${DATA_DOWNLOAD_DIR:-$PWD/data}" --max-workers 4
