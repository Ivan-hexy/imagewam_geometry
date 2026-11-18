#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MS_HUB="${MS_HUB:-.download-venv/bin/ms-hub}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
revision="$($PYTHON_BIN -c 'import json; print(json.load(open("configs/dataset_release.json"))["revision"])')"
destination="${DATA_DOWNLOAD_DIR:-$PWD/data/simple_rgbd_1aa44615}"
# Never silently overlay different episode numbering onto an existing snapshot.
"$PYTHON_BIN" - "$destination" <<'PY'
import hashlib,json,sys
from pathlib import Path
release=json.loads(Path('configs/dataset_release.json').read_text());root=Path(sys.argv[1])
for relative,expected in release['files'].items():
    p=root/relative
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'Refusing to mix dataset versions at {p}. Set DATA_DOWNLOAD_DIR to a new directory.')
PY
"$MS_HUB" download ivanhe666/simple_rgbd --repo-type dataset --revision "$revision" --local-dir "$destination" --max-workers 4
echo "Dataset downloaded to $destination; configure SIMPLE_RGBD_ROOT and SIMPLE_RGBD_CACHE together if using a custom location."
