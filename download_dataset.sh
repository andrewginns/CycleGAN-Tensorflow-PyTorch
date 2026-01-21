#!/usr/bin/env bash
set -euo pipefail

# Legacy wrapper kept for compatibility.
# Uses Python so we don't depend on `unzip` and can handle URL changes.

python3 ./download_dataset.py "$@"
