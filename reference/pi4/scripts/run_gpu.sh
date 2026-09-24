#!/usr/bin/env bash
set -euo pipefail
# Affects preflight AND child workers; does not change CUDA_VISIBLE_DEVICES.
export NVIDIA_TF32_OVERRIDE=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python scripts/run.py run --stage gpu --device cuda:0 --out "${1:?Supply a new output directory}"
