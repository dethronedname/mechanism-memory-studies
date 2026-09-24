#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
OUT="${1:-runs/time_gpu_01}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/run_t1.py verify || exit $?
python -m pytest -q || exit $?
python scripts/run_t1.py run --stage gpu --device cuda:0 --out "$OUT"
code=$?
# No retries. Collect successes, failures and unrun configurations alike.
python scripts/run_t1.py collect --run "$OUT" --out "returns/$(basename "$OUT")_lite.zip"
collect_code=$?
if [ "$collect_code" -ne 0 ]; then exit "$collect_code"; fi
exit "$code"
