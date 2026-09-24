#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-runs/confirm_gpu_01}"
python scripts/run.py verify || exit $?
python -m pytest -q || exit $?
python scripts/run.py run --stage gpu --device cuda:0 --out "$RUN"
rc=$?
# Collect even after a scientific or numerical failure; never run another experiment automatically.
if [ -f "$RUN/protocol.json" ]; then
  python scripts/run.py collect --run "$RUN" --out "returns/$(basename "$RUN")_lite.zip"
  cr=$?
  if [ "$rc" -eq 0 ] && [ "$cr" -ne 0 ]; then rc=$cr; fi
fi
exit "$rc"
