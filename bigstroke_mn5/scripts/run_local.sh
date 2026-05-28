#!/usr/bin/env bash
# Run the M1 microcircuit at 1% scale on a laptop / workstation.
# Expected wall time: ~30 s.  Useful for code-path validation.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Activate the project venv if present
if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

python -m bigstroke_mn5.m1_microcircuit.simulate \
    --scale 0.01 \
    --stroke-fraction 0.0 \
    --sim-time-ms 1000 \
    --label local_1pct \
    --threads 4

# Also run a stroke condition for comparison
python -m bigstroke_mn5.m1_microcircuit.simulate \
    --scale 0.01 \
    --stroke-fraction 0.5 \
    --sim-time-ms 1000 \
    --label local_1pct_stroke \
    --threads 4
