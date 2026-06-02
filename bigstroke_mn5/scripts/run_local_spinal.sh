#!/usr/bin/env bash
# Local spinal CPG run: healthy + stroke (35.6% left descending deficit).
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

echo "── HEALTHY ──"
python -m bigstroke_mn5.spinal_cpg.simulate \
    --scale 0.05 --stroke 0.0 \
    --sim-time-ms 4000 --label local_healthy --threads 4

echo ""
echo "── STROKE (35.6% L deficit) ──"
python -m bigstroke_mn5.spinal_cpg.simulate \
    --scale 0.05 --stroke 0.356 \
    --sim-time-ms 4000 --label local_stroke --threads 4
