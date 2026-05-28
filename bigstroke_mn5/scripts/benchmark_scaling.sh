#!/usr/bin/env bash
# Strong-scaling benchmark for the M1 microcircuit.
# Runs the same problem at increasing node counts to measure efficiency.
#
# Submit from MN5 with:  sbatch scripts/benchmark_scaling.sh
#
# Output: bigstroke_mn5/output/scaling_benchmark.csv

#SBATCH --job-name=bigstroke_bench
#SBATCH --output=output/bench-%j.out
#SBATCH --time=04:00:00
#SBATCH --qos=gp_bsc

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc openmpi nest python

OUT=output/scaling_benchmark.csv
echo "nodes,ranks_per_node,sim_time_s,wall_s,rtf" > "$OUT"

for NODES in 1 2 4 8 16; do
    echo ""
    echo "── Benchmark: $NODES nodes ──"
    srun --nodes="$NODES" --ntasks-per-node=4 --cpus-per-task=28 \
        python -m bigstroke_mn5.m1_microcircuit.simulate \
            --scale 1.0 \
            --sim-time-ms 1000 \
            --label "bench_n${NODES}" \
            --threads 28 \
        | tee "output/bench_n${NODES}.log"

    # Parse wall time and RTF from the log
    WALL=$(grep "DONE" "output/bench_n${NODES}.log" | awk '{print $3}')
    RTF=$(grep "real-time factor" "output/bench_n${NODES}.log" | awk '{print $NF}' | sed 's/x//')
    echo "$NODES,4,1.0,$WALL,$RTF" >> "$OUT"
done

echo ""
echo "Benchmark complete.  See $OUT."
