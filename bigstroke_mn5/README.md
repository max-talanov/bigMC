# BigStroke-MN5 — Code Skeleton

NEST-based multi-scale stroke simulation infrastructure, intended for production
runs on MareNostrum 5 but designed to be **runnable at reduced scale on a
laptop or workstation** for development and testing.

## Status

**Stage 1** (current) — M1 microcircuit skeleton:
- Bilateral Potjans-Diesmann 2014 cortical column
- 8 populations × 2 hemispheres = 16 sub-populations
- Scalable from 1% (laptop, ~1,500 neurons) to 100% (MN5, ~150,000 neurons)
- Background Poisson input; spike recording; basic firing-rate analysis
- Stroke hook: silence-zone implementation in left M1

**Stage 2** (planned) — Spinal CPG (V0–V3 + MN pools).

**Stage 3** (planned) — TVB ↔ NEST coupling via `tvb-multiscale`.

**Stage 4** (planned) — STDP plasticity, biophysical NIBS, full pipeline.

## Layout

```
bigstroke_mn5/
├── m1_microcircuit/
│   ├── __init__.py
│   ├── params.py          # All model parameters (size, connectivity, neurons)
│   ├── network.py         # Build neurons + connections in NEST
│   ├── stimulus.py        # Background Poisson + (later) NIBS injection
│   ├── recording.py       # Spike detectors, voltage meters
│   ├── simulate.py        # Main simulation driver
│   └── analysis.py        # Firing rates, ISI CV, raster plots
├── scripts/
│   ├── run_local.sh       # Small-scale (1%) run on laptop
│   ├── run_mn5.sh         # SLURM submission for MareNostrum 5
│   └── benchmark_scaling.sh
├── tests/
│   └── test_network_build.py
└── output/                # Created at runtime; results, figures
```

## Quick start (local, reduced scale)

```bash
# 1. Install NEST (see "NEST installation" below)
# 2. Activate the project venv
source ../.venv/bin/activate

# 3. Run at 1% scale (~1,500 neurons, ~30 s wall time)
bash scripts/run_local.sh
```

This produces `output/raster_local.png` and `output/firing_rates_local.csv`.

## NEST installation

NEST can be tricky to install. Three options, in order of preference:

### Option A — pip wheel (easiest, often works)
```bash
pip install nest-simulator
```
Works on most Linux x86_64. May not work on macOS arm64 or Python 3.14.

### Option B — conda
```bash
conda install -c conda-forge nest-simulator
```
Reliable cross-platform; recommended for macOS / Apple Silicon.

### Option C — build from source (production / HPC)
```bash
git clone --depth 1 -b v3.7 https://github.com/nest/nest-simulator
cd nest-simulator
cmake -DCMAKE_INSTALL_PREFIX=$HOME/nest -Dwith-mpi=ON .
make -j4 && make install
source $HOME/nest/bin/nest_vars.sh
```
On MN5: NEST is pre-installed in the BSC software stack:
```bash
module load nest/3.7
```

## Scaling parameter

Edit `m1_microcircuit/params.py`:
```python
NETWORK_SCALE = 0.01   # 1% scale, ~1,500 neurons   (laptop quick test)
NETWORK_SCALE = 0.1    # 10% scale, ~15,000 neurons (workstation)
NETWORK_SCALE = 1.0    # full Potjans-Diesmann × 2  (MN5 production)
```

The connection probabilities scale appropriately to maintain mean indegree.

## Validation targets (when fully scaled)

| Population | Target firing rate (Hz) | Source |
|------------|--------------------------|--------|
| L2/3e | 0.74 | Potjans-Diesmann 2014 Fig. 6 |
| L4e   | 4.40 | "" |
| L5e   | 6.78 | "" |
| L6e   | 1.10 | "" |
| All inhibitory | 5–9 | "" |

When the skeleton is properly tuned (Stage 1 milestone), these rates should
emerge spontaneously from background Poisson input alone.

## Roadmap to MN5 production

| Step | Status | Outcome |
|------|--------|---------|
| 1. Local 1% run | ⏳ Stage 1 | Validate code paths |
| 2. Local 10% run | ⏳ Stage 1 | Validate firing rates |
| 3. Single-node 100% run | ⏳ Stage 1 | Memory profile |
| 4. Multi-node MPI scaling | ⏳ Stage 1 | Strong-scaling curve |
| 5. Add spinal CPG | Stage 2 | Coupled cortex+spine |
| 6. Add TVB integration | Stage 3 | Full multi-scale |
| 7. Add STDP + NIBS | Stage 4 | Production runs |

## References

- Potjans, T.C., Diesmann, M. (2014). The cell-type specific cortical microcircuit. *Cerebral Cortex* 24:785–806.
- Gewaltig, M.-O., Diesmann, M. (2007). NEST (NEural Simulation Tool). *Scholarpedia* 2(4):1430.
- Jordan et al. (2018). Extremely scalable spiking neuronal network simulation. *Frontiers in Neuroinformatics* 12:2.
