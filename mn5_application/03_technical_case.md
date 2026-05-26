# BigStroke-MN5 — Technical Case

## 1. Software stack and licensing

| Software | Version | License | Role | Pre-installed on MN5? |
|----------|---------|---------|------|------------------------|
| **NEST** | 3.7+ | GPLv2 | Spiking neural simulator | Yes (BSC software stack) |
| **NESTML** | 6.0+ | GPLv2 | Custom neuron model DSL | Module load |
| **TVB** | 2.10+ | GPLv2 | Whole-brain mesoscale | Pip-installable; tested |
| **tvb-multiscale** | 2.x | GPLv2 | TVB↔NEST coupling | Pip + manual |
| **NEURON** | 8.2+ | BSD | Detailed motoneuron models | Yes (BSC stack) |
| **CoreNEURON** | 1.0+ | BSD | GPU-accelerated NEURON | H100 partition |
| **MPI** | OpenMPI 5 / IntelMPI | open | Process communication | Native |
| **Python** | 3.11 | PSF | Pipeline, analysis | Native |
| **Snakemake** | 8.x | MIT | Workflow manager | Pip |
| **Singularity / Apptainer** | 1.2+ | BSD | Containers for reproducibility | Native |
| **HDF5** | 1.14 | BSD | Output format | Native |
| **NumPy / SciPy / matplotlib** | latest | BSD | Analysis | Native |

**No proprietary software** is required. The complete stack can be reconstructed by any researcher on any HPC site.

## 2. Computational model — resource breakdown

### 2.1 Neuron and synapse counts

| Component | Neurons | Synapses (sparse) | Memory (NEST) |
|-----------|---------|-------------------|----------------|
| M1 microcircuit (Potjans-Diesmann × 2 hemispheres) | 160,000 | 3.0 × 10⁸ | ~24 GB |
| Spinal CPG (L1–S2 × 2 sides) | 160,000 | 5.0 × 10⁸ | ~40 GB |
| Motoneuron pools (multi-compartment, NEURON) | 6,000 | 1.2 × 10⁷ | ~8 GB (per H100) |
| TVB whole-brain (76 ROIs) | (continuous) | (76² connectome) | ~2 GB |
| Coupling buffers, generators, recorders | — | — | ~16 GB |
| **TOTAL per simulation instance** | **~326,000** | **~8.1 × 10⁸** | **~90 GB** |

Comfortable headroom in MareNostrum 5 GPP node memory (256 GB per node).

### 2.2 Single-simulation cost estimate

**Biological time per condition:**
- 24 weeks × 1 representative gait cycle per week × 30 s of simulated activity = **720 s of biological time**, plus 6 × 10 s "stimulation snapshots" between weeks = **780 s total**.
- Simulation timestep: 0.1 ms (NEST default) — well-suited to AdEx + STDP.

**Wall-clock cost (NEST strong-scaling benchmark on similar microcircuits):**
- Potjans-Diesmann on JURECA: real-time-factor ~3–5× (3–5 s wall per 1 s sim) at 1–2 nodes.
- Our doubled cortex + spinal CPG ~3× larger: expected RTF ~10–15 at 4 nodes.
- Therefore: ~780 s × 12 RTF ≈ **9,400 s wall ≈ 2.6 h** per condition on 4 GPP nodes.
- Plus 1 h warm-up / 1 h post-processing = **~5 h wall total** per condition.

**Core-hours per condition** = 5 h × 4 nodes × 112 cores = **2,240 core-h** (round to 2,500 for buffer).

**Total over project**:
- Production runs: 960 × 2,500 = **2.4 M core-h**
- Validation runs: 60 × 2,500 = **0.15 M**
- Sensitivity / Latin-hypercube sweep: 400 × 2,500 = **1.0 M**
- Pilot, debugging, re-runs (20% reserve): **0.7 M**
- **Subtotal CPU**: **~4.3 M GPP core-h**
- GPU motoneuron pool detailed sims: 20,000 ACC GPU-h (~80,000 GPU-core-h equivalent, distinct allocation)

**Conservative total request: 7.2 M core-h.** Allows for unforeseen scaling overhead.

### 2.3 Storage

| Output | Per condition | Total (~1,400 conditions) |
|--------|---------------|----------------------------|
| Spike rasters (HDF5, compressed) | ~6 GB | ~8 TB |
| STDP weight matrices (10 snapshots × 24 weeks) | ~2 GB | ~3 TB |
| BOLD timeseries (76 ROIs × 24 weeks) | ~0.2 GB | ~0.3 TB |
| Motoneuron pool detailed traces | ~1 GB | ~1.4 TB |
| Logs, parameter files, checkpoints | ~0.5 GB | ~0.7 TB |
| Analysis derivatives, figures | — | ~2 TB |
| **Reserve / scratch** | — | ~5 TB |
| **TOTAL** | — | **~20 TB GPFS** |

### 2.4 Parallelization strategy

**NEST**: Hybrid MPI + OpenMP.
- MPI ranks distributed across cores; recommended 4 MPI ranks per GPP node (each owning 28 OpenMP threads) for our problem size.
- 4 GPP nodes per simulation × 16 concurrent simulations = **64 GPP nodes / batch**.
- Tested scaling for Potjans-Diesmann shows ~85% strong-scaling efficiency up to 8 nodes.

**CoreNEURON (GPU)**:
- Motoneuron pool detailed models run separately on H100 nodes; output spike trains passed via file-based handoff to NEST.
- Each H100 handles ~2,000 multi-compartment MNs in real-time. Use 3 H100 nodes per simulation for full bilateral coverage.

**Snakemake**:
- DAG of ~1,400 jobs with explicit dependencies.
- Submits to SLURM with `--cluster` integration.
- Automatic checkpoint and resume on failure.

### 2.5 Scalability evidence

Published benchmarks support feasibility:

| Reference | System | Network | Performance |
|-----------|--------|---------|-------------|
| Jordan 2018 (Frontiers) | JUQUEEN | ~10⁹ neurons, 10¹³ synapses | Real-time-factor ~6 at 1024 nodes |
| Kunkel 2014 (Frontiers) | K computer | ~10⁹ neurons | RTF 3 at 82,944 cores |
| Schirner 2018 (eLife) | Tier-1 | TVB+NEST hybrid | demonstrated coupling |
| Lionheart 2022 | EBRAINS | tvb-multiscale | working code, examples |

Our problem (~3×10⁵ neurons + TVB) is **two to three orders of magnitude smaller** than benchmarked systems, leaving large headroom. MareNostrum 5's modern Sapphire Rapids CPUs and EDR-200 Infiniband further improve over the benchmarks above (which used older Skylake / older interconnects).

## 3. Performance plan and pilot benchmarks (Cycle 1)

Before production runs, we will execute a tiered benchmark in Cycle 1:

| Pilot job | Nodes | Wall-clock | Purpose |
|-----------|-------|------------|---------|
| 1 node, M1 microcircuit only | 1 | 2 h | NEST install & validation |
| 2 nodes, M1 + spinal CPG | 2 | 4 h | Memory & MPI behaviour |
| 4 nodes, full model | 4 | 8 h | Realistic timing |
| 8 nodes, full model | 8 | 4 h | Strong scaling test |
| 16 nodes, full model | 16 | 2 h | Beyond-comfort scaling check |

Total pilot cost: ~20,000 core-h. Outputs: a tuned scaling curve that drives the production allocation decision.

## 4. Reproducibility plan

- **Singularity container** built with full software stack, frozen at submission of each major batch.
- All parameter files version-controlled under git; each output HDF5 stamped with `git rev-parse HEAD`.
- Snakemake DAG dumped per batch.
- All seeds recorded; `numpy.random.seed`, `nest.SetKernelStatus({'rng_seed': N})` etc.
- Public release: code → GitHub, data → EBRAINS Knowledge Graph, container → Sylabs cloud or BSC Singularity registry.

## 5. Data management plan (FAIR)

| FAIR principle | Implementation |
|----------------|----------------|
| **Findable** | DOI per dataset via EBRAINS Knowledge Graph; ORCID linkage |
| **Accessible** | Public, no embargo beyond standard publication acceptance |
| **Interoperable** | NIX/HDF5 formats compatible with major neuroscience tools |
| **Reusable** | Full metadata (parameter sets, software versions); Apache 2.0 / MIT licensing |

## 6. Knowledge transfer and training

- All junior researchers in the team will be trained on MN5 (BSC's user-support helpdesk + PRACE training events).
- Final deliverable includes a **tutorial Jupyter notebook** demonstrating end-to-end usage on a smaller (single-node) instance, so any neuroscience lab can adopt the framework.
- A **workshop** will be organized in month 11 (at BSC or virtually) for stakeholders in the stroke / spinal cord injury communities.

## 7. Why MareNostrum 5 specifically

| Feature | Why it matters |
|---------|----------------|
| 256 GB RAM per GPP node | Our microcircuits fit comfortably with headroom |
| 112 cores per GPP node (Sapphire Rapids) | Excellent OpenMP scaling for NEST |
| H100 GPUs on ACC partition | CoreNEURON detailed MN simulation; fastest hardware available in Europe for HOC models |
| EDR-200 Infiniband | Low-latency MPI for NEST inter-rank spike communication |
| BSC software stack | NEST, NEURON, TVB pre-installed; reduces our setup burden |
| RES "Acceso de Excelencia" track | Designed for projects of this scale; alternative tracks (clase A, B, C) are insufficient |
| Spanish-EU integration with EBRAINS | Aligns with our data deposition plan |

Alternative HPC sites considered:
- **JURECA-DC (FZJ)**: H100 available, but NEST/JURECA workflows now prefer JUWELS Booster which is GPU-only; less ideal for our hybrid model.
- **LUMI-G (CSC)**: AMD MI250X, less mature NEST GPU support.
- **Leonardo (CINECA)**: comparable, but our team's existing Spanish institutional affiliation gives MN5 a logistical edge.

## 8. Risk-adjusted resource flexibility

If allocation granted is less than requested, scope adjusts:
- 70% of request: drop sensitivity-analysis batch (Aim 3 partial). Still publishable.
- 50% of request: also reduce Latin-hypercube to 50 configs. Major paper still achievable.
- 30% of request: Aim 1 + Aim 2 only; deferred Aim 3 to follow-on project.

If allocation granted exceeds expectations:
- Add detailed exploration of spreading depolarization dynamics (Aim 1 extension).
- Add cerebellar contribution to gait via existing TVB cerebellar parcellation.

## 9. Letters of support (to be attached)

- [ ] BSC user support / scientific liaison
- [ ] Co-PI clinical neurology collaborator
- [ ] tvb-multiscale developers (Lionheart group / TVB consortium)
- [ ] Spinal cord modeling collaborator (Danner / Rybak group, optional)

## 10. Summary table

| Item | Value |
|------|-------|
| GPP core-hours requested | 4,300,000 + 20% buffer = **~5.2 M** |
| ACC GPU-hours requested | **20,000** |
| Total normalized | **~7.2 M core-h equivalent** |
| Storage | **20 TB** |
| Duration | **12 months** |
| Concurrent jobs (peak) | **16** |
| Strong-scaling validated up to | **16 nodes / simulation** |
| Software readiness | **High** — all components used in preliminary work |
| Risk profile | **Medium-low** — most components individually validated; novelty is the integration |
