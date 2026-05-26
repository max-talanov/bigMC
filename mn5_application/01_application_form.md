# RES "Acceso de Excelencia" — Application Form Data

**Call**: Red Española de Supercomputación — Acceso de Excelencia
**Cycle**: [Next quarterly cycle — Feb / Jun / Oct / Dec 2026]
**Target system**: MareNostrum 5 (BSC-CNS, Barcelona)

---

## A. Administrative information

| Field | Value |
|-------|-------|
| **Project title** | BigStroke-MN5: Multi-scale bio-plausible simulation of stroke rehabilitation under combined cortical (TMS) and spinal (epidural SCS) neurostimulation |
| **Project acronym** | BigStroke-MN5 |
| **Scientific area** | BIO — Biomedicine and Life Sciences (Computational Neuroscience) |
| **Sub-area** | Multi-scale brain simulation / Neurorehabilitation modeling |
| **Duration** | 12 months (4 cycles × 3 months extension if results warrant) |
| **Confidentiality** | Public abstract; full proposal restricted |

## B. Principal Investigator

| Field | Value |
|-------|-------|
| Name | Max Talanov |
| Position | [PI title — to be filled by applicant] |
| Affiliation | [Institution — to be filled] |
| Department | memBrain Computational Neuroscience Group |
| ORCID | [to be filled] |
| Email | [to be filled] |
| Phone | [to be filled] |
| Address | [to be filled] |
| H-index | [to be filled] |

## C. Co-Investigators

| Name | Role | Affiliation | Expertise |
|------|------|-------------|-----------|
| [Co-PI 1] | Senior researcher | [Inst.] | Computational neuroscience / NEST |
| [Co-PI 2] | Clinical collaborator | [Hospital] | Stroke rehabilitation / NIBS |
| [Co-PI 3] | HPC engineer | [Inst.] | MPI scaling / NEST optimization |

## D. Resources requested

| Resource | Quantity | Justification |
|----------|----------|---------------|
| **GPP node-hours** | 100,000 | NEST microcircuit + spinal CPG (CPU-bound) |
| **ACC GPU-hours** | 20,000 | NEURON+CoreNEURON detailed motoneuron pools on H100 |
| **Total core-hours** | ~7,200,000 | (100,000 nodes × 112 cores) + (20,000 × 4 GPUs) |
| **Storage (GPFS)** | 20 TB | Spike rasters, weight matrices, BOLD timeseries |
| **Wall-time/job** | 24 h | Single condition simulation |
| **Concurrent jobs** | 16 | Latin-Hypercube sampled parameter sweep |

## E. Activity allocation request

- **Cycle 1 (mo. 1–3)**: 25% allocation — pilot/validation runs (≈1.8 M core-h)
- **Cycle 2 (mo. 4–6)**: 35% allocation — full healthy/stroke comparison (≈2.5 M)
- **Cycle 3 (mo. 7–9)**: 30% allocation — NIBS protocol sweep (≈2.2 M)
- **Cycle 4 (mo. 10–12)**: 10% allocation — sensitivity analysis + reproductions (≈0.7 M)

## F. Related funding

| Source | Status | Role |
|--------|--------|------|
| [National grant if any] | [active/pending] | Salary support |
| EBRAINS / Human Brain Project legacy infrastructure | Open access | TVB / NEST tooling |
| Phase 1–4D groundwork (GitHub: max-talanov/bigMC) | Self-funded | Phenomenological proof of concept |

## G. Ethics & data management

- **Ethics**: No human/animal subjects in computational modeling. All parameters derived from published, peer-reviewed literature.
- **Data plan**: All code released under MIT license at https://github.com/max-talanov/bigMC; simulation outputs deposited at EBRAINS Knowledge Graph upon publication.
- **Reproducibility**: NESTML model files, Snakemake pipeline definitions, and singularity containers will be released alongside results.

## H. Signatures required

- [ ] PI signature
- [ ] Institution legal representative
- [ ] BSC liaison endorsement (recommended but optional at Excelencia level)
