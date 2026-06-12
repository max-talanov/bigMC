# BigStroke Pipeline — what to run, in what order

The phenomenological multi-scale stroke→gait→rehabilitation pipeline. One
script per phase; later phases consume the outputs of earlier ones. Run from
the repo root with the project venv active (`source .venv/bin/activate`).

> **Start here:** the entry point that generates the data everything else reads
> is **`tvb_healthy_vs_stroke.py`** (Phase 1). Run it first.

## Run order

| # | Phase | Script | Produces | Reads |
|---|-------|--------|----------|-------|
| 1 | Phase 1 — cortex model | `tvb_motor_cortex_phase1.py` | TVB cortical firing-rate series | — |
| 1 | Phase 1 — **entry point** | **`tvb_healthy_vs_stroke.py`** | `tvb_output/comparison/tvb_{healthy,stroke}_*.h5` (CST drives) | — |
| 2 | Phase 2 — spinal CPG | `tvb_phase2_spinal_cpg.py` | `tvb_output/phase2/` gait/force figure; **canonical `simulate()`** | Phase 1 `.h5` |
| 2B | Phase 2B — stroke severity | `tvb_phase2b_stroke_severity.py` | `tvb_output/phase2b/` flaccid-severity figure | imports Phase 2 |
| 3 | Phase 3 — rehabilitation | `tvb_phase3_rehabilitation.py` | `tvb_output/phase3/` recovery figure | imports Phase 2 |
| 4 | Phase 4 A–C — BOLD/FC/NIBS | `tvb_phase4_bold_nibs.py` | `tvb_output/phase4/` BOLD + FC + NIBS figures | imports Phase 2 |
| 4D | Phase 4D — protocol comparison | `tvb_phase4d_rehab_comparison.py` | `tvb_output/phase4d_overlay/` unified A–F comparison | imports Phase 2 |

## The one simulator that matters

`tvb_phase2_spinal_cpg.py::simulate()` is the **single canonical CPG + muscle
model**. Every downstream phase imports and calls it, so improvements made there
(e.g. the Phase 2B motoneuron-recruitment / flaccid-stroke model, and the
optional epidural-SCS drive) propagate everywhere automatically.

If you add a new rehabilitation experiment, **call `p2.simulate(...)`** — do not
copy the integration loop into a new file. (That copy-paste forking is exactly
what `archive/` cleaned up.)

`simulate()` key options:
- `recruit=True` — motoneuron-pool recruitment gain (flaccid below `C_CRIT`)
- `scs_amp_L/scs_freq_L`, `scs_amp_R/scs_freq_R` — epidural SCS sinusoidal drive

## Severity presets

`tvb_phase2_spinal_cpg.py::STROKE_SEVERITY` maps clinical severity →
descending-drive scale: `healthy / mild / moderate / severe / complete`.
`mild` reproduces the original Phase-1 TVB deficit (back-compatible);
`severe`/`complete` cross the recruitment threshold into a flaccid limb.

## Archived / superseded

Earlier Phase 4D drafts live in `archive/` (see `archive/README.md`). They are
frozen references, not part of the live pipeline.

## The other track

`bigstroke_mn5/` is the separate spiking-network (NEST) track aimed at
MareNostrum 5 — independent of this phenomenological pipeline. See
`bigstroke_mn5/README.md`.
