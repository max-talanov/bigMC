# Archived Phase 4D iterations

These files are **frozen, superseded drafts** kept for reference only. They were
intermediate iterations toward the rehabilitation-protocol comparison figure,
now produced by the single canonical file in the repo root:

> **`tvb_phase4d_rehab_comparison.py`** (formerly `tvb_phase4d_step2_overlay.py`)

| File | Was | Superseded because |
|------|-----|--------------------|
| `tvb_phase4d_multimodal_rehab.py` | original 0–12 wk, 4 protocols (k=0.70) | timeline + figure folded into the canonical comparison |
| `tvb_phase4d_slow_recovery.py` | "Step 1", k=0.30 variant | k=0.30/0.18 now in the canonical file |
| `tvb_phase4d_step2_detailed.py` | electrical-activity figure draft | one of four figure iterations |
| `tvb_phase4d_step2_extended.py` | 24-week figure draft | one of four figure iterations |
| `tvb_phase4d_step2_phase2style.py` | phase-2-style figure draft | one of four figure iterations |

**Note:** these scripts are *not runnable as-is from this folder* — they load
`tvb_phase2_spinal_cpg.py` via a path relative to their own location, which now
resolves inside `archive/`. They are retained for history/reference; git history
also preserves every prior version. Do not import or extend them — use the
canonical pipeline (see `../PIPELINE.md`).
