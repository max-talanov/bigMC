# BigStroke-MN5 — RES "Acceso de Excelencia" Application

This folder contains the draft application package for accessing MareNostrum 5 through the **Red Española de Supercomputación (RES) "Acceso de Excelencia"** track, in support of the BigStroke-MN5 multi-scale stroke rehabilitation simulation project.

## Files

| File | Purpose | Submission target |
|------|---------|-------------------|
| `01_application_form.md` | Administrative metadata (PI, team, resources, ethics) | RES web form fields |
| `02_scientific_case.md` | Full scientific proposal (~10 pages, hypotheses, methods, references) | PDF attachment |
| `03_technical_case.md` | HPC / software / scalability justification | PDF attachment |
| `04_executive_summary.md` | One-page abstract for reviewers | Cover page |

## Status

**Current**: Draft v1 — content complete, awaiting PI fill-in of personal/institutional details (marked `[to be filled]`).

**Next steps** (PI tasks):
- [ ] Fill in administrative fields in `01_application_form.md`:
  - PI institutional affiliation, position, ORCID, H-index
  - Co-PIs (HPC engineer, clinical collaborator, spinal-cord modeller)
  - Address and contact details
- [ ] Identify and contact potential co-investigators (recommended profiles in the file).
- [ ] Request letters of support:
  - BSC scientific liaison (helps with feasibility endorsement)
  - Clinical neurology / rehab dept partner (validates clinical relevance)
  - tvb-multiscale developers (TVB consortium endorsement)
  - Spinal-cord modelling collaborator (optional but strengthens proposal)
- [ ] Verify next RES "Acceso de Excelencia" cycle deadline. RES typically has 3–4 cycles/year; check at https://www.res.es
- [ ] Convert markdown to required PDF format (RES uses standard letter PDF).
- [ ] Final sign-off from institutional legal representative.

## Submission cycle reference

RES "Acceso de Excelencia" calls have been historically:
- Cycle openings: February, June, October
- Decision turn-around: ~6–8 weeks
- Allocations: 4-month execution blocks, renewable up to 3 cycles

Verify current schedule at: https://www.res.es/es/acceso

## Pre-submission internal review checklist

Before submitting to RES:
- [ ] Senior advisor / colleague reviews scientific case for novelty
- [ ] HPC engineer reviews technical case for resource estimates
- [ ] Clinical collaborator reviews validation criteria
- [ ] All references checked and cross-linked
- [ ] Executive summary fits on 1 page when rendered to PDF
- [ ] Full proposal (scientific + technical) fits within RES page limits (typically 15 pages)

## Companion preliminary work

The application heavily references the preliminary phenomenological work in the parent directory:
- Phase 1: TVB cortex stroke (`tvb_motor_cortex_phase1.py`, `tvb_healthy_vs_stroke.py`)
- Phase 2: Spinal CPG (`tvb_phase2_spinal_cpg.py`)
- Phase 3: Rehabilitation recovery (`tvb_phase3_rehabilitation.py`)
- Phase 4A–C: BOLD / FC / NIBS (`tvb_phase4_bold_nibs.py`)
- Phase 4D: Multi-protocol comparison (`tvb_phase4d_*.py`)

These results constitute the "Preliminary Data" section that anchors the proposal in demonstrated feasibility.

## Notes on adapting to other HPC tracks

If RES Excellence is not the right fit (e.g., scale too large or wrong area), the same content can be adapted to:
- **PRACE Tier-0** (formerly): cross-European competitive access; longer review cycles, larger awards
- **EuroHPC Joint Undertaking** (Extreme Scale Access, Regular Access): pan-European; LUMI, Leonardo, MareNostrum 5
- **EBRAINS computing time** (HBP legacy): if framed primarily as neuroscience tool development
- **RES "Clase A/B/C"**: smaller allocations, simpler application; suitable as a backup or scale-down path

## License

This application package is **internal / pre-submission** material. Once accepted (or rejected), the proposal will be incorporated into project documentation.

The underlying code and methods are MIT-licensed (see top-level project repo).
