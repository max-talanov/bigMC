# Data Request — Patient Personalization of the BigStroke Model

**To:** Neuroimaging and Advanced Neurophysiology and Neurorehabilitation Models
Department, IRCCS Centro Neurolesi "Bonino-Pulejo"
**From:** [PI name], memBrain Computational Neuroscience Group
**Re:** Per-patient data to individualize a multi-scale stroke→gait→
rehabilitation simulation
**Date:** [date]

---

## 1. Purpose (one paragraph)

We have a working, multi-scale computational model that simulates, for a single
patient, the chain **cortical stroke → corticospinal drive → spinal central
pattern generator → leg muscle force → gait asymmetry → rehabilitation
recovery** under four neurostimulation protocols (TMS, epidural SCS, combined,
sequential). The model currently runs on literature-average parameters. To turn
it into a **patient-specific digital twin** we need a defined set of imaging,
neurophysiology, and clinical measurements. This document lists exactly what we
need, which model parameter each item fixes, and the format/timepoints. Your
department covers the great majority of these in a single collaboration.

## 2. What each measurement personalizes (current model parameters)

| # | Measurement | Model parameter it sets | File |
|---|-------------|--------------------------|------|
| 1 | **Lesion mask** (T1/FLAIR/DWI) + **lesion volume** | which regions silenced; recovery-rate / ceiling prior | Phase 1 |
| 2 | **DTI – CST tractography**, fractional-anisotropy (FA) **asymmetry ratio** (affected vs intact) | descending drive `c_L`; acute severity bin `ACUTE_SEVERITY` | Phase 2 / 4D |
| 3 | **Perfusion (PWI/CTP)**, acute — diffusion–perfusion mismatch | ischemic-core vs penumbra ratio `STROKE_CORE_FRACTION` | Phase 1/2 |
| 4 | **Resting-state fMRI** — interhemispheric M1–M1 functional connectivity | diaschisis / interhemispheric coupling terms | Phase 4 |
| 5 | **TMS — MEP** (presence/absence, amplitude) and **resting motor threshold (RMT)**, affected + intact, lower limb | CST functional integrity → `c_L`; MEP⁻ ⇒ flaccid severity bin | Phase 2/4D |
| 6 | **TMS response** — change in MEP amplitude / RMT after a single rTMS or tDCS session | TMS protocol gain `tms_gain` (recovery-rate boost) | Phase 4D |
| 7 | **H-reflex / motoneuron recruitment curve** (stimulus–response), affected limb | recruitment threshold `C_CRIT`, slope `K_RECRUIT` | Phase 2B |
| 8 | **Dynamometry** — peak isometric force, affected vs intact leg | `FORCE_MAX`, recruitment severity, AmpAI baseline | Phase 2B |
| 9 | **SCS response** (if a transcutaneous/epidural SCS program exists) — force/EMG change with stimulation on vs off | SCS threshold drop `scs_crit_drop`, late gain `scs_gain` | Phase 4D |
| 10 | **Surface EMG** during gait — flexor/extensor, both legs | validates motoneuron-pool output & timing | Phase 2 |
| 11 | **Instrumented gait / force plates** — step frequency, ground-reaction force, left–right **force-asymmetry index** | the model's **AmpAI** validation target; step frequency | Phase 2/4D |
| 12 | **Clinical scales (longitudinal)** — Fugl-Meyer (lower limb), MRC strength, NIHSS, 10-m gait speed | acute severity bin; recovery rate `k`, inflection `t0` | Phase 3/4D |
| 13 | **Demographics** — age, time-since-stroke | recovery-rate `k` and ceiling modifiers | Phase 3/4D |

> Note on items 6 and 9: these are **NIBS *response* measurements**, not just
> baselines. They are what let the four simulated protocols (TMS / SCS /
> combined / sequential) be calibrated to *this* patient rather than to
> literature averages — currently the single biggest source of uncertainty in
> the model.

## 3. Minimum-viable set (if full acquisition is not possible)

In priority order — these five already produce a meaningfully individualized run:

1. **Lesion mask** (item 1) — T1/FLAIR or DWI
2. **CST integrity** — DTI FA-asymmetry (item 2) **or** TMS MEP± (item 5)
3. **Fugl-Meyer lower-limb (+ MRC)** (item 12) — acute severity bin
4. **Dynamometry or instrumented gait** (item 8 or 11) — AmpAI target
5. **Age + time-since-stroke** (item 13)

This mirrors the clinically-validated **PREP2** prediction approach
(SAFE score + TMS MEP + age + NIHSS; Stinear et al. 2017).

## 4. Requested formats & timepoints

- **Imaging (1–4):** NIfTI volumes + a short table of derived scalars
  (CST FA affected/intact and ratio; lesion volume cm³; core/penumbra volumes;
  M1–M1 FC coefficient). Acquire **acute** (0–7 d) where possible for lesion +
  perfusion; **subacute** (1–12 wk) for DTI + rs-fMRI.
- **Neurophysiology (5–9):** per-session tables — MEP amplitude (mV), RMT
  (% stimulator output), H-reflex recruitment (H/M curve points),
  stimulation-on vs -off force/EMG. **Acute + subacute**, repeated if feasible.
- **Motor output (10–11):** EMG envelopes (per muscle) and gait report
  (cadence, GRF, asymmetry index). **Subacute + longitudinal.**
- **Clinical (12–13):** standard scale values at admission and at each
  follow-up. **Longitudinal** (this is what fits the recovery curve).

All data fully de-identified; we handle only derived numerical parameters.
We are glad to collaborate on the DTI tractography and rs-FC processing, which
are research-grade rather than routine clinical readouts.

## 5. What we provide back

A per-patient simulated recovery trajectory under each NIBS protocol, the fitted
parameters, and the predicted protocol ranking — for comparison against the
patient's actual clinical course. All code is open (MIT) and documented
(`PIPELINE.md`, `docs/patient_personalization_data.md`).

---

### Cover note (short form, for email)

> We are personalizing a multi-scale stroke-rehabilitation simulation to
> individual patients and would value a collaboration with your department.
> From a single MRI session we would need a lesion mask, DTI with corticospinal-
> tract FA asymmetry, and — if feasible — resting-state fMRI and perfusion;
> from the neurophysiology lab, TMS MEP/RMT (and ideally the change after a
> single TMS or tDCS session), plus an H-reflex recruitment curve and
> dynamometry of the affected leg; and from the gait lab, an instrumented-gait
> force-asymmetry report. Clinical Fugl-Meyer / MRC / gait-speed scores over
> follow-up complete the set. A minimum-viable subset (lesion mask + CST
> integrity + Fugl-Meyer + a force/gait measure + age/time-since-stroke) is
> already enough to begin. All data de-identified; we work only from derived
> numerical parameters and are happy to co-process the advanced imaging.

---

*Companion: `docs/patient_personalization_data.md` (full data-source mapping,
who-provides-it table, and EN/IT request templates per unit).*
