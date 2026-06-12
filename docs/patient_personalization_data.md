# Patient Personalization — Inbound Data Sources

How to constrain each layer of the BigStroke pipeline to an *individual*
patient. The model is multi-scale, so personalization data is multi-modal:
each layer has one or more clinical/imaging sources that fix its parameters.

Legend for acquisition timing:
**A** = acute (hours–days), **S** = subacute (weeks), **L** = longitudinal (repeated).

---

## Layer 1 — Whole-brain structure, connectome & lesion (TVB)

| Data source | Modality | Constrains in our model | Timing | Key ref |
|---|---|---|---|---|
| **T1-weighted MRI** | Structural MRI | Subject anatomy & region parcellation (the TVB node set) | A/S | — |
| **Diffusion MRI (DTI/DWI) + tractography** | dMRI | Structural connectome = TVB coupling matrix; **CST fractional anisotropy (FA) & FA-asymmetry ratio** → descending drive `c_L` | A/S | Stinear 2007 |
| **Lesion mask** | T1 / FLAIR / DWI | Which regions are core vs spared → `STROKE_FRACTION`, which M1 neurons silenced | A | — |
| **Perfusion imaging (PWI / CT-perfusion)** | PWI / CTP | **Diffusion–perfusion mismatch = penumbra vs core** → our core/penumbra split & `STROKE_CORE_FRACTION` | A | — |
| **Lesion volume** | T1/FLAIR | Severity & recovery-ceiling modulation | A | — |

## Layer 2 — Functional connectivity (Phase 4 BOLD/FC)

| Data source | Modality | Constrains | Timing | Key ref |
|---|---|---|---|---|
| **Resting-state fMRI** | rs-fMRI | Functional connectivity matrix; **interhemispheric M1–M1 FC** (our Phase 4C target) | S/L | Carter 2010 |
| **Task fMRI (affected-limb motor task)** | task fMRI | Motor-map location, **activation laterality index**, peri-lesional recruitment | S/L | — |

## Layer 3 — Corticospinal integrity & cortical excitability (drive `c`, NIBS dose)

| Data source | Modality | Constrains | Timing | Key ref |
|---|---|---|---|---|
| **TMS motor evoked potentials (MEP)** | TMS-EMG | **MEP±** and amplitude = CST functional integrity → sets `c_L` / recruitment severity (MEP− ⇒ severe/flaccid) | A/S | Stinear/PREP2 |
| **Resting motor threshold (RMT)** | TMS | Cortical excitability → `tDCS`/`rTMS` dose, neuron excitability params | A/S | — |
| **Interhemispheric inhibition / ipsilateral silent period** | paired-pulse TMS | **Diaschisis & interhemispheric balance** → callosal/diaschisis terms | S | Murase 2004 |
| **EEG (peri-lesional delta, spectral)** | EEG | Network state, excitability; cheap & bedside | A/S | — |

## Layer 4 — Spinal output & the force model (Phase 2/2B)

| Data source | Modality | Constrains | Timing | Key ref |
|---|---|---|---|---|
| **Surface EMG, affected vs unaffected legs** | sEMG | Muscle activation envelopes & timing → validates motoneuron-pool output, co-contraction | A/S/L | — |
| **Dynamometry / force plates** | force | **Peak force & force-asymmetry index = our AmpAI**; sets `FORCE_MAX`, recruitment severity | A/S/L | — |
| **Instrumented gait (3D mocap, GRF, treadmill)** | kinematics | Step frequency, stance/swing, ground reaction forces → validates gait output | S/L | Olney & Richards 1996 |
| **H-reflex / M-wave** | electrophysiology | Spinal motoneuron excitability → spinal CPG / `C_CRIT` recruitment threshold | S | — |

## Layer 5 — Clinical scores (severity anchoring & recovery curve)

| Data source | What it is | Constrains | Timing |
|---|---|---|---|
| **Fugl-Meyer Assessment (FMA)** | Motor impairment 0–66 (LL) / 0–100 | Maps to `STROKE_SEVERITY` (healthy→complete) | A/S/L |
| **NIHSS** | Stroke severity at admission | Initial severity, recovery prior | A |
| **MRC manual muscle test (0–5)** | Per-muscle strength | Per-muscle recruitment / severity | A/S/L |
| **Modified Ashworth** | Spasticity / tone | Later-phase tone (post-flaccid) | S/L |
| **Gait speed (10 m), 6-min walk** | Functional output | Validation endpoint | S/L |
| **Modified Rankin (mRS)** | Global disability | Outcome anchor | L |

## Layer 6 — Demographics & timing (recovery-rate modulators)

Age · time-since-stroke · lesion volume · premorbid fitness · comorbidities
→ modulate the logistic recovery rate `k` and the recovery ceiling.

---

## Minimum-viable personalization set

If full multimodal imaging is unavailable, the smallest set that meaningfully
individualizes the model:

1. **Lesion mask** (T1/FLAIR) → which regions are hit
2. **CST integrity** — either DTI FA-asymmetry **or** TMS MEP± → descending drive `c_L`
3. **Fugl-Meyer (or MRC)** → severity bin (`STROKE_SEVERITY`)
4. **Force/dynamometry or instrumented gait** → AmpAI target for validation
5. **Age + time-since-stroke** → recovery-rate prior

This mirrors the clinically-validated **PREP2 algorithm** (Stinear et al. 2017),
which combines **S**houlder-**A**bduction-**F**inger-**E**xtension score + **TMS MEP**
+ age + NIHSS to predict upper-limb outcome — a real precedent for the
imaging-light personalization path.

---

## Direct map: data source → our code parameters

| Our parameter (file) | Personalized from |
|---|---|
| TVB coupling matrix (Phase 1) | DTI tractography connectome |
| Silenced regions / `STROKE_FRACTION` | Lesion mask (T1/FLAIR/DWI) |
| `STROKE_CORE_FRACTION` (core vs penumbra) | PWI/CTP diffusion–perfusion mismatch |
| `c_L` descending drive (`tvb_phase2_spinal_cpg.py`) | CST FA-asymmetry **or** TMS MEP amplitude |
| `STROKE_SEVERITY` bin | Fugl-Meyer / MRC / NIHSS |
| `C_CRIT`, `K_RECRUIT` recruitment (Phase 2B) | H-reflex + dynamometry recruitment curve |
| `FORCE_MAX`, AmpAI target | Force plates / dynamometry |
| Interhemispheric / diaschisis terms | rs-fMRI M1–M1 FC; paired-pulse TMS |
| tDCS/rTMS dose (Phase 4D) | RMT, E-field modeling on subject T1 |
| Recovery rate `k`, ceiling (Phase 3/4D) | Age, time-since-stroke, lesion volume, FMA trajectory |

---

## Acquisition timeline (practical protocol)

- **Acute (0–7 d):** T1/FLAIR/DWI + PWI (lesion & penumbra), NIHSS, TMS MEP±, baseline dynamometry → *initialize* the model.
- **Subacute (1–12 wk):** DTI (CST FA), rs-fMRI, FMA, instrumented gait, EMG → *refine* drive/FC/severity.
- **Longitudinal (repeat):** FMA, gait speed, force, (optional) rs-fMRI → *fit* recovery rate `k` and *validate* the predicted trajectory.

---

## Who provides each data type (IRCCS neurorehabilitation hospital)

CST is the *structure*; **DTI-FA** measures its structural integrity and
**TMS-MEP** its functional integrity — request both, they can disagree
(structurally thinned but still conducting, or vice versa), and the
disagreement is itself informative.

| Data type | One-line description | Model use | Unit (IT name) | Who runs it |
|---|---|---|---|---|
| **CST** | Main motor pathway, cortex→cord; top recovery predictor | sets `c_L` / severity | — (measured via DTI or TMS) | — |
| **DTI-FA** | MRI sequence; FA(0–1) of the CST, lesioned-vs-intact **asymmetry ratio** | continuous → `c_L`, severity | Neuroradiology (*UOC Neuroradiologia*) | MRI techs acquire; tractography/FA by neuroradiologist or **imaging-research / MRI physicist / bioengineer** |
| **TMS-MEP** | Magnetic pulse over M1; EMG twitch = **MEP± / amplitude / RMT** | MEP− ⇒ flaccid; amp+RMT → `c_L`, NIBS dose | Clinical Neurophysiology (*Neurofisiopatologia*) | neurophysiologist / neurologist (TMS-EMG) |
| **Lesion mask** | Dead-tissue outline on T1/FLAIR/DWI | regions silenced, `STROKE_FRACTION` | Neuroradiologia | neuroradiologist |
| **PWI** | Perfusion map; DWI–PWI mismatch = penumbra | core-vs-penumbra split | Neuroradiologia | neuroradiologist (research MRI) |
| **rs-fMRI** | Resting co-activation; interhemispheric M1 FC | diaschisis / FC targets | Neuroradiologia (research MRI) | imaging-research team |
| **sEMG** | Surface muscle activity, affected vs unaffected | validates motoneuron output | Neurofisiopatologia / gait lab | neurophysiology tech / bioengineer |
| **Dynamometry / force plates / 3D gait** | Measured force & gait asymmetry | = our **AmpAI**, `FORCE_MAX` | Movement-analysis lab (*Laboratorio di Analisi del Movimento*) | bioengineers / movement scientists |
| **Fugl-Meyer / MRC / NIHSS / gait speed** | Clinical motor / strength / severity scores | severity bin, recovery anchor | Physiatry & Neurology (*Medicina Fisica e Riabilitazione, Neurologia*) | physiatrists / physiotherapists |

**Three units to approach:**
1. **Neuroradiology research group** → DTI-FA tractography, lesion mask, PWI, rs-fMRI (one MRI session yields most imaging).
2. **Clinical Neurophysiology (TMS lab)** → MEP±, RMT (also your future NIBS/tDCS partner).
3. **Movement-analysis lab + physiatry** → force/gait/AmpAI and clinical scores (Fugl-Meyer).

> Note: DTI-FA tractography and rs-FC are usually **research-grade processing**,
> not routine clinical reports — collaborate with the imaging *research* team
> for those two, not the standard radiology service.

---

## Data request templates

Short notes to send each unit. Replace **[patient/cohort]** and contact details.

### EN — Neuroradiology
> We are building a patient-specific computational model of post-stroke motor
> recovery. For **[patient/cohort]** we would need, from a single MRI session:
> (1) a **lesion mask** (T1/FLAIR or DWI); (2) **DTI** with **CST tractography**
> and the **fractional-anisotropy (FA) asymmetry ratio** (affected vs unaffected
> hemisphere); and, if feasible, (3) **resting-state fMRI** for interhemispheric
> M1 connectivity and (4) **perfusion (PWI)** for the penumbra. We are happy to
> collaborate on the tractography/FC processing. Output as NIfTI + a short table
> of FA values.

### EN — Clinical Neurophysiology (TMS)
> For the same model we need **TMS motor-evoked potentials** of the affected
> lower limb: **MEP presence/absence (MEP±)**, **MEP amplitude**, and **resting
> motor threshold (RMT)**, ideally bilaterally. These set the descending-drive
> parameter and, later, the NIBS dosing. A short table per session is sufficient.

### EN — Movement-analysis lab & Physiatry
> We need the patient's **motor output**: **peak force / dynamometry** and/or
> **instrumented gait** (step frequency, ground reaction forces, left–right
> **force-asymmetry index**) plus **surface EMG** of leg flexors/extensors; and
> the clinical scores **Fugl-Meyer (lower limb)**, **MRC strength**, and **gait
> speed (10 m)**. These calibrate and validate the force model.

### IT — Neuroradiologia
> Stiamo sviluppando un modello computazionale paziente-specifico del recupero
> motorio post-ictus. Per **[paziente/coorte]** servirebbero, da una singola
> sessione RM: (1) una **maschera della lesione** (T1/FLAIR o DWI); (2) **DTI**
> con **trattografia del tratto corticospinale (CST)** e il **rapporto di
> asimmetria della frazione di anisotropia (FA)** (emisfero affetto vs sano);
> e, se possibile, (3) **fMRI a riposo** per la connettività interemisferica di
> M1 e (4) **perfusione (PWI)** per la penombra. Disponibili a collaborare sul
> processing di trattografia/connettività. Output in NIfTI + tabella dei valori FA.

### IT — Neurofisiopatologia (TMS)
> Per lo stesso modello servono i **potenziali evocati motori (MEP) da TMS**
> dell'arto inferiore affetto: **presenza/assenza del MEP (MEP±)**, **ampiezza
> del MEP** e **soglia motoria a riposo (RMT)**, idealmente bilaterali.
> Definiscono il parametro di drive discendente e, in seguito, il dosaggio NIBS.
> È sufficiente una breve tabella per sessione.

### IT — Laboratorio di Analisi del Movimento & Fisiatria
> Servono i dati di **output motorio**: **forza di picco / dinamometria** e/o
> **analisi strumentale del cammino** (cadenza, forze di reazione al suolo,
> **indice di asimmetria della forza** destra–sinistra), più **EMG di superficie**
> dei flessori/estensori dell'arto inferiore; e i punteggi clinici **Fugl-Meyer
> (arto inferiore)**, **forza MRC** e **velocità del cammino (10 m)**. Servono a
> calibrare e validare il modello di forza.

---

## References

- Stinear (2007) *Brain* 130:170 — CST integrity & motor potential
- Stinear et al. (2017) *Ann Clin Transl Neurol* — PREP2 prediction algorithm
- Carter et al. (2010) *Ann Neurol* 67:365 — interhemispheric FC predicts outcome
- Murase et al. (2004) *Ann Neurol* 55:400 — interhemispheric inhibition
- Olney & Richards (1996) *Gait Posture* 4:136 — hemiparetic gait
