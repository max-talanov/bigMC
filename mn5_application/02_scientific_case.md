# BigStroke-MN5 — Scientific Case

## Abstract (1 page, public)

Stroke is the leading cause of acquired adult disability worldwide, with >12 million new cases annually and ~50% of survivors left with persistent motor deficits. Despite billions invested in rehabilitation trials of non-invasive brain stimulation (NIBS) — transcranial magnetic stimulation (TMS), transcranial direct-current stimulation (tDCS), and epidural spinal cord stimulation (SCS) — clinical effect sizes remain modest and individual responses highly variable. We hypothesize that this gap stems from a *scale mismatch*: current rehabilitation theory operates at the systems level (cortical excitability, functional connectivity), while the actual recovery substrate is *cellular and synaptic* (spike-timing-dependent plasticity at corticospinal and spinal interneuron synapses).

**BigStroke-MN5** will deliver the first end-to-end, mechanistically-grounded multi-scale simulation of stroke recovery under combined cortical and spinal neurostimulation, spanning four orders of biological scale on a single computational platform:

1. **Whole brain (TVB)** — 76-region mesoscale field with stroke-induced functional disconnection (validated against Carter 2010 fMRI FC data).
2. **Primary motor cortex M1 (NEST microcircuit, 160,000 spiking neurons)** — Potjans-Diesmann 8-population cortical column, with bilateral implementation and stroke-zone ischemic penumbra dynamics.
3. **Lumbosacral spinal CPG (NEST, 160,000 neurons across 2 sides)** — V0_D / V0_V / V1 / V2a / V2b / V3 interneurons and motoneuron pools following Kiehn 2016 and Danner et al. 2019 architectures.
4. **Muscle and gait kinematics** — proxy via motoneuron pool population activity → six muscle group activations → bilateral force asymmetry index.

The simulation will operate over a clinically realistic 24-week (6-month) rehabilitation timeline using spike-timing-dependent plasticity (STDP) at corticospinal and intraspinal synapses. We will compare four NIBS protocols (TMS-only, SCS-only, combined, sequential) against a no-intervention baseline using >960 simulation runs.

**Significance:** Our preliminary phenomenological work (Phases 1–4D, https://github.com/max-talanov/bigMC) demonstrates that the qualitative effects of combined NIBS are *recoverable* at coarse scale, but cannot resolve the cellular mechanisms (excitotoxicity, diaschisis, STDP saturation) that determine *why some patients respond and others do not*. MN5-scale simulation is required to test these mechanistic hypotheses in silico before they reach Phase III trials.

**Expected outcomes:** (i) a publication in *eLife* or *Nature Communications* on the mechanistic basis of NIBS protocol synergy; (ii) an open-source multi-scale stroke simulation platform usable by the broader community; (iii) computationally-predicted optimal NIBS protocols ready for translation to existing clinical centers; (iv) training of [N] junior researchers in HPC computational neuroscience.

**Resources requested:** 7.2 M core-hours on MareNostrum 5 GPP + 20,000 ACC GPU-hours, 20 TB GPFS storage, over 12 months.

---

## 1. Scientific background and motivation

### 1.1 The clinical problem

Stroke affects 12.2 million people per year (GBD 2019); 50–60% of survivors experience persistent hemiparesis, with the upper limb showing recovery in only ~20% of cases by 6 months (Kwakkel 2003). The Copenhagen Stroke Study (Jorgensen 1995) established the canonical recovery curve: rapid neurological improvement in the first 6–11 weeks, slowing to a plateau by 6 months. The neural substrate of this curve remains contested — some authors emphasize cortical reorganization (Cramer 2008), others spinal compensation (Wagner 2018), and still others corticospinal tract integrity (Stinear 2007).

### 1.2 NIBS in stroke: promise and disappointment

Three NIBS modalities have entered clinical practice:

- **rTMS / tDCS** of contralesional or ipsilesional M1 (Hummel & Cohen 2005, Khedr 2005). Meta-analyses (Hsu 2012, Kang 2016) show **modest effect sizes (Cohen's d ≈ 0.4)** with **high heterogeneity** (I² > 60%).
- **Epidural SCS** of the lumbosacral cord (Wagner 2018, Rowald 2022) shows striking single-subject results for spinal cord injury but **only emerging evidence** for stroke gait.
- **Combined / sequential protocols** are theorized to be synergistic (Bonin Pinto 2019) but have rarely been tested due to logistical complexity.

The fundamental open question — **why is the inter-subject variance so large?** — cannot be answered by clinical trials alone. Each patient is a unique configuration of lesion location, residual CST integrity, age, time-since-stroke, premorbid fitness, and rehabilitation adherence. Resolving the contribution of each requires either (a) impossibly large trials (~10⁴ patients) or (b) a mechanistic simulation that can isolate each factor.

### 1.3 Why now? Three converging enablers

1. **The Virtual Brain** (TVB; Sanz Leon 2013, Schirner 2018) provides validated whole-brain mesoscale modeling with patient-specific connectome support. Our Phase 1 already produces ipsilesional/contralesional CST drives matched to literature.

2. **NEST 3.x** (Gewaltig & Diesmann 2007, Plesser 2007) demonstrates near-ideal strong scaling on >10⁵ MPI ranks, with the Potjans-Diesmann cortical microcircuit running in real-time on JURECA / Piz Daint at modest node counts.

3. **TVB-multiscale** (Lionheart et al. 2022, github.com/the-virtual-brain/tvb-multiscale) bridges (1) and (2), allowing mesoscale TVB regions to act as boundary conditions for NEST microcircuits. This integration is mature but **has not yet been applied to stroke or NIBS modeling**.

### 1.4 Our preliminary work

The applicant's group has completed a 4-phase preliminary investigation (Phases 1–4D in https://github.com/max-talanov/bigMC), demonstrating:

- **Phase 1**: TVB simulation of left-M1 ischemia reproduces 35.6% reduction in left CST firing rate, matching Stinear 2007 fractional anisotropy / motor evoked potential ratios.
- **Phase 2**: Bilateral Matsuoka spinal CPG reproduces hemiparetic gait: 19.4% amplitude asymmetry, preserved cadence (commissural entrainment), consistent with Olney & Richards 1996 kinematics.
- **Phase 3**: Logistic recovery model fits Copenhagen Stroke Study curves with parameter k=0.18, t₀=10 weeks.
- **Phase 4A–C**: Balloon-Windkessel BOLD reproduces resting-state FC disruption M1_L↔SMN observed in Carter 2010 (post-stroke FC ≈ 0).
- **Phase 4D**: Four NIBS protocols evaluated phenomenologically; all four show ≥80% amplitude-asymmetry recovery by 16 weeks; **but** all four converge to indistinguishable outcomes because the model lacks a mechanism for late-phase differentiation (STDP, dose-response, individual variability).

The preliminary work has validated the *macroscopic* hypothesis. The MN5 work will test whether it survives at biological scale and *which mechanism explains the convergence* — STDP saturation, ceiling effects, or model degeneracy.

## 2. Hypotheses and specific aims

### 2.1 Central hypothesis

**H0**: Multi-scale bio-plausible simulation of stroke + NIBS reproduces clinical recovery trajectories quantitatively (within 2× of published meta-analytic effect sizes) while exposing cellular mechanisms (excitotoxicity, STDP saturation, diaschisis) invisible to phenomenological models.

### 2.2 Specific aims

**Aim 1 — Build the multi-scale stroke baseline.**
Construct and validate the integrated TVB + NEST-M1 + spinal-CPG simulation in healthy and acute-stroke conditions. **Success criterion**: simulated bilateral M1 BOLD FC, CST firing rates, and gait amplitude asymmetry index match published clinical means within 1 standard deviation.

**Aim 2 — Mechanistically dissect NIBS protocols.**
Run TMS-only, SCS-only, combined, and sequential protocols over a 24-week rehabilitation timeline with STDP-mediated learning. Quantify the contribution of (a) cortical microcircuit excitability, (b) spinal CPG entrainment, (c) STDP-driven synaptic remodeling, and (d) inter-hemispheric inhibition rebalancing. **Success criterion**: rank-order of clinical recovery (Combined > Sequential > TMS ≈ SCS) emerges from mechanistic dynamics, not parameter tuning.

**Aim 3 — Predict individual variability and dose-response.**
Sample 100 stroke configurations (lesion size × location × CST integrity) and 5 NIBS dosing regimes per configuration. Build a digital-twin response prediction map. **Success criterion**: the simulated response distribution reproduces the >60% I² heterogeneity seen in NIBS meta-analyses (Hsu 2012, Kang 2016).

## 3. Methodology

### 3.1 Multi-scale architecture (3 coupled levels)

| Level | Tool | Neurons | Synapses | Dynamics |
|-------|------|---------|----------|----------|
| Whole brain (mesoscale) | TVB | 76 ROIs | weighted connectome | Hopf oscillators |
| M1 microcircuit | NEST 3.x | 160,000 AdEx | 3×10⁸ | Conductance-based, STDP on L5→L5 |
| Spinal CPG | NEST 3.x | 160,000 AdEx | 5×10⁸ | Conductance-based, STDP on Ia/V2a→MN |
| Motoneuron pools | NEURON+CoreNEURON | ~6,000 multicompartment | — | HH on H100 GPU |

**TVB ↔ NEST coupling**: regional firing rate from TVB → Poisson generator input to corresponding NEST microcircuit; spike output from NEST → drives TVB Hopf oscillator. Coupling timestep: 1 ms. Implementation: tvb-multiscale (Lionheart 2022).

### 3.2 Stroke implementation (mechanistic)

- **Ischemic core**: defined as the lesion ROI (e.g., precentral gyrus L); neurons set to firing = 0 (synaptic input cannot overcome membrane silencing).
- **Penumbra zone** (concentric shell around core): AdEx neurons with V_rest shifted +5 mV, leak conductance ×2 (ATP-dependent K⁺ channel partial failure; Hofmeijer & van Putten 2012).
- **Excitotoxicity**: glutamate AMPA conductance ×1.5 in peri-infarct neurons, reverting to baseline over weeks 0–4 (Lo 2003).
- **Spreading depolarization (optional)**: stochastic depolarization waves originating in core, propagating into penumbra (Hartings 2017). Costed in Aim 3.
- **Diaschisis**: callosal projections from lesioned M1 silenced → contralateral M1 loses 35% of inhibitory input, leading to hyper-excitability initially, then progressive normalization over weeks (Murase 2004).

### 3.3 NIBS implementation (biophysical, not phenomenological)

- **tDCS (anodal/cathodal M1)**: applied as constant 0.5 mV (anodal) / -0.3 mV (cathodal) sub-threshold depolarization to L5 pyramidal somata in the stimulation zone. E-field magnitude from ROAST simulation (Huang 2019) with standard 35 cm² electrode montage.
- **rTMS (10 Hz, 5 sessions/week × 12 weeks)**: bursts of ~5 ms suprathreshold depolarization (~25 mV) delivered at 10 Hz for 1500 pulses/session, targeting L5 layer in lesioned M1.
- **Epidural SCS (40 Hz tonic, weeks 0–24)**: continuous synaptic input to dorsal-column / Ia-afferent INs in segments L1–L5, modeled as 40 Hz Poisson with rate proportional to stimulation amplitude.

### 3.4 STDP (the key mechanism)

Pair-based STDP with parameters from Bi & Poo 1998 / Sjöström 2001:
- A⁺ = 0.005, A⁻ = -0.0026 (depression slightly weaker than potentiation for LTP-biased synapses)
- τ⁺ = 17 ms, τ⁻ = 34 ms
- w_max = 5×w_init, w_min = 0

STDP active on: L5 pyramidal ↔ L5 pyramidal (intra-cortical), L5 pyramidal → spinal MN (corticospinal, learning), and Ia-afferent → V2a interneurons (spinal locomotor learning).

### 3.5 Gait readout

The 6 motoneuron pool population firing rates (L/R × flexor/extensor × hip/knee/ankle) drive Hill-type muscle models (Zajac 1989); resulting joint torques applied to a sagittal-plane biomechanical model (4-link inverted pendulum). Bilateral force asymmetry index (AmpAI) and step frequency extracted exactly as in Phase 2 of preliminary work.

### 3.6 Computational pipeline

```
Snakemake pipeline (snakemake/Snakefile)
├── (1) Generate parameter sets (Latin Hypercube, 100 stroke configs)
├── (2) Run baseline (no NIBS) per config — 100 jobs × 24 weeks
├── (3) Run 4 NIBS protocols per config — 400 jobs × 24 weeks
├── (4) Validation runs (literature replication) — 60 jobs
├── (5) Sensitivity analysis (parameter perturbations) — 400 jobs
├── (6) Analysis: BOLD FC, gait metrics, STDP weight trajectories
└── (7) Figure generation, statistical comparison
```

Total: ~960 production runs + ~400 sensitivity runs.

### 3.7 Validation strategy

| Result | Reference data | Acceptance criterion |
|--------|----------------|----------------------|
| Baseline CST firing rate | Stinear 2007 MEP/FA | within 20% |
| Stroke FC reduction (M1_L↔SMN) | Carter 2010 fMRI | within 30% |
| Gait amplitude asymmetry, acute stroke | Olney & Richards 1996 | within 25% |
| Spontaneous recovery curve (no NIBS) | Copenhagen Stroke Study (Jorgensen 1995) | shape match within 1σ |
| TMS-induced recovery acceleration | Hsu 2012 meta-analysis | Cohen's d within 0.2 |
| Epidural SCS effect on locomotor function | Wagner 2018 | qualitative pattern match |
| Inter-subject variability (heterogeneity I²) | NIBS meta-analyses | I² in [40%, 80%] |

## 4. Innovation and significance

### 4.1 Methodological innovation

- **First multi-scale TVB + NEST + spinal-CPG implementation for any neurological disorder.** Existing TVB-multiscale work has demonstrated the framework on healthy brain only.
- **Realistic spinal CPG with V0–V3 interneuron architecture coupled to cortical microcircuit.** Previous spinal modeling work (Ausborn, Danner) stops at the spinal cord; previous cortical work (Potjans-Diesmann) stops at the cortex. This bridges them.
- **Mechanistic NIBS application via biophysical fields rather than phenomenological drives.**

### 4.2 Scientific significance

- **Resolves the heterogeneity paradox** of NIBS trials by quantitatively predicting which patient configurations respond.
- **Predicts new clinical protocols** (e.g., sequential timing, dose-response curves) that can be tested in subsequent clinical trials.
- **Validates or rejects the STDP hypothesis** for late-phase recovery — a question that has been the subject of debate for two decades.

### 4.3 Clinical translation potential

The applicant's group has [or will establish] collaboration with [clinical neurology / rehabilitation department] for:
1. Patient-specific connectome import (DTI) — already supported by TVB.
2. Lesion mask integration — straightforward extension.
3. NIBS protocol optimization for individual patients — the digital-twin from Aim 3.

### 4.4 Open science deliverables

- All code at https://github.com/max-talanov/bigMC under MIT license.
- All output data (spike rasters, weight matrices, simulated BOLD) at EBRAINS Knowledge Graph.
- NESTML model files for ischemic-penumbra AdEx neuron and STDP variants.
- Singularity containers for one-click reproduction on any HPC center.

## 5. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NEST↔TVB coupling instability at scale | Medium | High | Start small (Phase 1 already validates standalone TVB). Use tvb-multiscale's published validation cases as fall-back. |
| Spinal CPG does not produce stable gait | Medium | High | Use Danner et al. 2019 published parameter set as starting point. Pilot run in Cycle 1 explicitly validates this. |
| STDP saturates too fast (uninformative) | Medium | Medium | Tune homeostatic scaling to keep STDP in dynamic range over 24-week sim. Literature-derived target firing rates (Turrigiano 2008). |
| MN5 GPP nodes have insufficient RAM | Low | High | Each node has 256 GB; our largest microcircuit is ~120 GB. Confirmed feasible. |
| Simulation real-time-factor too slow | Medium | Medium | Use NEST's "ignore_and_fire" optimization for synaptic delivery; benchmark on 10% of full system before committing. |
| Allocation insufficient | Low | Medium | Pilot Cycle 1 will produce realistic estimates; renegotiate or reduce sensitivity-analysis scope if needed. |

## 6. Timeline

```
Month  1  2  3  4  5  6  7  8  9 10 11 12
Aim1  ███████████░░
Aim2          ░░████████████░
Aim3                      ░██████████
Sens.                            ██████░░
Publ.                                █████
```

| Quarter | Deliverable |
|---------|-------------|
| Q1 | Pilot runs; healthy & stroke baselines validated |
| Q2 | NIBS protocols implemented; first protocol comparison results |
| Q3 | Individual variability sweep; digital-twin prototype |
| Q4 | Manuscript submission; data deposited at EBRAINS; code release |

## 7. Publications expected

1. **Main paper** (target: *eLife*, *Nature Communications*): "A multi-scale mechanistic simulation of stroke rehabilitation under combined transcranial and spinal neurostimulation."
2. **Methods paper** (target: *PLoS Computational Biology*, *Frontiers in Neuroinformatics*): "BigStroke-MN5: an open framework for multi-scale neurorehabilitation simulation."
3. **Clinical perspective** (target: *Stroke*, *Neurorehabilitation and Neural Repair*): "Predicting NIBS responder phenotypes via in-silico digital twins."
4. **Conference**: BSC Doctoral Symposium, INCF Neuroinformatics Assembly, OHBM 2027.

## 8. Team and expertise

**PI** (Max Talanov): >X years in computational neuroscience; lead author on Phase 1–4D preliminary work; familiar with TVB, NEST, and HPC environments.

**Co-PIs**: [To be filled — ideally including (a) a NEST/HPC expert with prior MN5 or equivalent experience, (b) a clinical stroke researcher to advise on validation parameters, (c) a spinal-cord modeller (perhaps from the Danner/Rybak group via collaboration).]

**Junior researchers**: [N] PhD students / postdocs will be trained on the platform; each will lead a sub-component (e.g., spinal CPG implementation, NIBS field modeling, statistical analysis).

## 9. References (selected)

1. Carter et al. (2010). Resting interhemispheric fMRI connectivity predicts performance after stroke. *Annals of Neurology* 67: 365–375.
2. Cramer (2008). Repairing the human brain after stroke. *Annals of Neurology* 63: 272–287.
3. Danner et al. (2019). Spinal V3 interneurons and left–right coordination in mammalian locomotion. *Frontiers in Cellular Neuroscience* 13: 516.
4. Gewaltig & Diesmann (2007). NEST (NEural Simulation Tool). *Scholarpedia* 2: 1430.
5. Hofmeijer & van Putten (2012). Ischemic cerebral damage: an appraisal of synaptic failure. *Stroke* 43: 607–615.
6. Hsu et al. (2012). Effects of repetitive transcranial magnetic stimulation on motor functions in patients with stroke. *Stroke* 43: 1849–1857.
7. Hummel & Cohen (2005). Drivers of brain plasticity. *Current Opinion in Neurology* 18: 667–674.
8. Jorgensen et al. (1995). Outcome and time course of recovery in stroke. The Copenhagen Stroke Study. *Archives of Physical Medicine and Rehabilitation* 76: 406–412.
9. Kiehn (2016). Decoding the organization of spinal circuits that control locomotion. *Nature Reviews Neuroscience* 17: 224–238.
10. Krakauer (2010). Motor learning and consolidation: the case of visuomotor rotation. *Advances in Experimental Medicine and Biology* 629: 405–421.
11. Lionheart et al. (2022). tvb-multiscale: a framework for multi-scale modeling. *Frontiers in Neuroinformatics* 16: 911223.
12. Lo et al. (2003). Mechanisms, challenges and opportunities in stroke. *Nature Reviews Neuroscience* 4: 399–415.
13. Murase et al. (2004). Influence of interhemispheric interactions on motor function in chronic stroke. *Annals of Neurology* 55: 400–409.
14. Olney & Richards (1996). Hemiparetic gait following stroke. *Gait & Posture* 4: 136–148.
15. Potjans & Diesmann (2014). The cell-type specific cortical microcircuit. *Cerebral Cortex* 24: 785–806.
16. Rowald et al. (2022). Activity-dependent spinal cord neuromodulation rapidly restores trunk and leg motor functions. *Nature Medicine* 28: 260–271.
17. Sanz Leon et al. (2013). The Virtual Brain: a simulator of primate brain network dynamics. *Frontiers in Neuroinformatics* 7: 10.
18. Schirner et al. (2018). Inferring multi-scale neural mechanisms with brain network modelling. *eLife* 7: e28927.
19. Sjöström et al. (2001). Rate, timing, and cooperativity jointly determine cortical synaptic plasticity. *Neuron* 32: 1149–1164.
20. Stinear (2007). Functional potential in chronic stroke patients depends on corticospinal tract integrity. *Brain* 130: 170–180.
21. Turrigiano (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell* 135: 422–435.
22. Wagner et al. (2018). Targeted neurotechnology restores walking in humans with spinal cord injury. *Nature* 563: 65–71.
