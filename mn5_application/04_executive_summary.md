# BigStroke-MN5 — Executive Summary (1 page)

**Project title.** BigStroke-MN5: Multi-scale bio-plausible simulation of stroke rehabilitation under combined cortical (TMS) and spinal (epidural SCS) neurostimulation.

**Principal Investigator.** Max Talanov, memBrain Computational Neuroscience Group.

**Scientific area.** Computational neuroscience / multi-scale brain simulation / neurorehabilitation modeling.

**Resources requested.** ~5.2 M GPP core-hours + 20,000 ACC (H100 GPU) hours + 20 TB GPFS storage, over 12 months on MareNostrum 5.

---

**The problem.** Non-invasive brain stimulation (TMS, tDCS, epidural SCS) is widely tested in stroke rehabilitation, but clinical effect sizes remain modest (Cohen's d ≈ 0.4) and inter-subject variance is enormous (I² > 60% in meta-analyses). Existing models cannot resolve *why* — they either operate at the systems scale (whole-brain functional connectivity) or at the cellular scale (single neurons), but never both. The cellular mechanisms hypothesized to drive recovery — STDP, excitotoxicity, diaschisis — are invisible to systems-level models, while the network reorganization they cause is invisible to cellular-level models.

**The approach.** Couple three established simulation platforms across four orders of biological scale:
1. **TVB** whole-brain mesoscale (76 ROIs) — our preliminary Phase 1 already reproduces stroke-induced functional disconnection (Carter 2010).
2. **NEST** spiking microcircuits — primary motor cortex (M1) bilateral implementation of the Potjans-Diesmann column (~160,000 AdEx neurons, 3×10⁸ synapses), with stroke-zone penumbra dynamics.
3. **NEST** spinal CPG — V0–V3 interneuron architecture (Kiehn 2016) over L1–S2 segments (~160,000 neurons, 5×10⁸ synapses), driving 6 muscle motoneuron pools.
4. **NEURON+CoreNEURON** detailed multi-compartment motoneuron pools on H100 GPU partition.

Coupling via **tvb-multiscale** (Lionheart 2022, already-validated bridge). STDP active on corticospinal and intraspinal synapses. NIBS applied biophysically (E-field from ROAST montage simulation → per-neuron polarization), not phenomenologically.

**The outputs.**
- 960 production simulation runs across 4 NIBS protocols × 100 stroke configurations × 24-week rehabilitation timeline.
- A digital-twin response prediction map for individual patient variability.
- Three peer-reviewed publications (target: *eLife* / *Nature Communications*, *PLoS Comp Biol*, clinical journal).
- Open-source platform + Singularity container + EBRAINS-deposited data.

**Why MN5.** Our preliminary phenomenological work (https://github.com/max-talanov/bigMC) demonstrates the scientific direction is correct. Scaling to biologically-plausible neuron counts (~3.2×10⁵ neurons, ~8×10⁸ synapses per simulation, ~1,400 simulations) is *not feasible on a workstation or modest cluster*. MN5 GPP (256 GB / node, Sapphire Rapids 112-core, EDR-200 IB) is ideal: enough memory to host the full network on 4 nodes, enough cores for NEST's hybrid MPI+OpenMP, and H100 GPU access for the detailed motoneuron compartmental models. Equivalent European systems (LUMI, JURECA, Leonardo) were considered; MN5 is preferred due to mature NEST/NEURON software stack, EBRAINS integration, and Spanish institutional alignment.

**Risk profile.** Medium-low. Each component (TVB, NEST microcircuit, spinal CPG, STDP, NIBS) has been individually validated in the literature. The innovation — and the risk — lies in their **integration at scale**. A staged pilot (Cycle 1, ~5% of allocation) validates the scaling and integration before committing the bulk of resources.

**Significance.** This will be the first multi-scale, mechanistic, biologically-plausible simulation of stroke rehabilitation. It will resolve a 20-year debate about whether cortical or spinal mechanisms dominate late-phase recovery. It will produce clinical predictions that can be tested in subsequent trials. It will release the platform as open infrastructure for the global stroke/spinal-cord-injury community.

**Timeline (12 months).**
- Q1: Pilot + healthy/stroke baselines.
- Q2: NIBS protocols implemented and compared.
- Q3: Individual variability sweep + digital twin.
- Q4: Sensitivity analysis, manuscript submission, data and code release.

---

*This proposal builds directly on Phase 1–4D of our prior open-source work (https://github.com/max-talanov/bigMC), which has validated the macroscopic hypothesis at coarse scale. MN5 will let us test whether the hypothesis survives, and which mechanism explains the observed clinical heterogeneity, at biological scale.*
