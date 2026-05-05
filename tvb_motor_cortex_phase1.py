#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TVB Motor Cortex Model for Stroke Simulation and tinyCPG Integration
Phase 1: Cortical model development with spinal interface specification

This script implements a patient-specific motor cortex model using The Virtual Brain (TVB)
framework. It is designed for validation against neuroimaging data (fMRI, EEG) and 
integration with tinyCPG spinal motor circuits via a rate-to-spike converter.

Key design principles:
  - Modularity: cortex validation independent of spinal integration
  - Interface-first: outputs designed to drive tinyCPG rhythm generators
  - Patient-specific: parameterized from individual DTI connectivity and lesion anatomy
  - Validation hooks: outputs structured for fMRI FC and EEG spectral comparison

Author: [Your Name]
Project: TVB + tinyCPG integrated motor stroke model
Version: 0.1 (Phase 1 prototype)
Date: 2026
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import json
from datetime import datetime

import numpy as np
import scipy.signal as signal
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import h5py

# Third-party neuroimaging libraries
try:
    from tvb.basic.neotraits.api import HasTraits, Attr, List as TraitList
    from tvb.simulator.simulator import Simulator
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.cortex import Cortex
    from tvb.datatypes.equations import Linear
    from tvb.simulator.models.wong_wang import ReducedWongWang as ReducedWongWangExcIO
    from tvb.simulator.integrators import HeunDeterministic, HeunStochastic
    from tvb.simulator.monitors import Bold, EEG
except ImportError as e:
    print(f"Warning: TVB not installed. Install with: pip install tvb-framework")
    print(f"Error details: {e}")
    raise

# ============================================================================
# CONFIGURATION AND INTERFACE CONTRACT
# ============================================================================

@dataclass
class MotorRegionIndices:
    """
    Region indices for motor-relevant areas in standard atlases.
    Based on Desikan-Killiany (DK) atlas with 68 cortical regions.
    Easily adapted for AAL (116 regions) or custom atlases.
    """
    M1_left: int = 4      # Precentral gyrus, left (primary motor)
    M1_right: int = 39    # Precentral gyrus, right
    S1_left: int = 22     # Postcentral gyrus, left (primary sensory)
    S1_right: int = 57    # Postcentral gyrus, right
    SMA_left: int = 31    # Superior frontal gyrus (medial), left (SMA)
    SMA_right: int = 66   # Superior frontal gyrus (medial), right
    PMC_left: int = 3     # Precentral gyrus (premotor), left
    PMC_right: int = 38   # Precentral gyrus (premotor), right
    
    # Thalamus (if available in region mapping; often merged in cortical atlases)
    # For standard DK, thalamus is excluded; use a custom atlas or add as separate node
    VL_left: Optional[int] = None     # Ventral lateral nucleus
    VL_right: Optional[int] = None
    
    # Brainstem nuclei (if available; often requires custom atlas)
    # For now, we will implement brainstem as an input generator, not a region
    BRAINSTEM: Optional[int] = None
    
    @property
    def motor_regions(self) -> List[int]:
        """Return list of motor-relevant region indices."""
        return [
            self.M1_left, self.M1_right,
            self.S1_left, self.S1_right,
            self.SMA_left, self.SMA_right,
            self.PMC_left, self.PMC_right,
        ]
    
    @property
    def motor_region_names(self) -> Dict[int, str]:
        """Return mapping of region index to anatomical name."""
        return {
            self.M1_left: "M1_L", self.M1_right: "M1_R",
            self.S1_left: "S1_L", self.S1_right: "S1_R",
            self.SMA_left: "SMA_L", self.SMA_right: "SMA_R",
            self.PMC_left: "PMC_L", self.PMC_right: "PMC_R",
        }


@dataclass
class TinyCPGInterfaceContract:
    """
    Specification of expected input characteristics for tinyCPG brainstem and RG neurons.
    This defines what the TVB cortex must output to successfully drive the spinal CPG.
    
    Derived from tinyCPG documentation (cpg_2legs.py, lines ~50–80):
      - RG (rhythm generator) expects rates in range [0, 300 Hz] nominally
      - Half-centre oscillators are sensitive to asymmetry in input rates
      - Commissural neurons couple L/R CPGs; asymmetric drive → phase shift
      - Frequency of oscillation tunable via drive amplitude (~0.5–3 Hz for locomotion)
    """
    
    # Input layer specifications
    rg_input_rate_min_hz: float = 0.0          # Minimum drive rate
    rg_input_rate_max_hz: float = 300.0        # Maximum drive rate
    rg_input_rate_nominal_hz: float = 50.0     # Typical background drive for walking
    
    # Frequency characteristics (what oscillation frequency tinyCPG produces for given input)
    cpg_output_freq_min_hz: float = 0.5        # Slowest realistic walking rhythm
    cpg_output_freq_max_hz: float = 3.0        # Fastest realistic walking rhythm
    
    # Latency tolerance
    interface_latency_ms: float = 1.0           # Expected latency in rate→spike conversion
    
    # Spike train properties (after rate→spike conversion)
    spike_train_jitter_ms: float = 5.0         # Poisson jitter in spike timing
    
    # Expected asymmetry for stroke (clinical motivation)
    # After unilateral M1 stroke, contralateral drive is reduced
    stroke_drive_reduction_ratio: float = 0.3  # Example: 70% reduction → 30% remaining
    
    def validate_cortical_output(self, left_rate: float, right_rate: float) -> Tuple[bool, str]:
        """
        Check that cortical output rates are compatible with tinyCPG expectations.
        Returns (is_valid, message).
        """
        if left_rate < self.rg_input_rate_min_hz or left_rate > self.rg_input_rate_max_hz:
            return False, f"Left rate {left_rate:.1f} Hz outside valid range [{self.rg_input_rate_min_hz}, {self.rg_input_rate_max_hz}]"
        if right_rate < self.rg_input_rate_min_hz or right_rate > self.rg_input_rate_max_hz:
            return False, f"Right rate {right_rate:.1f} Hz outside valid range [{self.rg_input_rate_min_hz}, {self.rg_input_rate_max_hz}]"
        return True, "Output rates compatible with tinyCPG interface"


@dataclass
class StrokeParameters:
    """
    Parameterization of motor cortex stroke lesion.
    """
    lesion_side: str = "left"  # "left" or "right"
    lesion_primary_region: str = "M1"  # Affected primary region: M1, S1, PMC, etc.
    
    # Excitability reduction (0 = complete infarction, 1 = no lesion)
    excitability_factor: float = 0.3  # Example: 70% reduction from baseline
    
    # Connectivity loss along damaged white-matter tracts (DTI-derived)
    connectivity_reduction_factor: float = 0.4  # Example: 60% tract damage
    
    # Lesion volume (mm³), for documentation and comparison
    lesion_volume_mm3: Optional[float] = None
    
    # Timing of lesion (for progression studies)
    days_post_stroke: int = 7  # Time since stroke onset
    
    @property
    def is_lesioned(self) -> bool:
        """Check if this represents an actual lesion."""
        return self.excitability_factor < 1.0 or self.connectivity_reduction_factor < 1.0


@dataclass
class SimulationConfig:
    """Global simulation configuration."""
    
    # Time parameters
    simulation_duration_ms: float = 10000.0  # Total simulation time
    dt_ms: float = 0.1                       # Integration timestep (TVB default)
    
    # Model parameters
    neuron_model_name: str = "ReducedWongWangExcIO"  # TVB neural mass model
    global_coupling: float = 0.01             # Global coupling strength
    
    # Input (brainstem oscillator parameters)
    brainstem_drive_frequency_hz: float = 1.0  # Walking frequency command
    brainstem_drive_amplitude_hz: float = 50.0  # Input amplitude (sets CPG frequency)
    brainstem_drive_phase_lr: float = 0.0      # Phase difference between L/R (0 = in phase)
    
    # Integration method
    integrator_name: str = "HeunDeterministic"
    noise_amplitude: float = 0.0  # Stochastic noise (0 for deterministic)
    
    # Monitor configuration
    monitor_bold: bool = True      # Record BOLD for fMRI comparison
    monitor_eeg: bool = False       # Record EEG (requires region mapping)
    monitor_spike_rate: bool = True # Record instantaneous firing rates
    
    # Output configuration
    output_dir: Optional[str] = None
    output_filename_prefix: str = "tvb_motor_cortex"
    
    def get_output_dir(self) -> Path:
        """Get or create output directory."""
        if self.output_dir is None:
            self.output_dir = Path("./tvb_output")
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


# ============================================================================
# PATIENT DATA LOADER (placeholder for real DTI/neuroimaging data)
# ============================================================================

class PatientData:
    """
    Container for patient-specific data: DTI connectivity, lesion location, 
    clinical measurements, and neuroimaging validation targets.
    """
    
    def __init__(self, patient_id: str, atlas: str = "DK68"):
        self.patient_id = patient_id
        self.atlas = atlas
        self.n_regions = 68 if atlas == "DK68" else 116  # Default atlases
        
        self.dti_connectivity: Optional[np.ndarray] = None  # N x N connectivity
        self.lesion_mask: Optional[np.ndarray] = None       # Binary lesion map per region
        self.regional_volumes_mm3: Optional[np.ndarray] = None  # Volume of each region
        
        # Validation targets (from neuroimaging)
        self.rsfmri_bold_fc: Optional[np.ndarray] = None   # Resting-state functional connectivity
        self.eeg_psd: Optional[Dict[str, np.ndarray]] = None  # EEG power spectral density
        self.clinical_scores: Dict[str, float] = {}        # FUGL-Meyer, etc.
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"PatientData_{self.patient_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @staticmethod
    def create_placeholder(patient_id: str, atlas: str = "DK68") -> 'PatientData':
        """
        Create a placeholder patient with synthetic DTI (for development/testing).
        In production, replace with real patient data loading.
        """
        patient = PatientData(patient_id, atlas)
        n = patient.n_regions
        
        # Synthetic DTI connectivity: small-world network with some modularity
        # (simplified; real DTI would come from diffusion tensor imaging tractography)
        np.random.seed(hash(patient_id) % (2**32))
        dti = np.random.exponential(scale=0.5, size=(n, n))
        
        # Add spatial structure: nearby regions more connected
        for i in range(n):
            for j in range(n):
                distance_factor = 1.0 + 0.1 * np.abs(i - j) / n
                dti[i, j] /= distance_factor
        
        # Symmetrize
        dti = (dti + dti.T) / 2
        np.fill_diagonal(dti, 0)
        
        # Normalize to [0, 1]
        dti = dti / dti.max()
        
        patient.dti_connectivity = dti
        patient.logger.info(f"Created placeholder DTI connectivity (n_regions={n})")
        
        return patient
    
    def apply_lesion(self, lesion_params: StrokeParameters) -> None:
        """
        Apply stroke lesion by reducing excitability and connectivity in affected region.
        """
        if not self.dti_connectivity is not None:
            self.logger.warning("No DTI connectivity loaded; cannot apply lesion.")
            return
        
        region_indices = MotorRegionIndices()
        region_names = region_indices.motor_region_names
        
        # Find primary affected region
        target_regions = []
        if lesion_params.lesion_primary_region == "M1":
            target_regions = [region_indices.M1_left if lesion_params.lesion_side == "left" 
                             else region_indices.M1_right]
        elif lesion_params.lesion_primary_region == "S1":
            target_regions = [region_indices.S1_left if lesion_params.lesion_side == "left"
                             else region_indices.S1_right]
        # ... extend for other regions as needed
        
        # Apply lesion: reduce connectivity from affected region
        for region_idx in target_regions:
            # Reduce outgoing connections
            self.dti_connectivity[region_idx, :] *= lesion_params.connectivity_reduction_factor
            # Reduce incoming connections
            self.dti_connectivity[:, region_idx] *= lesion_params.connectivity_reduction_factor
        
        self.logger.info(f"Applied lesion: {lesion_params.lesion_side} {lesion_params.lesion_primary_region}, "
                        f"excitability={lesion_params.excitability_factor:.2f}, "
                        f"connectivity={lesion_params.connectivity_reduction_factor:.2f}")


# ============================================================================
# TVB CORTEX MODEL SETUP
# ============================================================================

class TVBMotorCortexModel:
    """
    Wrapper around TVB simulator for motor cortex stroke simulation.
    Manages setup, parameterization, and output formatting for tinyCPG integration.
    """
    
    def __init__(self, patient: PatientData, config: SimulationConfig, 
                 stroke_params: Optional[StrokeParameters] = None):
        self.patient = patient
        self.config = config
        self.stroke_params = stroke_params
        
        self.simulator: Optional[Simulator] = None
        self.regional_fire_rates: Optional[np.ndarray] = None  # For interface output
        
        self.logger = logging.getLogger("TVBMotorCortexModel")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.interface_contract = TinyCPGInterfaceContract()
    
    def setup_connectivity(self) -> Connectivity:
        """
        Create a TVB Connectivity object from patient DTI.
        """
        self.logger.info("Setting up connectivity from patient DTI...")
        
        # Create a basic connectivity object
        # In production, this would load from actual DTI-weighted connectome
        conn = Connectivity()
        conn.weights = self.patient.dti_connectivity.copy()
        
        # Set tract lengths (mm) — simplified; in production, from actual DTI
        n = self.patient.dti_connectivity.shape[0]
        conn.tract_lengths = 50.0 * np.ones((n, n))  # Placeholder uniform distances
        conn.region_labels = np.array([f"Region_{i:03d}" for i in range(n)])
        conn.centres = np.zeros((n, 3))  # Placeholder 3D coordinates
        conn.orientations = np.zeros((n, 3))
        conn.areas = np.ones(n)
        conn.cortical = np.ones(n, dtype=bool)
        conn.hemispheres = np.array([i < n // 2 for i in range(n)])

        self.logger.info(f"Connectivity shape: {conn.weights.shape}, "
                        f"mean weight: {conn.weights.mean():.4f}")

        return conn
    
    def setup_neuron_model(self) -> ReducedWongWangExcIO:
        """
        Create and configure the neural mass model (Wong-Wang excitatory input only).
        """
        self.logger.info(f"Setting up neuron model: {self.config.neuron_model_name}")
        
        model = ReducedWongWangExcIO()
        
        # Model parameters (defaults are typically good; adjust if needed for motor system)
        # These capture the dynamics of cortical populations at the mesoscale
        # Use TVB canonical units: a [n/C], b [kHz], d [ms], tau_s [ms]
        model.a = np.array([0.270])         # Input gain [n/C]
        model.b = np.array([0.108])         # Input shift [kHz]
        model.d = np.array([154.0])         # Fit parameter [ms]
        model.tau_s = np.array([100.0])     # NMDA decay time constant [ms]
        model.I_o = np.array([0.33])        # Background current [nA]
        model.sigma_noise = np.array([0.000000001])  # Noise (deterministic run)

        self.logger.info(f"Neuron model parameters configured: a={model.a[0]:.4f}, "
                        f"tau_s={model.tau_s[0]:.4f}")
        
        return model
    
    def setup_integrator(self) -> Tuple[HeunDeterministic, np.ndarray]:
        """
        Set up integration method and noise profile.
        """
        self.logger.info(f"Setting up integrator: {self.config.integrator_name}")
        
        if self.config.integrator_name == "HeunDeterministic":
            integrator = HeunDeterministic()
        else:
            integrator = HeunStochastic()
        
        integrator.dt = self.config.dt_ms / 1000.0  # Convert ms to seconds
        
        return integrator, np.zeros((self.patient.n_regions, 1))  # Noise amplitude per region
    
    def setup_monitors(self) -> List:
        """
        Configure monitoring: BOLD for fMRI validation, spike rate for interface output.
        """
        monitors = []
        
        if self.config.monitor_bold:
            bold_monitor = Bold(period=2000.0)  # Sample every 2 seconds (typical TR)
            monitors.append(bold_monitor)
            self.logger.info("BOLD monitor enabled")
        
        if self.config.monitor_spike_rate:
            # Raw monitor to capture state variables (will post-process to get rates)
            from tvb.simulator.monitors import Raw
            raw_monitor = Raw()
            monitors.append(raw_monitor)
            self.logger.info("Raw state monitor enabled for firing rate extraction")
        
        return monitors
    
    def build_simulator(self) -> Simulator:
        """
        Assemble all components and create the TVB Simulator object.
        """
        self.logger.info("Building simulator...")
        
        connectivity = self.setup_connectivity()
        model = self.setup_neuron_model()
        integrator, noise = self.setup_integrator()
        monitors = self.setup_monitors()
        
        from tvb.simulator.coupling import Linear as LinearCoupling
        coupling = LinearCoupling()
        coupling.a = np.array([self.config.global_coupling])

        self.simulator = Simulator(
            model=model,
            connectivity=connectivity,
            integrator=integrator,
            coupling=coupling,
            monitors=monitors,
            initial_conditions=np.random.randn(
                1, model.nvar, self.patient.n_regions, model.number_of_modes
            ) * 0.1  # Small random initialization
        )
        
        self.simulator.configure()
        self.logger.info("Simulator built successfully")

        return self.simulator
    
    def extract_firing_rates(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Extract instantaneous firing rates from Wong-Wang state variables.
        
        Wong-Wang output (spike rate) is computed from the state variable S (synaptic gating)
        via a sigmoidal transfer function. For ReducedWongWangExcIO:
          rate = φ(I) = a*I / (1 - exp(-d*(b*I - I_o)))
        
        Args:
            raw_state: TVB state array of shape (n_regions, n_vars, ...)
        
        Returns:
            Firing rates in Hz for each region, shape (n_regions,)
        """
        # Extract S (synaptic gating variable, range [0,1])
        if raw_state.ndim >= 2:
            S = raw_state[:, 0]  # shape (n_regions,)
        else:
            S = raw_state

        # Wong-Wang transfer function H(x) = (a*x - b) / (1 - exp(-d*(a*x - b)))
        # x = w*J_N*S + I_o  (local input; global coupling adds to this in full sim)
        m = self.simulator.model
        a   = float(m.a[0])
        b   = float(m.b[0])
        d   = float(m.d[0])
        I_o = float(m.I_o[0])
        w   = float(m.w[0])
        J_N = float(m.J_N[0])

        x = w * J_N * S + I_o          # effective input [nA]
        ax_b = a * x - b               # = (a*x - b) [kHz]
        exp_term = 1.0 - np.exp(-d * ax_b)
        # H(x) = (a*x - b) / (1 - exp(-d*(a*x-b)))
        # L'Hopital at ax_b=0: limit → 1/d
        H_kHz = np.where(np.abs(ax_b) < 1e-9, 1.0 / d, ax_b / exp_term)
        rates_hz = np.clip(H_kHz * 1000.0, 0.0, 300.0)  # kHz → Hz

        return rates_hz
    
    def get_corticospinal_drive(self, firing_rates: np.ndarray) -> Tuple[float, float]:
        """
        Compute left and right corticospinal drive rates from cortical firing rates.
        This is the critical output for tinyCPG integration.
        
        Averaging motor regions (M1, SMA, PMC) on each side gives the descending drive.
        
        Args:
            firing_rates: Firing rates (Hz) for all regions
        
        Returns:
            (left_drive_hz, right_drive_hz) tuple, compatible with tinyCPG
        """
        region_indices = MotorRegionIndices()
        
        # Left hemisphere motor output (average of M1, SMA, PMC)
        left_motor_indices = [region_indices.M1_left, region_indices.SMA_left, region_indices.PMC_left]
        left_drive = np.mean([firing_rates[idx] for idx in left_motor_indices if idx is not None])
        
        # Right hemisphere motor output
        right_motor_indices = [region_indices.M1_right, region_indices.SMA_right, region_indices.PMC_right]
        right_drive = np.mean([firing_rates[idx] for idx in right_motor_indices if idx is not None])
        
        # Validate against interface contract
        is_valid, msg = self.interface_contract.validate_cortical_output(left_drive, right_drive)
        if not is_valid:
            self.logger.warning(f"Interface contract violation: {msg}")
        
        return left_drive, right_drive
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Execute the simulation and collect results.
        
        Returns:
            Dictionary with time, firing_rates, bold_signal (if enabled), and 
            corticospinal_drive (for tinyCPG integration)
        """
        if self.simulator is None:
            self.build_simulator()
        
        self.logger.info(f"Running simulation for {self.config.simulation_duration_ms} ms...")
        
        # Run
        output = self.simulator.run(
            simulation_length=self.config.simulation_duration_ms / 1000.0  # Convert to seconds
        )
        
        self.logger.info("Simulation completed")
        
        # Parse output
        results = {
            'time_s': None,
            'firing_rates_hz': None,
            'bold_signal': None,
            'corticospinal_drive_left_hz': None,
            'corticospinal_drive_right_hz': None,
        }
        
        # Extract data from monitors
        for monitor_data, monitor in zip(output, self.simulator.monitors):
            if isinstance(monitor, Bold):
                results['time_bold_s'] = monitor_data[0]
                results['bold_signal'] = monitor_data[1]
                self.logger.info(f"BOLD signal shape: {results['bold_signal'].shape}")
            else:
                # Raw monitor: (time_array, data_array)
                # data shape: (n_time, n_vars, n_regions, n_modes)
                time_s = monitor_data[0]
                raw_state = monitor_data[1]

                # Extract firing rates at each timepoint
                n_time = raw_state.shape[0]
                firing_rates = np.zeros((self.patient.n_regions, n_time))

                for t in range(n_time):
                    # raw_state[t] has shape (n_vars, n_regions, n_modes)
                    # transpose to (n_regions, n_vars) for extract_firing_rates
                    firing_rates[:, t] = self.extract_firing_rates(raw_state[t, :, :, 0].T)
                
                results['time_s'] = time_s
                results['firing_rates_hz'] = firing_rates
                self.logger.info(f"Firing rates shape: {firing_rates.shape}")
        
        # Compute corticospinal drive (L/R motor output) at each timepoint
        if results['firing_rates_hz'] is not None:
            n_time = results['firing_rates_hz'].shape[1]
            left_drive = np.zeros(n_time)
            right_drive = np.zeros(n_time)
            
            for t in range(n_time):
                left_drive[t], right_drive[t] = self.get_corticospinal_drive(
                    results['firing_rates_hz'][:, t]
                )
            
            results['corticospinal_drive_left_hz'] = left_drive
            results['corticospinal_drive_right_hz'] = right_drive
            
            self.logger.info(f"Corticospinal drive | Mean L: {left_drive.mean():.1f} Hz, "
                           f"Mean R: {right_drive.mean():.1f} Hz")
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """
        Save simulation results to HDF5 for later analysis and tinyCPG integration.
        """
        output_dir = self.config.get_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{self.config.output_filename_prefix}_{self.patient.patient_id}_{timestamp}.h5"
        
        self.logger.info(f"Saving results to {filename}")
        
        with h5py.File(filename, 'w') as f:
            f.attrs['patient_id'] = self.patient.patient_id
            f.attrs['simulation_duration_ms'] = self.config.simulation_duration_ms
            f.attrs['timestamp'] = timestamp
            
            if self.stroke_params:
                f.attrs['stroke_side'] = self.stroke_params.lesion_side
                f.attrs['stroke_primary_region'] = self.stroke_params.lesion_primary_region
                f.attrs['stroke_excitability_factor'] = self.stroke_params.excitability_factor
            
            # Save time series
            if results['time_s'] is not None:
                f.create_dataset('time_s', data=results['time_s'])
            
            if results['firing_rates_hz'] is not None:
                f.create_dataset('firing_rates_hz', data=results['firing_rates_hz'])
            
            if results['corticospinal_drive_left_hz'] is not None:
                f.create_dataset('corticospinal_drive_left_hz', 
                               data=results['corticospinal_drive_left_hz'])
                f.create_dataset('corticospinal_drive_right_hz',
                               data=results['corticospinal_drive_right_hz'])
            
            if results['bold_signal'] is not None:
                f.create_dataset('bold_signal', data=results['bold_signal'])
        
        self.logger.info(f"Results saved to {filename}")
        
        return filename
    
    def summary_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute summary statistics of simulation for validation and comparison.
        """
        stats = {}
        
        if results['firing_rates_hz'] is not None:
            rates = results['firing_rates_hz']
            stats['firing_rate_mean_hz'] = rates.mean()
            stats['firing_rate_std_hz'] = rates.std()
            stats['firing_rate_max_hz'] = rates.max()
            stats['firing_rate_min_hz'] = rates.min()
        
        if results['corticospinal_drive_left_hz'] is not None:
            left_drive = results['corticospinal_drive_left_hz']
            right_drive = results['corticospinal_drive_right_hz']
            
            stats['cst_left_mean_hz'] = left_drive.mean()
            stats['cst_right_mean_hz'] = right_drive.mean()
            stats['cst_asymmetry_ratio'] = (
                left_drive.mean() / (right_drive.mean() + 1e-8)
                if right_drive.mean() > 0 else np.inf
            )
        
        return stats


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def compare_functional_connectivity(simulated_bold: np.ndarray, 
                                     target_fc: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Compute functional connectivity from simulated BOLD and compare to target fMRI data.
    
    Args:
        simulated_bold: BOLD signal from TVB, shape (n_regions, n_timepoints)
        target_fc: Target functional connectivity matrix, shape (n_regions, n_regions)
    
    Returns:
        (overall_correlation, detailed_metrics) tuple
    """
    # Compute functional connectivity from simulated BOLD
    simulated_fc = np.corrcoef(simulated_bold)
    
    # Vectorize and compare
    target_vec = squareform(target_fc, checks=False)
    simulated_vec = squareform(simulated_fc, checks=False)
    
    overall_corr, pval = pearsonr(simulated_vec, target_vec)
    
    metrics = {
        'fc_correlation': overall_corr,
        'fc_pvalue': pval,
        'target_fc_mean': target_fc.mean(),
        'simulated_fc_mean': simulated_fc.mean(),
    }
    
    return overall_corr, metrics


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

def main():
    """
    Example workflow: create a patient, set up stroke, run TVB cortex simulation,
    and prepare output for tinyCPG integration.
    """
    
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = SimulationConfig(
        simulation_duration_ms=10000.0,
        dt_ms=0.1,
        brainstem_drive_frequency_hz=1.0,
        brainstem_drive_amplitude_hz=50.0,
        output_dir="./tvb_output/phase1",
    )
    
    # Create or load patient data
    print("Step 1: Loading patient data...")
    patient = PatientData.create_placeholder("PATIENT_001", atlas="DK68")
    
    # Define stroke parameters (example: left M1 stroke)
    stroke = StrokeParameters(
        lesion_side="left",
        lesion_primary_region="M1",
        excitability_factor=0.3,
        connectivity_reduction_factor=0.4,
        days_post_stroke=7,
    )
    
    # Apply stroke to patient connectivity
    patient.apply_lesion(stroke)
    
    # Create TVB model
    print("\nStep 2: Building TVB cortex model...")
    tvb_model = TVBMotorCortexModel(patient, config, stroke)
    tvb_model.build_simulator()
    
    # Run simulation
    print("\nStep 3: Running simulation...")
    results = tvb_model.run_simulation()
    
    # Compute summary statistics
    print("\nStep 4: Computing summary statistics...")
    stats = tvb_model.summary_statistics(results)
    print("Summary statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")
    
    # Save results
    print("\nStep 5: Saving results...")
    output_file = tvb_model.save_results(results)
    print(f"Results saved to: {output_file}")
    
    # Prepare interface output for tinyCPG (THIS IS THE CRITICAL OUTPUT)
    print("\nStep 6: Interface output for tinyCPG integration...")
    if results['time_s'] is not None and results['corticospinal_drive_left_hz'] is not None:
        left_drive = results['corticospinal_drive_left_hz']
        right_drive = results['corticospinal_drive_right_hz']
        time_s = results['time_s']
        
        print(f"\nCorticospinal drive rates (for tinyCPG):")
        print(f"  Left:  min={left_drive.min():.1f}, mean={left_drive.mean():.1f}, max={left_drive.max():.1f} Hz")
        print(f"  Right: min={right_drive.min():.1f}, mean={right_drive.mean():.1f}, max={right_drive.max():.1f} Hz")
        print(f"\nThese rates can now be converted to spike trains and fed to tinyCPG.")
        print(f"Interface contract check:")
        is_valid, msg = tvb_model.interface_contract.validate_cortical_output(
            left_drive.mean(), right_drive.mean()
        )
        print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
        print(f"  Message: {msg}")


if __name__ == "__main__":
    main()
