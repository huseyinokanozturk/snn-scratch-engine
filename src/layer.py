"""
Layer Class: Modular LIF Neuron Layer

This module implements a single layer of Leaky Integrate-and-Fire neurons
that can be composed to build multi-layer spiking neural networks.

Each layer handles:
- Membrane potential dynamics (LIF)
- Spike generation with adaptive thresholds
- Metabolic energy tracking
- Eligibility traces for R-STDP learning
- Neuromodulatory sensitivity

Usage:
    layer = Layer(num_neurons=50, dt=1.0, name="hidden")
    spikes = layer.step(input_current)
"""
import numpy as np


class Layer:
    """
    A single layer of LIF neurons with metabolic dynamics.
    
    Features:
    - Vectorized LIF computation
    - Per-neuron metabolic energy
    - Adaptive thresholds based on fatigue and neuromodulation
    - Eligibility traces for credit assignment
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,
        name: str = "layer",
        tau: float = 20.0,
        base_threshold: float = 1.2,
        energy_max: float = 100.0,
        energy_cost: float = 2.0,
        recovery_rate: float = 0.1,
        fatigue_factor: float = 0.1,
        refractory_ms: float = 2.0,
    ):
        """
        Initialize a neuron layer.

        Args:
            num_neurons: Number of neurons in this layer.
            dt: Time step in milliseconds.
            name: Layer name for debugging.
            tau: Membrane time constant (ms).
            base_threshold: Base firing threshold.
            energy_max: Maximum metabolic energy.
            energy_cost: Energy consumed per spike.
            recovery_rate: Energy recovery per timestep.
            fatigue_factor: How much energy deficit affects threshold.
            refractory_ms: Refractory period in milliseconds.
        """
        self.name = name
        self.N = num_neurons
        self.dt = dt

        # LIF parameters
        self.tau = tau
        self.decay = np.exp(-dt / tau)
        self.base_threshold = base_threshold
        self.v_reset = 0.0

        # Refractory period
        self.refractory_steps = int(refractory_ms / dt)

        # Metabolic parameters
        self.energy_max = energy_max
        self.energy_cost = energy_cost
        self.recovery_rate = recovery_rate
        self.fatigue_factor = fatigue_factor

        # State variables
        self.voltages = np.zeros(self.N)
        self.energies = np.full(self.N, energy_max)
        self.thresholds = np.full(self.N, base_threshold)
        self.spikes = np.zeros(self.N, dtype=bool)
        self.timers = np.zeros(self.N, dtype=int)

        # Eligibility traces (for R-STDP learning)
        self.tau_trace = 20.0
        self.decay_trace = np.exp(-dt / self.tau_trace)
        self.traces = np.zeros(self.N)

        # Neuromodulator sensitivity (can be set externally)
        self.acetylcholine = 0.0
        self.dopamine = 0.0
        self.serotonin = 0.0

    def step(self, input_current: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep of the layer.

        Args:
            input_current: External/synaptic current to neurons. Shape: (N,).

        Returns:
            spikes: Boolean array of which neurons fired. Shape: (N,).
        """
        # ========== ADAPTIVE THRESHOLDS ==========
        energy_deficit = self.energy_max - self.energies
        self.thresholds = np.maximum(
            self.base_threshold
            + (energy_deficit * self.fatigue_factor)
            - (self.acetylcholine * 0.5)
            - (self.dopamine * 0.8)
            + (self.serotonin * 0.5),
            0.1  # Minimum threshold
        )

        # ========== LIF DYNAMICS ==========
        is_refractory = self.timers > 0
        is_active = ~is_refractory

        # Membrane potential update for active neurons
        self.voltages[is_active] = (
            self.voltages[is_active] * self.decay
            + input_current[is_active] * (1.0 - self.decay)
        )

        # Refractory neurons stay at reset
        self.voltages[is_refractory] = self.v_reset
        self.timers[is_refractory] -= 1

        # ========== SPIKE GENERATION ==========
        self.spikes = (self.voltages >= self.thresholds) & is_active

        # Post-spike handling
        spiked_any = np.any(self.spikes)
        if spiked_any:
            self.voltages[self.spikes] = self.v_reset
            self.timers[self.spikes] = self.refractory_steps
            self.energies[self.spikes] -= self.energy_cost

        # ========== METABOLIC RECOVERY ==========
        self.energies += self.recovery_rate
        np.clip(self.energies, 0, self.energy_max, out=self.energies)

        # ========== TRACE UPDATE (for R-STDP) ==========
        self.traces *= self.decay_trace
        if spiked_any:
            self.traces[self.spikes] += 1.0

        return self.spikes

    def set_neuromodulators(self, ach: float = 0.0, dopamine: float = 0.0, serotonin: float = 0.0):
        """Set neuromodulator levels for this layer."""
        self.acetylcholine = ach
        self.dopamine = dopamine
        self.serotonin = serotonin

    def reset(self):
        """Reset layer state without changing parameters."""
        self.voltages = np.zeros(self.N)
        self.energies = np.full(self.N, self.energy_max)
        self.spikes = np.zeros(self.N, dtype=bool)
        self.timers = np.zeros(self.N, dtype=int)
        self.traces = np.zeros(self.N)

    def get_spike_count(self) -> int:
        """Return number of spikes in current timestep."""
        return int(np.sum(self.spikes))

    def __repr__(self):
        return f"Layer('{self.name}', N={self.N}, spikes={self.get_spike_count()})"
