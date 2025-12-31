"""
Network Class: Vectorized Spiking Neural Network (SNN) Simulation

This module implements a biologically inspired Spiking Neural Network using a
vectorized approach for efficient computation. The Network class encapsulates
the state and dynamics of a population of Leaky Integrate-and-Fire (LIF) neurons,
incorporating advanced features such as metabolic energy constraints, adaptive
thresholds, and neuromodulatory influences.

Key Features:
- **Leaky Integrate-and-Fire Dynamics**: Simulates membrane potential decay and
  spike generation mechanisms.
- **Refractory Periods**: Enforces a temporary inactivity period post-spike to
  mimic biological constraints.
- **Metabolic Energy Model**: Tracks neuronal energy levels, applying costs for
  spiking and allowing for recovery, influencing neuronal fatigue.
- **Synaptic Plasticity & Traces**: Maintains eligibility traces and synaptic
  weights to support learning rules (e.g., STDP).
- **Neuromodulation**: Includes placeholders for global neuromodulators like
  Dopamine, Acetylcholine, and Serotonin to modulate network behavior.

Usage:
    Initialize the network with the desired number of neurons, inputs, outputs,
    and time step. Use the `step()` method (not shown in initialization) to
    advance the simulation.

    net = Network(num_neurons=100, num_inputs=10, num_outputs=5, dt=0.1)

Dependencies:
    numpy: Used for all vector and matrix operations.
"""
import numpy as np

class Network:
    """
    RHEO SNN: Vectorized Spiking Neural Network Engine

    Features:
    - Vectorized operations for efficient simulation (NumPy)
    - Metabolic energy model
    - R-STDP (Reward-Modulated Plasticity)
    - Neuromodulation (Dopamine, Acetylcholine, Serotonin)
    """
    def __init__(self, num_neurons: int, num_inputs: int, num_outputs: int, dt: float = 0.1):
        """
        Initialize the network.

        Args:
            num_neurons (int): Number of neurons in the network.
            num_inputs (int): Number of input neurons.
            num_outputs (int): Number of output neurons.
            dt (float): Time step in milliseconds.
        """
        self.N = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dt = dt
        
        # Membrane potential time constant (ms) and decay factor
        self.tau = 20.0
        self.tau_trace = 20.0
        self.decay = np.exp(-dt/self.tau)
        self.decay_trace = np.exp(-dt/self.tau_trace)

        # Spike generation parameters
        self.base_threshold = 1.2
        self.v_reset = 0.0 

        # Refractory period (time steps to stay inactive after spiking)
        self.refractory_steps = int(2.0/self.dt)

        # Metabolic energy model parameters
        self.energy_max = 100.0
        self.energy_cost = 2.0
        self.recovery_rate = 0.1
        self.fatigue_factor = 0.1

        # Neuron state variables initialization
        self.voltages = np.zeros(self.N)
        self.energies = np.full(self.N, self.energy_max)
        self.thresholds = np.full(self.N, self.base_threshold)
        self.spikes = np.zeros(self.N, dtype=bool)
        self.timers = np.zeros(self.N, dtype=int)
        self.traces = np.zeros(self.N)

        # Synaptic weight matrix initialization (random small weights)
        self.weights = np.random.uniform(-2, 5.0, (self.N, self.N))

        np.fill_diagonal(self.weights, 0)

        # Lateral Inhibition
        motor_left_idx = self.N - 2
        motor_right_idx = self.N - 1

        self.weights[motor_left_idx, motor_right_idx] = -5.0
        self.weights[motor_right_idx, motor_left_idx] = -5.0

        # Plasticity traces (Eligibility traces for R-STDP)
        self.eligibility = np.zeros((self.N, self.N))
        self.tau_eligibility = 100.0
        self.eligibility_decay = np.exp(-dt/self.tau_eligibility)

        # Neuromodulator levels (Global context signals)
        self.dopamine = 0.0
        self.acetylcholine = 0.0
        self.serotonin = 0.0

        print("Network initialized with", self.N, "neurons (vectorized). ")

    def step(self, external_inputs: np.ndarray, reward: float = 0.0, learning: bool=True):
        """
        Simulate one time step (dt) of the network.

        Args:
            external_inputs (np.ndarray): External inputs to the network (Sensors). Shape: (num_inputs,).
            reward (float): Reward signal (dopamine).
            learning (bool): Whether to enable learning (default: True).
        
        Returns:
            spikes: Vector of spiking neurons (outputs). Shape: (num_outputs,).
        """
        
        # Input aggregation
        # Calculate synaptic current from other neurons (Recurrent)
        recurrent_inputs = self.weights @ self.spikes.astype(float)

        # Combine recurrent and external currents
        total_current = recurrent_inputs.copy()
        if len(external_inputs) > 0:
            # Add external stimulus to input neurons (first num_inputs neurons)
            total_current[:self.num_inputs] += external_inputs


        # Homeostasis & Neuromodulation
        # Adjust firing thresholds dynamically based on:
        # 1. Energy deficit (Fatigue): Less energy -> Higher threshold (harder to fire)
        # 2. Neuromodulators (ACh, DA, 5-HT): Global chemical signals shifting excitability
        energy_deficit = self.energy_max - self.energies 
        
        adaptive_thresholds = (self.base_threshold
                                + (energy_deficit * self.fatigue_factor)
                                + (self.acetylcholine * 0.5)
                                - (self.dopamine * 0.8)
                                + (self.serotonin * 0.5))

        # Ensure thresholds don't drop below a minimum safety margin
        adaptive_thresholds = np.maximum(adaptive_thresholds, 0.1)
        self.thresholds = adaptive_thresholds

        # LIF Neuron Update
        # Separate active neurons from those in refractory period
        is_refractory = self.timers > 0 
        is_active = ~is_refractory

        # Update Membrane Potential (Voltage) for active neurons
        # Equation: v(t+1) = v(t) * decay + I(t) * (1 - decay)
        # This models the leaky integration of incoming current
        self.voltages[is_active] = (self.voltages[is_active] * self.decay
                                    + total_current[is_active] * (1-self.decay))

        # Refractory dynamics: Hold potential at reset value and decrement timer
        self.voltages[is_refractory] = self.v_reset
        self.timers[is_refractory] -= 1


        # Spiking mechanism
        # Fire if voltage exceeds dynamic threshold AND neuron is not refractory
        self.spikes = (self.voltages >= self.thresholds) & is_active

        # Handle post-spike events
        if np.any(self.spikes):
            # Reset voltage to resting potential
            self.voltages[self.spikes] = self.v_reset
            # Enter refractory period
            self.timers[self.spikes] = self.refractory_steps
            # Consume metabolic energy (ATP cost of firing)
            self.energies[self.spikes] -= self.energy_cost

        # Metabolic recovery (Mitochondrial respiration)
        self.energies += self.recovery_rate
        self.energies = np.clip(self.energies, 0, self.energy_max) 

        # Learning (Synaptic Plasticity)
        if learning:
            # 1. Update synaptic traces (pre-synaptic activity memory)
            self.traces *= self.decay_trace  # Decay trace
            self.traces[self.spikes] += 1.0  # Increment trace on spike

            # 2. Update eligibility traces (coincidence of pre-post activity)
            self.eligibility *= self.eligibility_decay
            
            # Hebbian term: Pre (Trace) * Post (Spike)
            coincidence = np.outer(self.spikes, self.traces)
            self.eligibility += coincidence

            # 3. Apply reward modulation (R-STDP)
            self.dopamine = reward

            if reward != 0:
                # Weights change proportional to: Eligibility * Reward
                d_weights = 0.005 * self.eligibility * reward

                self.weights += d_weights

                # Bound weights and prevent self-connections
                self.weights = np.clip(self.weights, 0, 1)
                np.fill_diagonal(self.weights, 0)

        return self.spikes         




