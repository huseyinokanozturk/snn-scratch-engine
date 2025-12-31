"""
Leaky Integrate-and-Fire (LIF) Neuron Model
===========================================

This module implements a Leaky Integrate-and-Fire (LIF) neuron, a fundamental building block 
in Spiking Neural Networks (SNNs). It simulates the membrane potential dynamics of a biological neuron 
using a discretized mathematical model, extended with metabolic energy constraints and homeostatic adaptation.

Mathematical Foundation:
------------------------
The membrane potential V(t) is modeled using a differential equation that describes a leaky integrator:

    τ * dV/dt = -(V - V_rest) + R * I(t)

Where:
    - τ (tau): Membrane time constant (determining the decay speed).
    - V: Current membrane potential.
    - V_rest: Resting potential.
    - R: Membrane resistance.
    - I(t): Input current at time t.

Discrete Time Implementation:
-----------------------------
To simulate this in code, the continuous equation is discretized using a time step 'dt'. 
The exact integration over one time step leads to an exponential decay update rule:

    V[t+1] = V[t] * decay_factor + Input * (1 - decay_factor)

    Where:
        decay_factor = exp(-dt / tau)

    - The term (V[t] * decay_factor) represents the "leak" or natural decay of the potential towards zero 
      (or reference) in the absence of input.
    - The term (Input * (1 - decay_factor)) represents the contribution of the incoming current accumulated 
      over the time step 'dt'. The factor (1 - decay_factor) effectively acts as a gain, ensuring that 
      for a constant input I, the voltage stabilizes at I (assuming R=1 for unit consistency in this implementation).

Firing, Dynamics & Plasticity:
------------------------------
1. **Integration**: At each step, the neuron integrates input current into its membrane potential.
2. **Firing**: If the membrane potential exceeds the current 'threshold', the neuron fires (returns 1).
3. **Reset**: Immediately after firing, the potential is reset to 'reset_potential'.
4. **Refractory Period**: The neuron enters a refractory state, ignoring inputs for 'refractory_time'.

Metabolic Energy & Homeostatic Control:
---------------------------------------
This implementation incorporates biological metabolic constraints:
1. **Energy Dynamics (ATP)**:
   - The neuron maintains an energy level (initially 'energy_max').
   - **Firing Cost**: Firing consumes 'energy_cost'.
   - **Recovery**: Energy recovers by 'recovery_rate' at each step (clamped to max).
2. **Adaptive Threshold (Fatigue)**:
   - The firing threshold adapts dynamically based on available energy (Homeostasis).
   - **Formula**: Threshold_current = Base_Threshold + (Energy_Deficit * Fatigue_Factor)
   - Where Energy_Deficit = Energy_Max - Energy_Current.
   - As energy depletes, the threshold rises, making it harder to fire.

Synaptic Traces (STDP):
-----------------------
The neuron maintains a 'trace' to facilitate spike-timing-dependent plasticity (STDP):
- **Decay**: The trace decays exponentially at every step (tau_trace).
- **Update**: Upon firing, the trace is incremented, marking recent activity.

"""

import numpy as np


class LIFNeuron:
    """
    A neuron model.
    features:
        - Leaky Integrate-and-Fire (LIF) Neuron Model
        - Reward-Dependent Spike-Timing-Dependent Plasticity (RSTDP)
        - Metabolic Energy Consumption (ATP)
        - Homeostatic Threshold Adaptation (When energy is low, the threshold is raised)
    """
    def __init__(self, tau: float, dt: float, threshold: float, rest_potential: float, 
    reset_potential: float, refractory_time: float, energy_max: float = 100.0,
    energy_cost: float = 2.0, recovery_rate: float = 0.2, fatigue_factor: float = 0.1):
        '''
        Initialize the neuron with the given parameters.
        Args:
            tau (float): Time constant of the neuron in ms. If tau is large, the neuron will have a long memory.
            dt (float): Time step in ms. Sensitivity of the simulation.
            threshold (float): Threshold potential of the neuron in mV. If the neuron's potential exceeds this value, it will fire.
            rest_potential (float): Resting potential of the neuron in mV. 
            reset_potential (float): Reset potential of the neuron in mV. When the neuron fires, its potential is reset to this value.
            refractory_time (float): Refractory period of the neuron in ms. The neuron cannot fire during this period.
            energy_max (float): Maximum energy level of the neuron.
            energy_cost (float): Energy cost per spike.
            recovery_rate (float): Recovery rate of the neuron.
            fatigue_factor (float): Fatigue factor of the neuron.
        '''
        self.tau = tau
        self.dt = dt
        self.base_threshold = threshold
        self.current_threshold = threshold # Dynamic Threshold
        self.rest_potential = rest_potential
        self.reset_potential = reset_potential
        self.refractory_time = refractory_time
        self.decay_factor = np.exp(-self.dt/self.tau) # The number that determines how much of the voltage remains at each step
        self.ref_steps = int(self.refractory_time / self.dt) # Converts the refractory time(ms) to steps
        self.current_voltage = self.rest_potential # At the beginning the voltage is at rest potential
        self.ref_count = 0 # The number of steps the neuron has been in the refractory period

        # STDP Parameters (Spike-Timing-Dependent Plasticity)
        self.tau_trace = 20 # Time constant of the trace in ms
        self.trace_decay = np.exp(-self.dt / self.tau_trace) # The number that determines how much of the trace remains at each step
        self.trace = 0 # Trace of the neuron

        # Energy Parameters (ATP)
        self.energy = energy_max 
        self.energy_max = energy_max 
        self.energy_cost = energy_cost 
        self.recovery_rate = recovery_rate 
        self.fatigue_factor = fatigue_factor 



    def update(self, input_current: float):
        """
        Update the neuron's state one time step (dt) at a time. If there is a spike returns 1, otherwise returns 0.
        Args:
            input_current (float): Input current in mV. The current that flows into the neuron.
        """
        self.trace *= self.trace_decay # Decay the trace at each time step

        # Refractory control (is neuron is resting?)
        if self.ref_count > 0: # If the neuron is in the refractory period (tired) it cannot fire 
            self.current_voltage = self.reset_potential
            self.energy += self.recovery_rate
            self.ref_count -= 1
            return 0

        # Homeostatic threshold logic
        # For example:
            # If energy = 100, threshold = 1
            # If energy = 0, threshold = 6 (impossible to fire)
        energy_deficit = self.energy_max - self.energy
        self.current_threshold = self.base_threshold + (energy_deficit * self.fatigue_factor)
        
        # Update the voltage
        self.current_voltage = (self.current_voltage * self.decay_factor) + (input_current * (1-self.decay_factor)) # Update the voltage (1-self.decay_factor simulates the Resistance)

        # Check if the neuron fires
        if self.current_voltage >= self.current_threshold: # If the voltage exceeds the threshold, the neuron fires
            self.ref_count = self.ref_steps
            self.current_voltage = self.reset_potential
            self.trace += 1
            self.energy -= self.energy_cost
            self.energy = np.clip(self.energy, 0, self.energy_max)
            return 1
        else:
            self.energy += self.recovery_rate
            self.energy = np.clip(self.energy, 0, self.energy_max)
            return 0
        