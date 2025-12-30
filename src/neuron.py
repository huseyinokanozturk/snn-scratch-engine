"""
Leaky Integrate-and-Fire (LIF) Neuron Model
===========================================

This module implements a standard Leaky Integrate-and-Fire (LIF) neuron, a fundamental building block 
in Spiking Neural Networks (SNNs). It simulates the membrane potential dynamics of a biological neuron 
using a discredited mathematical model.

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

Firing & Dynamics:
------------------
1. **Integration**: At each step, the neuron integrates input current into its membrane potential using the 
   formula above.
2. **Firing**: If the membrane potential exceeds the 'threshold', the neuron fires (returns 1).
3. **Reset**: Immediately after firing, the potential is reset to 'reset_potential'.
4. **Refractory Period**: The neuron enters a refractory state for a duration of 'refractory_time' 
   (converted to 'ref_steps'), during which it cannot integrate inputs or fire again.

"""

import numpy as np


class LIFNeuron:
    """
    A simple neuron model.
    """
    def __init__(self, tau: float, dt: float, threshold: float, rest_potential: float, 
    reset_potential: float, refractory_time: float):
        '''
        Initialize the neuron with the given parameters.
        Args:
            tau (float): Time constant of the neuron in ms. If tau is large, the neuron will have a long memory.
            dt (float): Time step in ms. Sensitivity of the simulation.
            threshold (float): Threshold potential of the neuron in mV. If the neuron's potential exceeds this value, it will fire.
            rest_potential (float): Resting potential of the neuron in mV. 
            reset_potential (float): Reset potential of the neuron in mV. When the neuron fires, its potential is reset to this value.
            refractory_time (float): Refractory period of the neuron in ms. The neuron cannot fire during this period.
        '''
        self.tau = tau
        self.dt = dt
        self.threshold = threshold
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

    def update(self, input_current: float):
        """
        Update the neuron's state.
        Args:
            input_current (float): Input current in V. The current that flows into the neuron.
        """
        self.trace *= self.trace_decay # Decay the trace at each time step

        if self.ref_count > 0: # If the neuron is in the refractory period (tired) it cannot fire 
            self.current_voltage = self.reset_potential
            self.ref_count -= 1
            return 0
        else:
            self.current_voltage = (self.current_voltage * self.decay_factor) + (input_current * (1-self.decay_factor)) # Update the voltage (1-self.decay_factor simulates the Resistance)
            if self.current_voltage >= self.threshold: # If the voltage exceeds the threshold, the neuron fires
                self.ref_count = self.ref_steps
                self.current_voltage = self.reset_potential
                self.trace += 1
                return 1
            else:
                return 0
        