"""
Synapse Model with Reward-Modulated STDP (R-STDP)
=================================================

This module implements a dynamic synapse with Reward-Modulated Spike-Timing Dependent Plasticity (R-STDP).
Unlike standard STDP where weights change immediately upon spiking events, R-STDP uses an "eligibility trace"
to bridge the gap between spike timing (pre/post activity) and a delayed global reward signal.

Mathematical Foundation (R-STDP):
---------------------------------
The weight update mechanism involves two distinct processes:
    
    1. **Eligibility Trace (E)**:
       Tracks potential weight changes based on spike timing (similar to STDP).
       - Decays over time: E(t) = E(t-1) * exp(-dt / τ_eligibility)
       - Updates on spikes:
            * On Post-spike (LTP): E += Trace_pre
            * On Pre-spike (LTD):  E -= Trace_post

    2. **Weight Update (W)**:
       The actual weight modification happens only when a reward is present.
       
       ΔW = η * E * R

    Where:
        - E: Eligibility trace.
        - τ_eligibility: Time constant for the eligibility trace.
        - R: Reward signal (scalar) at time t.
        - η (eta): Learning rate.

Operational Dynamics:
---------------------
- **Signal Transmission**: When `spike_pre` is True, current flows to the post-neuron.
- **Trace Dynamics**: The eligibility trace accumulates "credit" or "blame" from pre/post spike coincidences
  but decays over time (`tau_eligibility`).
- **Learning**: Synaptic weights only change when `reward` is non-zero, scaling the pending eligibility trace.
- **Bounds**: Weights are clipped to [0, w_max].

Usage:
------
Call `step(spike_pre, spike_post, reward)` at every simulation step.
"""
import numpy as np
from .neuron import LIFNeuron

class Synapse:
    """
    A synapse is a connection between two neurons.
    """
    def __init__(self, weight: float, pre_neuron: LIFNeuron, post_neuron: LIFNeuron, w_max: float, lr: float, tau_eligibility: float,
    dt: float):
        """
        Initialize the synapse.

        Args:
            weight (float): The weight of the synapse.
            pre_neuron (LIFNeuron): The presynaptic neuron.
            post_neuron (LIFNeuron): The postsynaptic neuron.
            w_max (float): The maximum weight of the synapse.
            lr (float): The learning rate.
            tau_eligibility (float): The time constant of the eligibility trace.
            dt (float): The time step.
        """
        self.weight = weight # The weight of the synapse
        self.pre_neuron = pre_neuron # The presynaptic neuron
        self.post_neuron = post_neuron # The postsynaptic neuron
        self.w_max = w_max # The maximum weight of the synapse
        self.lr = lr # The learning rate
        self.tau_eligibility = tau_eligibility # The time constant of the eligibility trace (50-100ms)
        self.dt = dt # The time step
        self.eligibility_trace = 0 # The eligibility trace
        self.eligibility_decay = np.exp(-self.dt / self.tau_eligibility) # The eligibility decay

    def step(self, spike_pre: bool, spike_post: bool, reward: float = 0.0):
        """
        Update the synapse.

        Args:
            spike_pre (bool): The presynaptic spike.
            spike_post (bool): The postsynaptic spike.
            reward (float): The reward.
        """
        current_to_transfer = 0.0
        if spike_pre:
            current_to_transfer = self.weight

        self.eligibility_trace *= self.eligibility_decay

        if spike_post:
            self.eligibility_trace += self.pre_neuron.trace

        if spike_pre:
            self.eligibility_trace -= self.post_neuron.trace

        if reward != 0.0:
            self.weight += self.lr * self.eligibility_trace * reward
            self.weight = np.clip(self.weight, 0, self.w_max)

        return current_to_transfer