"""
Synapse Model with Spike-Timing Dependent Plasticity (STDP)
=========================================================

This module implements a dynamic synapse that connects two neurons (pre-synaptic and post-synaptic).
It facilitates charge transfer and adjusts its synaptic weight based on the relative timing of spikes 
from the connected neurons, a learning rule known as Spike-Timing Dependent Plasticity (STDP).

Mathematical Foundation (STDP):
-------------------------------
The weight update follows the standard STDP rule, which relies on "traces" (decaying memory of past spikes)
stored in the neurons.

    1. **Long-Term Potentiation (LTP)**:
       Occurs when the pre-synaptic neuron fires *before* the post-synaptic neuron (causal relationship).
       When the post-neuron spikes, the weight increases proportional to the pre-neuron's trace.
       
       Δw += η * x_pre    (if spike_post)

    2. **Long-Term Depression (LTD)**:
       Occurs when the post-synaptic neuron fires *before* the pre-synaptic neuron (acausal relationship).
       When the pre-neuron spikes, the weight decreases proportional to the post-neuron's trace.

       Δw -= η * x_post   (if spike_pre)

    Where:
        - Δw: Change in synaptic weight.
        - η (eta): Learning rate.
        - x_pre, x_post: Synaptic traces of the pre- and post-synaptic neurons, respectively.

Operational Dynamics:
---------------------
- **Signal Transmission**: When the pre-synaptic neuron spikes (`spike_pre` is True), the synapse transmits 
  current equal to its current `weight` to the post-synaptic neuron.
- **Weight Clipping**: After updates, weights are strictly bounded to the range [0, w_max] to prevent 
  runaway growth or negative conductance (excitatory synapses only).
"""
import numpy as np
from .neuron import LIFNeuron

class Synapse:
    """
    A synapse is a connection between two neurons.
    """
    def __init__(self, weight: float, pre_neuron: LIFNeuron, post_neuron: LIFNeuron, w_max: float, lr: float):
        """
        Initialize the synapse.

        Args:
            weight (float): The weight of the synapse.
            pre_neuron (LIFNeuron): The presynaptic neuron.
            post_neuron (LIFNeuron): The postsynaptic neuron.
            w_max (float): The maximum weight of the synapse.
            lr (float): The learning rate.
        """
        self.weight = weight # The weight of the synapse
        self.pre_neuron = pre_neuron # The presynaptic neuron
        self.post_neuron = post_neuron # The postsynaptic neuron
        self.w_max = w_max # The maximum weight of the synapse
        self.lr = lr # The learning rate

    def step(self, spike_pre: bool, spike_post: bool):
        """
        Update the synapse.

        Args:
            spike_pre (bool): The presynaptic spike.
            spike_post (bool): The postsynaptic spike.
        """
        current_to_transfer = 0.0
        if spike_pre:
            current_to_transfer = self.weight

        if spike_post:
            self.weight += self.lr * self.pre_neuron.trace

        if spike_pre:
            self.weight -= self.lr * self.post_neuron.trace

        if spike_pre or spike_post:
            self.weight = np.clip(self.weight, 0, self.w_max)

        return current_to_transfer