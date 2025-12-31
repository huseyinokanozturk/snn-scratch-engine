"""
Network Class: Modular Layer-Based Spiking Neural Network

This module implements a multi-layer SNN architecture using the Layer class.
The network consists of:
- Input Layer: Receives sensor data
- Hidden Layer: Recurrent processing with internal connections
- Output Layer: Motor commands

Synapse matrices:
- w_in_hidden: Feedforward from input to hidden
- w_hidden_hidden: Recurrent within hidden layer
- w_hidden_out: Feedforward from hidden to output

Learning:
- R-STDP on w_hidden_out synapses

Dependencies: numpy, Layer class
"""
import numpy as np
import os
from src.layer import Layer


class Network:
    """
    Modular Layer-Based Spiking Neural Network.
    
    Architecture:
    [Input Layer] -> [Hidden Layer] -> [Output Layer]
                       ^       |
                       +-------+ (recurrent)
    """

    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        dt: float = 1.0
    ):
        """
        Initialize the layered network.

        Args:
            num_inputs: Number of input neurons (sensors).
            num_hidden: Number of hidden neurons.
            num_outputs: Number of output neurons (motors).
            dt: Time step in milliseconds.
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.dt = dt
        self.N = num_inputs + num_hidden + num_outputs  # Total neurons

        # ==================== LAYERS ====================
        # Input Layer: High recovery (sensors don't tire easily)
        self.in_layer = Layer(
            num_neurons=num_inputs,
            dt=dt,
            name="input",
            tau=15.0,
            base_threshold=1.0,
            energy_max=100.0,
            energy_cost=1.0,       # Low cost
            recovery_rate=0.5,     # HIGH recovery
            fatigue_factor=0.05,
            refractory_ms=1.0,
        )

        # Hidden Layer: Moderate fatigue for processing
        self.hidden_layer = Layer(
            num_neurons=num_hidden,
            dt=dt,
            name="hidden",
            tau=20.0,
            base_threshold=1.2,
            energy_max=100.0,
            energy_cost=2.0,       # Moderate cost
            recovery_rate=0.1,     # Moderate recovery
            fatigue_factor=0.1,
            refractory_ms=2.0,
        )

        # Output Layer: High energy cost (motor neurons work hard)
        self.out_layer = Layer(
            num_neurons=num_outputs,
            dt=dt,
            name="output",
            tau=25.0,
            base_threshold=1.5,
            energy_max=100.0,
            energy_cost=3.0,       # HIGH cost
            recovery_rate=0.15,
            fatigue_factor=0.15,
            refractory_ms=2.0,
        )

        # ==================== SYNAPSE MATRICES ====================
        # w_in_hidden: Input -> Hidden (feedforward)
        self.w_in_hidden = np.random.uniform(-1.0, 3.0, (num_hidden, num_inputs))
        
        # w_hidden_hidden: Hidden -> Hidden (recurrent)
        self.w_hidden_hidden = np.random.uniform(-2.0, 2.0, (num_hidden, num_hidden))
        np.fill_diagonal(self.w_hidden_hidden, 0)  # No self-connections
        
        # w_hidden_out: Hidden -> Output (feedforward, LEARNABLE)
        self.w_hidden_out = np.random.uniform(-1.0, 3.0, (num_outputs, num_hidden))

        # ==================== LEARNING (R-STDP for w_hidden_out) ====================
        self.eligibility = np.zeros((num_outputs, num_hidden))
        # BOOSTED: tau_eligibility increased from 150 to 300 for ~5-10 second credit window
        self.tau_eligibility = 300.0
        self.eligibility_decay = np.exp(-dt / self.tau_eligibility)
        # BOOSTED: learning rate increased for stronger reward effect
        self.learning_rate = 0.025

        # ==================== NEUROMODULATORS ====================
        self.dopamine = 0.0
        self.acetylcholine = 0.0
        self.serotonin = 0.0
        
        # Exploration noise multiplier (increases after failure)
        self.exploration_noise = 1.0
        self.cumulative_reward = 0.0
        self.reward_decay = 0.995
        self.panic_level = 0.0

        # Base threshold for compatibility
        self.base_threshold = 1.2

        # Apply initial wiring
        self._apply_braitenberg_wiring()

        print(f"Network initialized: {num_inputs} inputs, {num_hidden} hidden, {num_outputs} outputs")

    def step(self, external_inputs: np.ndarray, reward: float = 0.0, learning: bool = True) -> np.ndarray:
        """
        Simulate one timestep of the network.

        Signal cascade:
        1. External inputs -> Input Layer
        2. Input spikes * w_in_hidden -> Hidden Layer
        3. Hidden spikes * w_hidden_hidden -> Hidden Layer (recurrent)
        4. Hidden spikes * w_hidden_out -> Output Layer

        Args:
            external_inputs: Sensor currents. Shape: (num_inputs,).
            reward: Reward signal for R-STDP.
            learning: Enable synaptic plasticity.

        Returns:
            all_spikes: Concatenated spikes [input, hidden, output].
        """
        # Update cumulative reward (clamped for display)
        self.cumulative_reward = self.cumulative_reward * self.reward_decay + reward
        self.cumulative_reward = np.clip(self.cumulative_reward, -50.0, 50.0)
        self.dopamine = reward

        # ==================== DISTRIBUTE NEUROMODULATORS ====================
        self.in_layer.set_neuromodulators(self.acetylcholine, self.dopamine, self.serotonin)
        self.hidden_layer.set_neuromodulators(self.acetylcholine, self.dopamine, self.serotonin)
        self.out_layer.set_neuromodulators(self.acetylcholine, self.dopamine, self.serotonin)

        # ==================== STEP 1: INPUT LAYER ====================
        # Input layer receives external sensor currents directly
        in_spikes = self.in_layer.step(external_inputs)

        # ==================== STEP 2: HIDDEN LAYER ====================
        # Hidden receives: feedforward from input + recurrent from previous hidden spikes
        hidden_ff = self.w_in_hidden @ in_spikes.astype(np.float32)
        hidden_rec = self.w_hidden_hidden @ self.hidden_layer.spikes.astype(np.float32)
        hidden_current = hidden_ff + hidden_rec
        
        hidden_spikes = self.hidden_layer.step(hidden_current)

        # ==================== STEP 3: OUTPUT LAYER ====================
        out_current = self.w_hidden_out @ hidden_spikes.astype(np.float32)
        out_spikes = self.out_layer.step(out_current)

        # ==================== LEARNING (R-STDP on w_hidden_out) ====================
        if learning:
            # Eligibility trace: outer product of post (output) traces and pre (hidden) traces
            self.eligibility *= self.eligibility_decay
            
            if np.any(out_spikes):
                # Update eligibility when output fires
                self.eligibility += np.outer(out_spikes, self.hidden_layer.traces)

            # Apply reward modulation
            if reward != 0.0:
                self.w_hidden_out += self.learning_rate * self.eligibility * reward
                np.clip(self.w_hidden_out, -50, 50, out=self.w_hidden_out)

        # ==================== CONCATENATE ALL SPIKES ====================
        all_spikes = np.concatenate([in_spikes, hidden_spikes, out_spikes])
        
        return all_spikes

    def _apply_braitenberg_wiring(self):
        """
        Apply hard-wired Braitenberg connections for wall avoidance.
        Sets up contralateral sensor-to-motor wiring through the hidden layer.
        """
        mid_sensor = self.num_inputs // 2

        # Strengthen connections from left sensors to hidden neurons that connect to right motor
        # and vice versa for contralateral avoidance behavior
        
        # Left sensors (0 to mid) -> increase weights to hidden neurons
        # that will excite right motor
        for i in range(mid_sensor):
            # Strengthen first half of hidden neurons
            hidden_subset = self.num_hidden // 2
            self.w_in_hidden[:hidden_subset, i] += 2.0
        
        # Right sensors (mid to end) -> strengthen second half of hidden
        for i in range(mid_sensor, self.num_inputs):
            hidden_subset = self.num_hidden // 2
            self.w_in_hidden[hidden_subset:, i] += 2.0

        # First half of hidden -> right motor (index 1)
        if self.num_outputs >= 2:
            hidden_half = self.num_hidden // 2
            self.w_hidden_out[1, :hidden_half] += 3.0  # Right motor
            self.w_hidden_out[0, hidden_half:] += 3.0  # Left motor
            
            # Center sensors inhibit both motors (panic brake)
            center_sensors = [mid_sensor - 1, mid_sensor] if mid_sensor > 0 else [0]
            for cs in center_sensors:
                if cs < self.num_inputs:
                    # These sensors inhibit forward motion through hidden
                    self.w_in_hidden[:, cs] -= 2.0

            # Motor mutual inhibition (prevent both full on)
            self.w_hidden_out[0, :] -= 0.5
            self.w_hidden_out[1, :] -= 0.5

    def apply_reactive_inhibition(self, sensor_data: np.ndarray, threshold: float = 0.7):
        """Apply reactive inhibition when near walls."""
        max_sensor = np.max(sensor_data) if sensor_data.size > 0 else 0.0

        if max_sensor > threshold:
            self.panic_level = (max_sensor - threshold) / (1.0 - threshold)
            self.serotonin = min(self.serotonin + self.panic_level * 2.0, 3.0)
        else:
            self.panic_level *= 0.9
            self.serotonin *= 0.95

        return self.panic_level

    def apply_epoch_failure(self, penalty: float = -50.0):
        """
        Apply penalty when epoch times out without reaching goal.
        
        Effects:
        1. Large negative reward to dopamine system
        2. Serotonin spike (stress response)
        3. Weaken recently active synapses (eligibility * penalty)
        4. Boost exploration noise for next epoch
        
        Args:
            penalty: Negative reward value (default -50.0).
        
        Returns:
            None
        """
        # 1. Apply negative dopamine (reward signal)
        self.dopamine = penalty
        self.cumulative_reward = np.clip(self.cumulative_reward + penalty, -50, 50)
        
        # 2. Major serotonin spike (stress)
        self.serotonin = 5.0  # Maximum stress
        
        # 3. Weaken synapses that were recently active (led to failure)
        # This creates a "forgetting" effect for failed paths
        self.w_hidden_out += self.learning_rate * 0.5 * self.eligibility * penalty
        np.clip(self.w_hidden_out, -50, 50, out=self.w_hidden_out)
        
        # 4. Boost exploration noise for next epoch
        self.exploration_noise = 3.0  # 3x normal noise
        
        # 5. Partially decay eligibility (fresh start but not complete reset)
        self.eligibility *= 0.3
        
        print(f"Epoch failed! Penalty={penalty}, Exploration noise boosted to {self.exploration_noise}")

    def decay_exploration_noise(self):
        """Gradually reduce exploration noise back to normal."""
        self.exploration_noise = max(1.0, self.exploration_noise * 0.995)

    def get_top_synapses(self, neuron_idx: int, top_k: int = 5):
        """
        Get top K strongest outgoing connections from a neuron.
        Works for visualization - maps global index to layer synapses.
        """
        # Determine which layer the neuron is in
        if neuron_idx < self.num_inputs:
            # Input neuron -> connections to hidden
            weights = self.w_in_hidden[:, neuron_idx]
            offset = self.num_inputs  # Hidden neurons start after inputs
        elif neuron_idx < self.num_inputs + self.num_hidden:
            # Hidden neuron
            local_idx = neuron_idx - self.num_inputs
            # Combine recurrent + output connections
            rec_weights = self.w_hidden_hidden[:, local_idx]
            out_weights = self.w_hidden_out[:, local_idx]
            
            # Return strongest connections
            results = []
            # Recurrent connections
            top_rec = np.argsort(np.abs(rec_weights))[-2:][::-1]
            for idx in top_rec:
                if rec_weights[idx] != 0:
                    results.append((int(idx + self.num_inputs), float(rec_weights[idx])))
            # Output connections
            top_out = np.argsort(np.abs(out_weights))[-2:][::-1]
            for idx in top_out:
                if out_weights[idx] != 0:
                    results.append((int(idx + self.num_inputs + self.num_hidden), float(out_weights[idx])))
            return results[:top_k]
        else:
            # Output neuron - no outgoing connections
            return []

        top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]
        return [(int(idx + offset), float(weights[idx])) for idx in top_indices if weights[idx] != 0]

    def reset_state(self):
        """Reset all layer states without changing weights."""
        self.in_layer.reset()
        self.hidden_layer.reset()
        self.out_layer.reset()
        self.panic_level = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0
        self.eligibility.fill(0)

    # ==================== PERSISTENT MEMORY ====================

    def save_weights(self, filename: str = "brain_weights.npz"):
        """Save all synapse weights to brain_weights directory."""
        from src.utils.file_manager import get_brain_weights_path
        
        weights_path = get_brain_weights_path() / filename
        np.savez(
            weights_path,
            w_in_hidden=self.w_in_hidden,
            w_hidden_hidden=self.w_hidden_hidden,
            w_hidden_out=self.w_hidden_out,
            eligibility=self.eligibility,
            cumulative_reward=self.cumulative_reward,
            num_inputs=self.num_inputs,
            num_hidden=self.num_hidden,
            num_outputs=self.num_outputs,
        )
        print(f"Weights saved to: {weights_path}")
        return True

    def load_weights(self, filename: str = "brain_weights.npz"):
        """Load synapse weights from brain_weights directory."""
        from src.utils.file_manager import get_brain_weights_path
        
        weights_path = get_brain_weights_path() / filename
        
        if not weights_path.exists():
            print(f"File not found: {weights_path}")
            return False

        try:
            data = np.load(weights_path)
            
            # Validate dimensions
            if (data['num_inputs'] != self.num_inputs or 
                data['num_hidden'] != self.num_hidden or
                data['num_outputs'] != self.num_outputs):
                print("Dimension mismatch! Cannot load weights.")
                return False
            
            self.w_in_hidden = data['w_in_hidden']
            self.w_hidden_hidden = data['w_hidden_hidden']
            self.w_hidden_out = data['w_hidden_out']
            self.eligibility = data['eligibility']
            self.cumulative_reward = float(data['cumulative_reward'])
            
            print(f"Weights loaded from: {weights_path}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
