"""
Monitor Class: Performance Tracking and Learning Curve Analysis

This module tracks simulation metrics over time for scientific analysis:
- Firing rates per layer (neural activity)
- Energy levels (metabolic efficiency)
- Reward accumulation (learning progress)
- Synaptic weight changes (plasticity)
- Success tracking (goal completion times)

Usage:
    monitor = Monitor()
    monitor.record(step, layers_dict, reward, weights)
    monitor.save_stats("results.json")
"""
import numpy as np
import json
import time
from typing import Dict, Any


class Monitor:
    """
    Performance monitor for SNN simulation.
    
    Tracks metrics efficiently for later analysis and visualization.
    Designed to have minimal impact on Turbo Mode performance.
    """

    def __init__(self, record_interval: int = 10):
        """
        Initialize the monitor.

        Args:
            record_interval: How often to record (every N steps).
        """
        self.record_interval = record_interval
        self.start_time = time.time()
        
        # Data storage (lists for efficient appending)
        self.history = {
            "steps": [],
            "timestamps": [],
            "firing_rates": [],      # {layer_name: avg_rate}
            "energy_levels": [],     # {layer_name: avg_energy}
            "rewards": [],           # Current reward
            "cumulative_rewards": [],
            "weights_mean": [],      # Mean absolute weight
            "weights_std": [],       # Weight std dev
            "panic_levels": [],
            "success_epochs": [],    # [{epoch, time}]
        }
        
        # Real-time stats (for UI display)
        self.current_firing_rates = {}
        self.current_energy_levels = {}
        self.cumulative_reward = 0.0
        self.neural_silence_detected = False
        self.last_record_step = 0
        
        # Performance counters
        self.total_wall_hits = 0
        self.total_food_collected = 0
        self.epochs_completed = 0

    def should_record(self, step: int) -> bool:
        """Check if we should record at this step."""
        return (step - self.last_record_step) >= self.record_interval

    def record(
        self,
        step: int,
        layers: Dict[str, Any],
        reward: float,
        weights: np.ndarray,
        panic_level: float = 0.0
    ):
        """
        Record metrics for the current timestep.

        Args:
            step: Current simulation step.
            layers: Dict of layer objects {"input": layer, "hidden": layer, "output": layer}.
            reward: Current reward value.
            weights: Weight matrix to track (typically w_hidden_out).
            panic_level: Current panic level.
        """
        self.last_record_step = step
        self.cumulative_reward += reward
        
        # Record step and time
        self.history["steps"].append(step)
        self.history["timestamps"].append(time.time() - self.start_time)
        self.history["rewards"].append(float(reward))
        self.history["cumulative_rewards"].append(float(self.cumulative_reward))
        self.history["panic_levels"].append(float(panic_level))
        
        # Firing rates (average spike activity per layer)
        rates = {}
        # Firing rates (average spike activity per layer)
        # Optimization: Direct comprehension
        rates = {name: float(np.mean(layer.spikes)) for name, layer in layers.items()}
        self.history["firing_rates"].append(rates)
        self.current_firing_rates = rates
        
        # Check for neural silence in hidden layer
        self.neural_silence_detected = (rates.get("hidden", 1.0) == 0.0)
        
        # Energy levels (metabolic health per layer)
        energies = {name: float(np.mean(layer.energies)) for name, layer in layers.items()}
        self.history["energy_levels"].append(energies)
        self.current_energy_levels = energies
        
        # Weight statistics (synaptic stability)
        # Optimization: np.mean(np.abs) is fast, but std is slower.
        # We don't need std every single step. Just record 0 if skipping.
        abs_weights = np.abs(weights)
        self.history["weights_mean"].append(float(np.mean(abs_weights)))
        
        # Only compute STD every 10 recorded steps to save time
        if len(self.history["weights_mean"]) % 10 == 0:
            self.history["weights_std"].append(float(np.std(weights)))
        else:
            self.history["weights_std"].append(0.0) # Placeholder

    def log_success(self, epoch: int, time_taken: float):
        """
        Record when agent reaches the goal.

        Args:
            epoch: Current epoch number.
            time_taken: Time in seconds to reach goal.
        """
        self.history["success_epochs"].append({
            "epoch": epoch,
            "time": round(time_taken, 2),
            "timestamp": time.time() - self.start_time
        })
        self.epochs_completed = epoch

    def log_wall_hit(self):
        """Increment wall hit counter."""
        self.total_wall_hits += 1

    def log_food_collected(self):
        """Increment food collection counter."""
        self.total_food_collected += 1

    def get_summary(self) -> Dict:
        """Get summary statistics for display."""
        return {
            "total_steps": len(self.history["steps"]),
            "epochs_completed": self.epochs_completed,
            "total_wall_hits": self.total_wall_hits,
            "total_food_collected": self.total_food_collected,
            "avg_reward": np.mean(self.history["rewards"]) if self.history["rewards"] else 0,
            "current_firing_rates": self.current_firing_rates,
            "current_energy_levels": self.current_energy_levels,
            "neural_silence": self.neural_silence_detected,
        }

    def get_avg_firing_rate(self, layer_name: str = "hidden") -> float:
        """Get current average firing rate for a layer."""
        return self.current_firing_rates.get(layer_name, 0.0)

    def get_avg_energy(self, layer_name: str = "hidden") -> float:
        """Get current average energy for a layer."""
        return self.current_energy_levels.get(layer_name, 0.0)

    def save_stats(self, filename: str = "simulation_stats.json"):
        """
        Save all recorded data to JSON file in experiments folder.

        Args:
            filename: Base output filename (will be timestamped).
        """
        from src.utils.file_manager import get_experiments_path, get_timestamped_filename
        
        # Add summary to output
        output = {
            "summary": self.get_summary(),
            "history": self.history,
            "metadata": {
                "record_interval": self.record_interval,
                "total_runtime": time.time() - self.start_time,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        
        # Create timestamped filename in experiments folder
        exp_path = get_experiments_path()
        base_name = filename.replace(".json", "")
        timestamped_name = get_timestamped_filename(base_name, ".json")
        full_path = exp_path / timestamped_name
        
        # Also save a "latest" version for easy access
        latest_path = exp_path / filename
        
        try:
            with open(full_path, "w") as f:
                json.dump(output, f, indent=2)
            with open(latest_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Stats saved to: {full_path}")
            return True
        except Exception as e:
            print(f"Error saving stats: {e}")
            return False

    def reset(self):
        """Reset all tracking data."""
        self.__init__(self.record_interval)
