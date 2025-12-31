import numpy as np


class Decoder:
    """
    Converts spike arrays (binary) to continuous motor commands (analog).
    
    Method: Exponential Moving Average with Sharp Brake
    
    Features:
    - When spikes come, motor values increase (capacitor charges)
    - Without spikes, motor values decay (capacitor discharges)
    - Sharp brake mechanism for emergency stopping
    """

    def __init__(self, num_outputs: int, tau: float = 5.0, dt: float = 0.1):
        """
        Initialize the decoder.

        Args:
            num_outputs: Number of motor outputs.
            tau: Time constant (lower = faster response). Default 5.0 for sharp turns.
            dt: Time step.
        """
        self.num_outputs = num_outputs
        self.tau = tau
        self.dt = dt
        
        # Decay factor: lower tau = faster decay = more responsive
        self.decay = np.exp(-self.dt / self.tau)
        
        # Motor state
        self.motor_values = np.zeros(num_outputs)
        
        # Brake state
        self.brake_active = False

    def step(self, spikes: np.ndarray, brake: bool = False) -> np.ndarray:
        """
        Update motor values based on spikes.

        Args:
            spikes: Boolean spike array from motor neurons.
            brake: If True, apply emergency brake (rapid decay).

        Returns:
            Motor command values (0-1 range).
        """
        if brake:
            # Emergency brake: rapid decay to zero
            self.motor_values *= 0.5
            self.brake_active = True
        else:
            self.brake_active = False
            # Normal decay
            self.motor_values *= self.decay
            # Add spike contribution (gain = 0.6 for stronger response)
            self.motor_values += spikes.astype(float) * 0.6

        # Clip to valid range
        self.motor_values = np.clip(self.motor_values, 0.0, 1.0)

        return self.motor_values

    def apply_differential(self, left_sensor_avg: float, right_sensor_avg: float, strength: float = 0.3):
        """
        Apply differential modulation based on sensor asymmetry.
        This helps with smoother turning behavior.
        
        Args:
            left_sensor_avg: Average of left-side sensors (0-1).
            right_sensor_avg: Average of right-side sensors (0-1).
            strength: How much to modulate (0-1).
        """
        diff = right_sensor_avg - left_sensor_avg
        # If obstacle on right (diff > 0), boost left motor, reduce right
        self.motor_values[0] += diff * strength  # Left motor
        self.motor_values[1] -= diff * strength  # Right motor
        self.motor_values = np.clip(self.motor_values, 0.0, 1.0)

    def reset(self):
        """Stop the motors immediately."""
        self.motor_values = np.zeros(self.num_outputs)
        self.brake_active = False