import numpy as np

class Decoder:
    """
    Converts spike arrays(binary) to continuous motor commands (analog).
    Method: Exponential Moving Average
    Logic: When spikes come condansator charges (speed increases), if not discharges slowly (speed decreases).
    """

    def __init__(self, num_outputs: int, tau: float = 100.0, dt: float = 0.1):
        """
        Initialize the decoder.
        
        Args:
            num_outputs (int): Number of outputs.
            tau (float): Time constant.
            dt (float): Time step.
        """
        self.num_outputs = num_outputs
        self.tau = tau
        self.dt = dt

        self.decay = np.exp(-self.dt / self.tau)

        self.motor_values = np.zeros(num_outputs)

    def step(self, spikes: np.ndarray) -> np.ndarray:
        """
        Update the motor values based on the spikes.
        
        Args:
            spikes (np.ndarray): Spike array.
        
        Returns:
            np.ndarray: Updated motor values.
        """
        self.motor_values *= self.decay

        self.motor_values += spikes.astype(float) * 0.5 # 0.5 is the gain

        self.motor_values = np.clip(self.motor_values, 0.0, 1.0)

        return self.motor_values

    def reset(self):
        """
        Stop the motors.
        """
        self.motor_values = np.zeros(self.num_outputs)

        
    