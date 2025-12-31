import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.neuron import LIFNeuron

neuron = LIFNeuron(tau=10, dt=0.1, threshold=1.0, rest_potential=0, reset_potential=0, refractory_time=2)

neuron.energy_max = 100.0
neuron.energy = 100.0

neuron.energy_cost = 15.0      
neuron.recovery_rate = 0.01    
neuron.fatigue_factor = 0.2   

input_current = 5.0  
sim_steps = 1500

volts = []
threshs = []
energies = []
spikes = []

for _ in range(sim_steps):
    spike = neuron.update(input_current)
    
    volts.append(neuron.current_voltage)
    threshs.append(neuron.current_threshold)
    energies.append(neuron.energy)
    spikes.append(1 if spike else 0)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(volts, label='Membrane Voltage', color='blue', alpha=0.6)
ax1.plot(threshs, label='Dynamic Threshold (Fatigue)', color='red', linestyle='--')
ax1.set_title('Homeostasis: Energy Decreases, Threshold Increases')
ax1.set_ylabel('Voltage (mV)')
ax1.legend(loc='upper right')

ax2.plot(energies, label='ATP Energy', color='green', linewidth=2)
ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
ax2.set_title('Metabolic Energy Consumption')
ax2.set_ylabel('Energy')
ax2.set_xlabel('Time (Step)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()