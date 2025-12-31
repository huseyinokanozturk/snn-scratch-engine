from src.synapse import Synapse
from src.neuron import LIFNeuron
import matplotlib.pyplot as plt

"""
Pavlovian Conditioning test

In this test, we have a pre-synaptic neuron that emits a signal every 200 steps.

The post-synaptic neuron is stimulated by the pre-synaptic neuron and also receives a teacher signal.

The synapse is a plastic synapse that can change its weight based on the spikes of the pre-synaptic and post-synaptic neurons.

The weight of the synapse is updated using the STDP rule.
"""

neuron_pre =  LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)
neuron_post =  LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)

synapse = Synapse(weight=20, pre_neuron=neuron_pre, post_neuron=neuron_post, w_max=200, lr=40.0)

weight_history = []
neuron_pre_spikes = []
neuron_post_spikes = []


for i in range(2000):
    
    # Pre-synaptic neuron emits a signal every 200 steps and spikes when the signal is received
    current_pre_neuron = 0.0
    if i % 200 == 0:
        current_pre_neuron = 100

    spike_pre = neuron_pre.update(current_pre_neuron)

    # Transmit: If the pre-synaptic neuron spikes, the synapse transmits the current to the post-synaptic neuron
    input_from_synapse = 0.0
    if spike_pre:
        input_from_synapse = synapse.weight

    # Teacher signal: When pre-synaptic neuron spikes 3 times (200ms, 600ms, 1000ms) the signal affects the post-synaptic neuron after 10ms
    # That means we are training the post-synaptic neuron to spike when the pre-synaptic neuron spikes three times
    teacher_signal = 0.0
    if i < 1200: 
        if i % 400 == 210: 
            teacher_signal = 100.0 

    # Post-synaptic neuron updates its state based on the input from the synapse and the teacher signal
    total_input_post_neuron = input_from_synapse + teacher_signal
    spike_post = neuron_post.update(total_input_post_neuron)

    # Synapse updates its weight based on the spikes of the pre-synaptic and post-synaptic neurons:
    #   - Long-Term Potentiation (LTP): When the pre-synaptic neuron spikes before the post-synaptic neuron the weight increases
    #   - Long-Term Depression (LTD): When the post-synaptic neuron spikes before the pre-synaptic neuron the weight decreases
    #   - When the synapse's weight is strong enough, post-neuron will spike when the pre-synaptic neuron spikes even without any teacher signal
    synapse.step(spike_pre, spike_post)

    # Append the weight and spikes to the history for visualization
    weight_history.append(synapse.weight)
    neuron_pre_spikes.append(1 if spike_pre else 0)
    neuron_post_spikes.append(1 if spike_post else 0)


"""
Visualization to see the weights and spikes.
"""

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(weight_history, label='Weight', color='blue')
ax1.set_title('Synapse Weight Evolution')
ax1.set_ylabel('Weight')
ax1.legend(loc='upper right')

ax2.plot(neuron_pre_spikes, label='Pre-Synaptic Spikes', color='green')
ax2.plot(neuron_post_spikes, label='Post-Synaptic Spikes', color='red')
ax2.set_title('Neuron Spikes')
ax2.set_xlabel('Time (Step)')
ax2.set_ylabel('Spikes')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()    

    
