# Spiking Neural Network (SNN) Engine from Scratch
### Reward-Modulated STDP Implementation

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![NumPy](https://img.shields.io/badge/Library-NumPy_Only-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=flat-square)

## 📖 Overview
This project is a biologically plausible **Spiking Neural Network (SNN)** simulation engine built entirely from scratch using **Python** and **NumPy**. It avoids high-level deep learning frameworks (like PyTorch or TensorFlow) to implement and analyze the core dynamics of **Third Generation Neural Networks** [1].

The primary goal is to solve classification tasks (e.g., MNIST) using **Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP)**, avoiding standard backpropagation. This mimics the brain's "Three-Factor Learning Rule":
1.  **Pre-synaptic activity** (Input)
2.  **Post-synaptic activity** (Output)
3.  **Global Neuromodulation** (Reward/Dopamine signal)

## 🚀 Key Features
* **No "Black Box" Libraries:** Every neuron update, spike propagation, and weight change is mathematically derived and coded manually.
* **LIF Neurons:** Implements Leaky Integrate-and-Fire dynamics with refractory periods.
* **R-STDP Learning:** A reinforcement learning approach applied to spiking neurons, enabling the network to learn from "correct/incorrect" feedback signals.
* **Vectorized Simulation:** Optimized using matrix operations for efficient CPU processing.

## 🧠 Theoretical Background
Unlike traditional ANNs (2nd Generation) that use continuous activation functions (Sigmoid/ReLU) and firing rates, this project utilizes **Spiking Neurons (3rd Generation)**. Information is encoded in the precise timing of single action potentials (spikes), which is computationally more powerful and biologically realistic [1].

## 🛠️ Tech Stack
* **Language:** Python
* **Core Logic:** NumPy (for linear algebra and vectorization)
* **Visualization:** Matplotlib (for raster plots and membrane potential traces)

## 📂 Project Structure
```text
snn-engine/
├── data/           # Dataset storage (e.g., MNIST spike-encoded)
├── src/            # Source code for the SNN engine
│   ├── neuron.py   # LIF Model physics
│   ├── network.py  # Layer management and propagation
│   └── stdp.py     # Learning rules (Synaptic Plasticity)
├── experiments/    # Training logs and visualization outputs
└── main.py         # Entry point for simulation
```

## 📅 Roadmap (5-Month Plan)
- [ ] **Month 1:** Modeling LIF Neuron physics & refractory periods.
- [ ] **Month 2:** Implementing Synapses & basic STDP learning rules.
- [ ] **Month 3:** Integrating Reward Modulation (Three-Factor Rule).
- [ ] **Month 4:** Encoding MNIST data into spike trains & Full training.
- [ ] **Month 5:** Analysis, ablation studies, and final reporting.

## 📚 References
1. Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.
