# Project VITALIS: Energy-Constrained Neuromorphic Learning Agent

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![NumPy](https://img.shields.io/badge/Library-NumPy_Only-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=flat-square)

## Overview

This project is a biologically plausible **Spiking Neural Network (SNN)** simulation engine built entirely from scratch using **Python** and **NumPy**. Rather than using high-level frameworks like PyTorch, I am building the mathematical backend to analyze the core dynamics of Third Generation Neural Networks.

The primary goal of Project VITALIS is to create an **autonomous agent** capable of navigation and survival. Unlike traditional AI agents that optimize solely for reward, this agent operates under **metabolic constraints**. It must manage its energy levels, utilize sleep cycles for memory consolidation, and adapt its learning rate based on environmental context (hunger/satiety).

## Key Features & Biological Mechanisms

This engine combines several neuroscientific concepts into a single cohesive system:

* **R-STDP (Reward-Modulated Spike-Timing-Dependent Plasticity):** Implements the brain's "Three-Factor Learning Rule" (Pre-synaptic activity, Post-synaptic activity, and Dopamine reward signal) to solve the credit assignment problem.
* **Metabolic Neurons:** Neurons consume energy with every spike. The system must learn to be energy-efficient (sparse coding) to survive.
* **Homeostasis:** Dynamic threshold adaptation prevents the network from becoming hyperactive (epileptic) or dormant.
* **Neuromodulation:** Global parameters like Dopamine (reward) and Acetylcholine (attention) are regulated by the agent's internal state (e.g., battery/hunger levels).

## Tech Stack

* **Language:** Python
* **Core Logic:** NumPy (vectorized matrix operations for performance)
* **Visualization:** Matplotlib / PyGame (for real-time agent visualization and spike rasters)

## Project Structure

```text
project-vitalis/
├── docs/               # Documentation and references
├── experiments/        # Logs and weight checkpoints
├── src/                # Core engine source code
│   ├── __init__.py
│   ├── neuron.py       # Metabolic LIF Neuron logic
│   ├── synapse.py      # R-STDP and Eligibility Trace implementation
│   ├── network.py      # Vectorized network management
│   ├── environment.py  # 2D Simulation world (Robot, Food, Walls)
│   └── monitor.py      # State monitoring (Energy, Spikes, Rewards)
├── tests/              # Unit tests for STDP and biological mechanics
├── main.py             # Main simulation entry point
├── requirements.txt
└── README.md