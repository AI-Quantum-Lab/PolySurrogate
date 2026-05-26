# Surrogate Model for Photonic Quantum State Design

This repository contains code for generating data, training surrogate neural-network models, and running optimisation workflow.

The main goal is to learn a surrogate model that maps graph/edge-weight representations of photonic setups to output quantum-state amplitudes. The trained model can then be used for fast evaluation and inverse-design experiments.

## Repository structure

```text
surrogate_model_clean/
├── config.py
├── data_generation.py
├── data_generation_utils.py
├── model_training.py
├── model_training_utils.py
├── model_training_streaming.py
├── model_training_streaming_utils.py
├── models.py
├── optimiser.py
├── optimisation_utils.py
├── target_states.py
├── optimiser_quick_test.ipynb
├── play_ground.ipynb
└── README.md