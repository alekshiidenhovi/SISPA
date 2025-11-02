# SISPA: Sharded, Isolated, Sliced, Pre-computed, Aggregated

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This paper introduces SISPA, a novel framework for exact machine unlearning in deep neural networks. Building upon the principles of the SISA (Sharding, Isolation, Slicing, and Aggregation) framework from Bourtoule et al. ([Machine Unlearning](https://arxiv.org/abs/1912.03817)), SISPA proposes an alternative to the aggregation step. Instead of using conventional methods like majority voting, SISPA employs a fully-connected neural network layer to aggregate the results from sharded models, with the goal of improving the predictive performance of the final ensemble model. The framework maintains efficient unlearning through a pre-computation step that caches model embeddings, ensuring only the necessary models are updated during data removal. This research evaluates whether the SISPA framework can enhance predictive accuracy over standard SISA while retaining strong, efficient unlearning capabilities.

## The SISPA Framework

The SISPA framework is based on the following key ideas:

- **Sharded:** The training data is split into multiple disjoint shards.
- **Isolated:** Models are trained independently on each shard.
- **Sliced:** The SISA architecture allows for efficient unlearning. To unlearn a data point, only the model trained on the shard containing that data point needs to be retrained from scratch.
- **Pre-computed:** To make the unlearning process more efficient, embeddings from the shard-specific submodels are pre-computed and stored. When an unlearning request occurs, only the submodels affected by the data removal need to recompute their embeddings.
- **Aggregated:** Instead of using conventional aggregation methods like majority vote, SISPA uses a trainable, fully-connected neural network layer to combine the outputs from all sharded submodels. This approach is designed to improve the predictive performance of the ensemble. This aggregation layer is fully retrained after any unlearning operation.

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) installed. `uv` is used for environment and package management.

### Installation

1.  **Clone the repository:**
    ```bash
    https://github.com/alekshiidenhovi/SISPA
    cd SISPA
    ```

2.  **Run the initialization script:**
    This script will create a virtual environment, activate it, and install all the required dependencies.

    ```bash
    source scripts/init-project.sh
    ```

3.  **Set up W&B Logging (Optional):**
    If you want to log your experiments with Weights & Biases, copy the `.env.template` file to `.env` and fill in your W&B project name.

    ```bash
    cp .env.template .env
    # Now edit .env and add your WANDB_PROJECT
    ```

## Running Experiments

To run the full training and evaluation for the SISPA framework, use the following command:

```bash
python -m src.training.sispa_full_training
```

To run the SISA baseline:

```bash
python -m src.training.sisa_full_training
```

## Project Structure

```
SISPA/
├───.env.template           # Template for environment variables (e.g., W&B project)
├───.gitignore
├───README.md               # This file
├───requirements.in         # Base Python dependencies
├───requirements-dev.in     # Development dependencies
├───scripts/                # Helper scripts for setup and environment management
│   ├───activate-venv.sh
│   ├───compile-deps-dev.sh
│   ├───create-venv.sh
│   ├───init-project.sh
│   └───sync-deps-dev.sh
└───src/
    ├───common/             # Common utilities for config, logging, etc.
    ├───datasets/           # Code for dataset handling and splitting strategies
    ├───models/             # Model definitions (ResNet, SISPA)
    ├───storage/            # Utilities for saving/loading artifacts
    ├───tests/              # Tests for the project
    └───training/           # Main training scripts and sub-jobs
        ├───sisa_full_training.py
        ├───sispa_full_training.py
        └───subjobs/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
