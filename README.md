<h1 align="center"> SCOUT: Language-based Trial and Error Fails in the Era of Experience </h1>

<p align="center">
  <strong>Sub-Scale Collaboration On Unseen Task (SCOUT)</strong><br>
  <em>Decoupling Exploration from Exploitation for Efficient LLM Agent Training</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/📄_Paper-Arxiv-EA4335?style=for-the-badge&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/Harry-mic/SCOUT"><img src="https://img.shields.io/badge/💻_Code-GitHub-181717?style=for-the-badge&logoColor=white" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License"></a>
</p>

## 📖 Overview

**SCOUT** is a novel framework that addresses the inefficiency of Large Language Models (LLMs) in exploring unseen, non-linguistic environments (e.g., symbolic or spatial tasks).

While LLMs excel at exploitation (reasoning based on knowledge), they are computationally expensive and inefficient at exploration (trial-and-error). SCOUT decouples these two processes:
1.  **Lightweight Scouts:** Use small networks (MLPs/CNNs) to rapidly master environmental dynamics via standard RL.
2.  **Sub-Scale Collaboration:** Distill the scout's expert trajectories into the LLM via SFT.
3.  **Evolution:** Activate the LLM's latent world knowledge through multi-turn RL (PPO).

Empirically, SCOUT enables a **Qwen2.5-3B** model to achieve an average score of **0.86** on complex tasks (including Rubik's Cube and 2048), significantly performing proprietary models like **Gemini-2.5-Pro (0.60)**, while reducing GPU hours by **~60%**.

This repository is built upon the [RAGEN](https://github.com/RAGEN-AI/RAGEN) framework.

## 🚀 The SCOUT Framework

<p align="center"><img src="pipeline117.png" width="800px" alt="SCOUT Framework Overview" /></p>

[cite_start]The training pipeline consists of three distinct stages[cite: 139, 140, 141]:

1.  **Exploration Stage (Scout Training):**
    * [cite_start]Agents: Small MLPs or CNNs ($~10^{-5}$B parameters)[cite: 604].
    * [cite_start]Algorithm: DQN or PPO[cite: 314].
    * [cite_start]Goal: Efficiently map transition dynamics and generate expert trajectories ($\tau_{scout}$)[cite: 222, 337].

2.  **Distillation Stage (SFT):**
    * [cite_start]Process: Transform $\tau_{scout}$ into text-based dialogue formats using a deterministic *Textualizer*[cite: 308].
    * [cite_start]Goal: "Warm up" the LLM to understand the physics of the unseen task[cite: 140, 223].

3.  **Evolving Stage (Multi-turn RL):**
    * [cite_start]Algorithm: Multi-turn PPO (via RAGEN)[cite: 428].
    * [cite_start]Goal: Refine reasoning and enable the LLM to self-evolve beyond the scout's capabilities[cite: 141, 599].

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/Harry-mic/SCOUT.git](https://github.com/Harry-mic/SCOUT.git)
cd SCOUT

# Setup the environment (based on RAGEN)
bash scripts/setup_ragen.sh
```

## 🎮 **Environments**

We introduce several OOD (Out-of-Distribution) symbolic and spatial tasks:

Rubik's Cube: Restore a 2x2 scrambled cube (spatial reasoning).

2048: Long-horizon planning (>800 turns).

Sudoku: Logic-based constraint satisfaction.

Sokoban: Box-pushing planning task.

FrozenLake: Stochastic navigation (Static & Slippery variants).

Bandit: Fundamental RL benchmark.



## ⚡ Usage

### 1. Exploration Stage (Train Scouts)
Train lightweight scouts (MLP/CNN) to collect expert trajectories.
```bash
# Example: Train a DQN scout for Rubik's Cube
python scout/train_scout.py --env RubiksCube --algo dqn --output_dir data/trajectories
```

### 2. Distillation Stage (SFT)
Fine-tune the base LLM on the collected trajectories. We utilize LLaMA-Factory for this stage.
```bash
# Format data and run SFT
bash scripts/run_sft.sh --model Qwen/Qwen2.5-3B-Instruct --data_path data/trajectories
```

### 3. Evolving Stage (Multi-turn RL)
Run multi-turn PPO on the SFT model using the RAGEN infrastructure.

Configure your experiment: Edit config/base.yaml or create a new config to select your environment (e.g., rubiks_cube, 2048).

Start Training:

```bash

# Train using the base configuration
python train.py --config-name base

# Or specifying parameters for a specific environment
python train.py --config-name base env=rubiks_cube actor.model.path=/path/to/sft/checkpoint
```


## 📊 Performance

| Model | Average Score | GPU Hours (Rubik's Cube) |
| :--- | :---: | :---: |
| **SCOUT (Qwen2.5-3B)** | **0.86** | **9.6h** |
| Gemini-2.5-Pro | 0.60 | - |
| GPT-4o-mini | 0.38 | - |
| Vanilla RL (PPO) | 0.38 | 24.0h |

*SCOUT achieves state-of-the-art performance on unseen tasks while saving 60% of computational costs compared to direct RL training.*

## 📂 Repository Structure

```text
SCOUT/
├── ragen/                  # Core RAGEN framework (Env Manager, Context Manager)
├── scout/                  # Lightweight scout training (DQN/PPO) & Textualizers
├── config/                 # Hydra configurations for PPO/GRPO
├── scripts/                # Setup and utility scripts
└── train.py                # Main entry point for Evolving Stage
```

## 📜 Citation
If you find SCOUT useful for your research, please cite our paper:

```text
@article{scout2026,
  title={Language-based Trial and Error Fails in the Era of Experience},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2026}
}
```
## Acknowledgements
This codebase is built upon RAGEN. We thank the RAGEN team for their infrastructure support.
