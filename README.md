# Hybrid Optimization Framework for Social Dynamics Modeling

## 🎯 Project Overview
This repository contains the technical implementation and research for my Master's thesis: **`Parametric Calibration of a Mathematical Model of the 2019 Chilean Social Unrest using a Hybrid Optimization Approach.`** The project addresses the challenges of calibrating discrete-time differential equations in systems with high numerical instability and complex spatial interactions.

## ⚖️ License
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This ensures that the software remains free to use, study, and modify, and that any derivative works are also distributed under the same license, protecting the open-source nature of this scientific contribution.

## 🛠️ Tech Stack & Key Innovations
- **Simulation Engine**: Built entirely from scratch using PyTorch. This implementation enables GPU acceleration (A100/L4), achieving a 112x speedup compared to the original NumPy/CPU baseline.
- **Hybrid Optimization Pipeline**:
  - Global Search: Comparison between a custom **Particle Swarm Optimization (PSO)** implemented in PyTorch and **Differential Evolution (DE)** (SciPy). **DE was selected as the superior approach** for identifying the optimal global parameter regions.
  - **Local Refinement**: **L-BFGS-B** (SciPy) algorithm used for high-precision convergence once the heuristic phase identifies a promising region.
- **Signal Processing & Preprocessing**: Data cleaning pipeline utilizing LOWESS Regression and cubic interpolation to generate differentiable and physically realistic target curves from noisy SOSAFE data.
- **Statistical Validation**: Goodness-of-fit evaluation through **Kolmogorov-Smirnov (K-S) tests**, **MAE**, and **RMSE** metrics.

## 🚀 Getting Started

Welcome! Whether you are a researcher, a student, or just curious about social dynamics modeling, this guide will help you get the simulation running in no time.

### 1. Prerequisites
We use **[uv](https://github.com/astral-sh/uv)** for lightning-fast Python package management. If you don't have it yet, you can install it via their official site, or just use standard `pip`.

### 2. Setting Up the Environment
Clone the repository and prepare your virtual environment:
```bash
# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Running Your First Calibration
The main entry point is `src/main.py`. You can choose between two optimization algorithms: **Differential Evolution (DE)** or **Particle Swarm Optimization (PSO)**.

**Try a quick run with PSO to see it in action:**
```bash
python src/main.py --optimizer pso --max_iter 10 --pop_size 5 --seed 42 --save --plot
```

### 4. Understanding the Controls
- `--optimizer`: Choose `de` (recommended for robust global search) or `pso`.
- `--max_iter`: How many generations the algorithm will run (higher = better convergence).
- `--seed`: Ensures you get the same results every time—essential for scientific auditability!
- `--save`: Automatically exports your results (JSON metrics and NPY vectors) to the `results/` folder.
- `--plot`: Generates a visual comparison between the model and the real-world SOSAFE data.

### 5. Configuring Search Bounds
The search space for parameters (`beta_r`, `gamma_r`, `alpha_p`, `gamma_p`) is managed via the `config.json` file in the root directory. You can adjust these ranges to explore different regions of the parameter space:
```json
{
    "bounds": {
        "beta_r": [0.05, 0.15],
        "gamma_r": [0.15, 0.25],
        "alpha_p": [0.05, 1.5],
        "gamma_p": [0.01, 0.05]
    }
}
```

### 6. Exploring the Output
Once the mission is complete, check the `results/` directory for:
- `best_results.json`: A detailed summary of the best parameters found and statistical metrics.
- `Rj_final.npy`: The raw simulation output for further analysis.
- `comparison_plot.png`: A high-resolution chart showing the model's calibration performance.

## 📂 Repository Structure
- `/notebooks:` Development history, including the transition from NumPy/TensorFlow experiments to the production-ready PyTorch architecture.
- `/src:` Modularized Python scripts including engine.py (differential equations), optimizer.py (Custom PSO class), and preprocessing.py.
- `/data:` Managed environment for ingesting SOSAFE reports and Santiago's Metro (Subway) network accessibility data.

## 🚀 Future Business Application (2026)
This framework is designed to be highly modular, serving as a template for **Industrial Data Analytics Consulting**. The core optimization engine can be adapted for logistics, inventory forecasting, and urban mobility challenges.
