# Hybrid Optimization Framework for Social Dynamics Modeling

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20262195-blue)](https://doi.org/10.5281/zenodo.20262195)

## 🎯 Project Overview
This repository contains the technical implementation and research for my Master's thesis: **`Parametric Calibration of a Mathematical Model of the 2019 Chilean Social Unrest using a Hybrid Optimization Approach.`** The project addresses the challenges of calibrating discrete-time differential equations in systems with high numerical instability and complex spatial interactions.

## ⚖️ License
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This ensures that the software remains free to use, study, and modify, and that any derivative works are also distributed under the same license, protecting the open-source nature of this scientific contribution.

## 🛠️ Tech Stack & Key Innovations
- **Simulation Engine**: Built entirely from scratch using PyTorch. This implementation enables GPU acceleration (A100/L4), achieving a 112x speedup compared to the original NumPy/CPU baseline.
- **Hybrid Optimization Pipeline**:
  - Global Search: Comparison between a custom **Particle Swarm Optimization (PSO)** implemented in PyTorch and **Differential Evolution (DE)** (SciPy). **DE was selected as the superior approach** for identifying the optimal global parameter regions.
  - **Local Refinement**: **L-BFGS-B** (SciPy) algorithm used for high-precision convergence once the heuristic phase identifies a promising region.
- **Signal Processing & Preprocessing**: Data cleaning pipeline utilizing LOWESS Regression and linear interpolation to generate differentiable and physically realistic target curves from noisy SOSAFE data.
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

**Run a performance benchmark (NumPy vs CuPy vs PyTorch):**
```bash
python src/main.py --benchmark
```

### 4. Understanding the Controls
- `--optimizer`: Choose `de` (recommended for robust global search) or `pso`.
- `--method`: Signal processing method — `linear` (default, fast) or `lowess` (smoother, slower).
- `--max_iter`: How many generations the algorithm will run (higher = better convergence).
- `--seed`: Ensures you get the same results every time—essential for scientific auditability!
- `--save`: Automatically exports your results to the `results/` folder, creating a unique subfolder with a timestamp for every mission.
- `--plot`: Generates diagnostic plots and spatial intensity maps.

### 5. Configuring Search Bounds
The search space for parameters (`beta_r`, `gamma_r`, `alpha_p`, `gamma_p`) is managed via the `config.json` file. You can adjust these ranges to explore different regions of the parameter space.

### 6. Exploring the Output
Each experiment creates a timestamped folder inside `results/` (e.g., `mission_20260516_183005/`) containing:
- `[optimizer]_best_results.json`: Full mission report including mission ID, timestamp, hardware profile (OS, CPU, RAM, GPU), optimizer, seed, search bounds, best parameters found, and statistical metrics.
- `[optimizer]_Rj_final.npy`: The raw simulation output (normalized attack density) for further analysis.
- `[optimizer]_ranked_activity.png`: Comparison of real vs predicted intensity ranked by site.
- `[optimizer]_density_distribution.png`: Histogram comparison of attack densities.
- `[optimizer]_ecdf.png`: Empirical Cumulative Distribution Function comparison.
- `[optimizer]_residuals.png`: Spatial error analysis (Real - Predicted).
- `[optimizer]_spatial_heatmap.png`: High-resolution side-by-side geographic comparison.

## 📂 Repository Structure
- `/notebooks`: Development history, including the transition from NumPy/TensorFlow experiments to the production-ready PyTorch architecture.
- `/src`: Modularized Python scripts:
  - `engine.py`: Davies model differential equations (PyTorch).
  - `optimizers.py`: Custom PSO and Differential Evolution wrappers.
  - `preprocessing.py`: Signal processing pipeline (peak detection, interpolation, LOWESS).
  - `utils.py`: Metrics, plotting, hardware diagnostics, and result persistence.
  - `benchmark.py`: Performance and numerical equivalence benchmark (NumPy vs CuPy vs PyTorch).
  - `main.py`: CLI entry point.
- `/data`: Managed environment for ingesting SOSAFE reports and Santiago's Metro (Subway) network accessibility data.
- `config.json`: Search bounds for the four model parameters.

## 🙏 Acknowledgements

The original NumPy simulation of the Davies model was developed by **[Dr. Carlos Gabriel Cartes Moraga](https://orcid.org/0000-0001-6770-4983)** (Universidad San Sebastián, Chile), who served as thesis advisor for this project. His implementation served as the ground-truth reference for the PyTorch port, the hybrid optimization framework, and the numerical equivalence validation presented here.

All work beyond the original NumPy baseline — including the PyTorch engine, the PSO/DE optimizers, the preprocessing pipeline, and the benchmarking infrastructure — was developed independently by the thesis author.

## 📄 Citation

If you use this software in your research, please cite it as:

```bibtex
@software{cayunao_erices_2026_hybrid_opt,
  author       = {Gonzalo Cayunao-Erices},
  title        = {{Technical implementation: Parametric Calibration of the 2019 Chilean Social Unrest}},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.20262195},
  url          = {https://doi.org/10.5281/zenodo.20262195}
}
```

## 🚀 Future Business Application (2026)
This framework is designed to be highly modular, serving as a template for **Industrial Data Analytics Consulting**. The core optimization engine can be adapted for logistics, inventory forecasting, and urban mobility challenges.
