# Hybrid Optimization Framework for Social Dynamics Modeling

## 🎯 Project Overview
This repository contains the technical implementation and research for my Master's thesis: "Parametric Calibration of a Mathematical Model of the 2019 Chilean Social Unrest using a Hybrid Optimization Approach." The project addresses the challenges of calibrating discrete-time differential equations in systems with high numerical instability and complex spatial interactions.

## 🛠️ Tech Stack & Key Innovations
- Simulation Engine: Built entirely from scratch using PyTorch. This implementation enables GPU acceleration (A100/L4), achieving a 112x speedup compared to the original NumPy/CPU baseline.
- Hybrid Optimization Pipeline:
  - Global Search: Custom Particle Swarm Optimization (PSO) implemented in PyTorch for high-dimensional parameter space exploration.
  - Local Refinement: L-BFGS-B (SciPy) algorithm used for high-precision convergence once the heuristic phase identifies a promising region.
- Signal Processing & Preprocessing: Data cleaning pipeline utilizing LOWESS Regression and cubic interpolation to generate differentiable and physically realistic target curves from noisy SOSAFE data.
- Statistical Validation: Goodness-of-fit evaluation through Kolmogorov-Smirnov (K-S) tests, MAE, and RMSE metrics.

## 📂 Repository Structure
- `/notebooks:` Development history, including the transition from NumPy/TensorFlow experiments to the production-ready PyTorch architecture.
- `/src:` Modularized Python scripts including engine.py (differential equations), optimizer.py (Custom PSO class), and preprocessing.py.
- `/data:` Managed environment for ingesting SOSAFE reports and Santiago's Metro (Subway) network accessibility data.

## 🚀 Future Business Application (2026)
This framework is designed to be highly modular, serving as a template for Industrial Data Analytics Consulting. The core optimization engine can be adapted for logistics, inventory forecasting, and urban mobility challenges.
