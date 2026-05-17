import argparse
import numpy as np
import torch
import os
import sys
import json
import time
from datetime import datetime

# Ensure src is in path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import DaviesModel
from src.preprocessing import scale_data, process_simulation_output
from src.utils import print_metrics, plot_results, set_seed, save_results, plot_heatmaps
from src.optimizers import DifferentialEvolutionOptimizer, ParticleSwarmOptimizer
from rich.console import Console
from scipy.optimize import minimize

console = Console()

def load_config(config_path):
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(fr"[bold red]\[Error][/bold red] Config file not found at {config_path}. Using default bounds.")
        return {
            "bounds": {
                "beta_r": [0.05, 0.15],
                "gamma_r": [0.15, 0.25],
                "alpha_p": [0.05, 1.5],
                "gamma_p": [0.01, 0.05]
            }
        }

def load_data(data_dir):
    """
    Loads data from the data directory.
    """
    try:
        origin = np.loadtxt(os.path.join(data_dir, "origin_dens_500m_5am10am.dat"))
        destination = np.loadtxt(os.path.join(data_dir, "destination_dens_500m_5am10am.dat"))
        targets = np.loadtxt(os.path.join(data_dir, "targets_500.dat"))
        distances = np.load(os.path.join(data_dir, "rij_500_no_network.npy"))
        return origin, destination, targets, distances
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def objective_function(params, model, target_data, processing_method="linear"):
    """
    Objective function to minimize (L2 norm of difference).
    """
    if params.ndim == 2:
        # Batch processing for PSO
        errors = []
        for p in params:
            beta_r, gamma_r, alpha_p, gamma_p = p.tolist()
            Rj = model.run_simulation(beta_r, gamma_r, alpha_p, gamma_p)
            processed_Rj = process_simulation_output(Rj, method=processing_method)
            scaled_Rj, _ = scale_data(processed_Rj)
            error = np.linalg.norm(scaled_Rj - target_data)
            errors.append(error)

        return torch.tensor(errors, device=model.device).unsqueeze(1)
    else:
        # Single parameter set (DE or single PSO evaluation)
        if isinstance(params, np.ndarray):
            beta_r, gamma_r, alpha_p, gamma_p = params
        elif isinstance(params, list):
            beta_r, gamma_r, alpha_p, gamma_p = params
        elif isinstance(params, torch.Tensor):
             beta_r, gamma_r, alpha_p, gamma_p = params.tolist()
        else:
            raise TypeError(f"Unsupported params type: {type(params)}")

        Rj = model.run_simulation(beta_r, gamma_r, alpha_p, gamma_p)
        processed_Rj = process_simulation_output(Rj, method=processing_method)
        scaled_Rj, _ = scale_data(processed_Rj)

        return np.linalg.norm(scaled_Rj - target_data)

def main():
    parser = argparse.ArgumentParser(description="Davies Model Optimization")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--optimizer", type=str, choices=["de", "pso"], default="de", help="Optimizer to use (de or pso)")
    parser.add_argument("--method", type=str, choices=["linear", "lowess"], default="linear", help="Processing method")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--pop_size", type=int, default=10, help="Population size")
    parser.add_argument("--Nt", type=int, default=500, help="Simulation steps Nt")
    parser.add_argument("--Ntt", type=int, default=10, help="Simulation steps Ntt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results", help="Base directory to save results")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark (NumPy vs CuPy vs PyTorch)")

    args = parser.parse_args()

    # Load data
    console.print("Loading data...")
    origin, destination, targets, distances = load_data(args.data_dir)
    Ii = origin + destination

    # Initialize model
    console.print("Initializing model...")
    model = DaviesModel(distances, Ii, targets, Nt=args.Nt, Ntt=args.Ntt)

    # Prepare target data (Scaled Zj)
    target_raw = targets[:, 2]
    target_scaled, _ = scale_data(target_raw)

    # Load configuration
    config = load_config(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    b = config["bounds"]
    
    # Define bounds for DE (list of tuples)
    bounds_de = [
        tuple(b["beta_r"]),
        tuple(b["gamma_r"]),
        tuple(b["alpha_p"]),
        tuple(b["gamma_p"])
    ]

    # For PSO (PyTorch tensor)
    bounds_pso = torch.tensor([
        b["beta_r"],
        b["gamma_r"],
        b["alpha_p"],
        b["gamma_p"]
    ])

    # Reproducibility
    set_seed(args.seed)

    if args.benchmark:
        from src.benchmark import perform_benchmark
        # Use baseline params for benchmark
        baseline_params = [0.1, 0.19, 0.97, 0.034] 
        perform_benchmark(args.data_dir, baseline_params)
        return

    # Mission Identifier and Directory Setup
    mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_dir = os.path.join(args.output_dir, f"mission_{mission_id}") if args.save else None

    console.print(fr"Mission ID: [bold cyan]{mission_id}[/bold cyan]")
    console.print(fr"Starting optimization with [bold cyan]{args.optimizer.upper()}[/bold cyan]...")

    best_params = None
    best_error = float('inf')
    elapsed_time = 0.0

    if args.optimizer == "de":
        optimizer = DifferentialEvolutionOptimizer(
            objective_function,
            bounds_de,
            maxiter=args.max_iter,
            popsize=args.pop_size
        )
        best_params, best_error, elapsed_time = optimizer.optimize(model, target_scaled, args.method)

    elif args.optimizer == "pso":
        def pso_objective(params):
            return objective_function(params, model, target_scaled, args.method)

        optimizer = ParticleSwarmOptimizer(
            pso_objective,
            bounds_pso,
            max_iter=args.max_iter,
            n_particles=args.pop_size
        )
        best_params, best_error, pso_time = optimizer.optimize()
        
        # HYBRID REFINEMENT: L-BFGS-B polishing for PSO
        console.print("Polishing PSO solution with [bold yellow]L-BFGS-B[/bold yellow]...")
        polish_start = time.time()
        res = minimize(
            objective_function,
            best_params,
            args=(model, target_scaled, args.method),
            method='L-BFGS-B',
            bounds=bounds_de
        )
        best_params = res.x
        best_error = res.fun
        elapsed_time = pso_time + (time.time() - polish_start)

    console.print("\n[bold green]Optimization Completed![/bold green]")
    console.print(f"Best Parameters: [bold yellow]{best_params}[/bold yellow]")
    console.print(f"Best Error: [bold magenta]{best_error:.6f}[/bold magenta]")
    console.print(f"Total Time: [bold yellow]{elapsed_time:.2f}s[/bold yellow]")

    # Final Run
    console.print("\nRunning final simulation with best parameters...")
    if isinstance(best_params, np.ndarray):
        best_params = best_params.tolist()

    Rj_final = model.run_simulation(*best_params)
    processed_Rj = process_simulation_output(Rj_final, method=args.method)
    scaled_Rj, _ = scale_data(processed_Rj)

    print_metrics(scaled_Rj, target_scaled)

    if args.save:
        save_results(
            best_params, 
            best_error, 
            elapsed_time, 
            args.seed, 
            scaled_Rj, 
            target_scaled, 
            mission_dir, 
            args.optimizer,
            config["bounds"],
            mission_id
        )

    if args.plot:
        plot_results(target_scaled, scaled_Rj, title=f"Optimization Result ({args.optimizer.upper()})", output_dir=mission_dir if args.save else None, optimizer_name=args.optimizer if args.save else None)
        plot_heatmaps(target_scaled, scaled_Rj, targets, title=f"Spatial Intensity Map ({args.optimizer.upper()})", output_dir=mission_dir if args.save else None, optimizer_name=args.optimizer if args.save else None)

if __name__ == "__main__":
    main()
