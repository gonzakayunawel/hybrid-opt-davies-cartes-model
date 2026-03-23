import argparse
import numpy as np
import torch
import os
import sys

# Ensure src is in path if running directly
sys.path.append(os.getcwd())

from src.engine import DaviesModel
from src.preprocessing import scale_data, process_simulation_output
from src.utils import calculate_errors, print_metrics, plot_results
from src.optimizers import DifferentialEvolutionOptimizer, ParticleSwarmOptimizer

def load_data(data_dir):
    """
    Loads data from the data directory.
    """
    try:
        origin = np.loadtxt(os.path.join(data_dir, "origin_dens_500m_5am10am.dat"))
        destination = np.loadtxt(os.path.join(data_dir, "destination_dens_500m_5am10am.dat"))
        targets = np.loadtxt(os.path.join(data_dir, "targets_500.dat"))
        distances = np.load(os.path.join(data_dir, "rij_500_no_network.npy"), allow_pickle=False)
        return origin, destination, targets, distances
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def objective_function(params, model, target_data, processing_method='linear'):
    """
    Objective function for optimization.
    """
    # Params: beta_r, gamma_r, alpha_p, gamma_p
    # Check if params is a batch (for PSO)
    if isinstance(params, torch.Tensor) and params.dim() > 1:
        # Batch processing
        errors = []
        for p in params:
            beta_r, gamma_r, alpha_p, gamma_p = p.tolist()
            # Run simulation
            Rj = model.run_simulation(beta_r, gamma_r, alpha_p, gamma_p)
            # Process output
            # Note: Rj is raw output. Target data is likely scaled.
            # We need to scale Rj before comparison?
            # In DE notebook: Rj is scaled then smoothed then compared.
            # In PyTorch notebook: Rj is interpolated then scaled.

            # Let's assume we scale Rj to compare with scaled target.

            # Process (interpolate/smooth)
            # Rj_processed = process_simulation_output(Rj, method=processing_method)

            # Scale
            # scaled_Rj, _ = scale_data(Rj_processed)

            # Wait, scale_data uses MinMaxScaler which fits on data.
            # If we fit on predicted data, we lose absolute magnitude info if target was scaled globally?
            # In notebook:
            # scaler = MinMaxScaler()
            # Rj_scaled = scaler.fit_transform(Rj_t.reshape(-1, 1)).flatten()
            # This fits scaler on current Rj_t.
            # This means we are comparing SHAPES, not amplitudes?
            # Yes, "Comparison of Riot Targets".

            # So we scale each run independently?
            # Yes, that seems to be the logic in notebooks.

            # Implement processing chain
            processed_Rj = process_simulation_output(Rj, method=processing_method)
            scaled_Rj, _ = scale_data(processed_Rj)

            # Calculate error
            # We use RMSE or similar as objective?
            # Notebook uses np.linalg.norm(simulation_results - target_results)
            # which is Euclidean distance (L2 norm).
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

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    origin, destination, targets, distances = load_data(args.data_dir)
    Ii = origin + destination

    # Initialize model
    print("Initializing model...")
    model = DaviesModel(distances, Ii, targets, Nt=args.Nt, Ntt=args.Ntt)

    # Prepare target data (Scaled Zj)
    # Target is Zj[:, 2]
    target_raw = targets[:, 2]
    target_scaled, _ = scale_data(target_raw)

    # Define bounds
    # beta_r, gamma_r, alpha_p, gamma_p
    # From DE notebook:
    # bounds = [(0.05, 0.15), (0.15, 0.25), (0.05, 1.5), (0.01, 0.05)]
    bounds_de = [(0.05, 0.15), (0.15, 0.25), (0.05, 1.5), (0.01, 0.05)]

    # For PSO (PyTorch tensor)
    bounds_pso = torch.tensor([
        [0.05, 0.15],
        [0.15, 0.25],
        [0.05, 1.5],
        [0.01, 0.05]
    ])

    print(f"Starting optimization with {args.optimizer.upper()}...")

    best_params = None
    best_error = float('inf')

    if args.optimizer == "de":
        optimizer = DifferentialEvolutionOptimizer(
            objective_function,
            bounds_de,
            maxiter=args.max_iter,
            popsize=args.pop_size
        )
        # Pass model and target_data as args
        best_params, best_error = optimizer.optimize(model, target_scaled, args.method)

    elif args.optimizer == "pso":
        # Wrap objective function for PSO
        def pso_objective(params):
            return objective_function(params, model, target_scaled, args.method)

        optimizer = ParticleSwarmOptimizer(
            pso_objective,
            bounds_pso,
            max_iter=args.max_iter,
            n_particles=args.pop_size
        )
        best_params, best_error = optimizer.optimize()

    print("\nOptimization Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Error: {best_error}")

    # Final Run
    print("\nRunning final simulation with best parameters...")
    if isinstance(best_params, np.ndarray):
        best_params = best_params.tolist()

    Rj_final = model.run_simulation(*best_params)
    processed_Rj = process_simulation_output(Rj_final, method=args.method)
    scaled_Rj, _ = scale_data(processed_Rj)

    print_metrics(scaled_Rj, target_scaled)

    if args.plot:
        plot_results(target_scaled, scaled_Rj, title=f"Optimization Result ({args.optimizer.upper()})")

if __name__ == "__main__":
    main()
