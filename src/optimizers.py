import numpy as np
import torch
import time
from scipy.optimize import differential_evolution
from rich.console import Console

console = Console()

class DifferentialEvolutionOptimizer:
    def __init__(self, objective_function, bounds, strategy='best1bin', popsize=10, tol=1e-6, maxiter=10):
        self.objective_function = objective_function
        self.bounds = bounds
        self.strategy = strategy
        self.popsize = popsize
        self.tol = tol
        self.maxiter = maxiter
        self.result = None

    def optimize(self, *args):
        """
        Runs the optimization.
        args: Additional arguments for the objective function.
        """
        start_time = time.time()
        self.result = differential_evolution(
            self.objective_function,
            self.bounds,
            args=args,
            strategy=self.strategy,
            popsize=self.popsize,
            tol=self.tol,
            disp=True,
            maxiter=self.maxiter
        )
        elapsed_time = time.time() - start_time
        return self.result.x, self.result.fun, elapsed_time

class ParticleSwarmOptimizer:
    def __init__(self, objective_function, bounds, n_particles=10, max_iter=10, w_range=(0.5, 0.5), c1=1.5, c2=1.5, device=None, tol=1e-6, patience=10):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_range = w_range
        self.c1 = c1
        self.c2 = c2
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tol = tol
        self.patience = patience
        self.best_position = None
        self.best_fitness = float('inf')

    def optimize(self): # Note: args must be handled by objective_function being a partial or wrapper
        start_time = time.time()
        # Initialize particles
        dim = self.bounds.shape[0]
        min_bounds = self.bounds[:, 0].to(self.device)
        max_bounds = self.bounds[:, 1].to(self.device)

        positions = (max_bounds - min_bounds) * torch.rand(self.n_particles, dim, device=self.device) + min_bounds
        velocities = torch.zeros(self.n_particles, dim, device=self.device)

        # Initial evaluation
        fitnesses = self.objective_function(positions)
        if fitnesses.ndim == 1:
            fitnesses = fitnesses.unsqueeze(1)

        # For initialization, we know `fitnesses` are the best seen so far (since it's the first step).
        personal_bests = fitnesses.clone()
        personal_best_pos = positions.clone()

        global_best_val, best_idx = torch.min(personal_bests, dim=0)
        global_best_pos = personal_best_pos[best_idx[0]].clone()

        patience_counter = 0
        last_best_val = global_best_val.item()

        for curr_iter in range(self.max_iter):
            iter_start = time.time()
            # Update w
            w = self.w_range[0] - (self.w_range[0] - self.w_range[1]) * (curr_iter / self.max_iter)

            r1, r2 = torch.rand(2, self.n_particles, dim, device=self.device)
            cognitive = self.c1 * r1 * (personal_best_pos - positions)
            social = self.c2 * r2 * (global_best_pos - positions)

            velocities = w * velocities + cognitive + social
            positions += velocities
            positions.clamp_(min=min_bounds, max=max_bounds)

            fitnesses = self.objective_function(positions)
            if fitnesses.ndim == 1:
                fitnesses = fitnesses.unsqueeze(1)

            # Update personal bests
            improved_mask = (fitnesses < personal_bests).squeeze()

            if improved_mask.any():
                personal_best_pos[improved_mask] = positions[improved_mask]
                personal_bests[improved_mask] = fitnesses[improved_mask]

            # Update global best
            current_best_val, current_best_idx = torch.min(personal_bests, dim=0)
            if current_best_val < global_best_val:
                global_best_val = current_best_val
                global_best_pos = personal_best_pos[current_best_idx[0]].clone()

            iter_time = time.time() - iter_start
            console.print(f"PSO Iter [bold cyan]{curr_iter+1}/{self.max_iter}[/bold cyan] - Error: [bold magenta]{global_best_val.item():.6f}[/bold magenta] - Time: [bold yellow]{iter_time:.2f}s[/bold yellow]")

            # Patience check
            if (last_best_val - global_best_val.item()) < self.tol:
                patience_counter += 1
            else:
                patience_counter = 0

            last_best_val = global_best_val.item()

            if patience_counter >= self.patience:
                console.print(f"[bold green]Convergence reached at iteration {curr_iter + 1}.[/bold green]")
                break

        self.best_position = global_best_pos
        self.best_fitness = global_best_val.item()
        elapsed_time = time.time() - start_time

        return self.best_position.cpu().numpy(), self.best_fitness, elapsed_time
