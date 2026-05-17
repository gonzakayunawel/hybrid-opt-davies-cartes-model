import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
from rich.console import Console
from rich.table import Table

console = Console()

def set_seed(seed: int):
    """
    Sets seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    console.print(fr"[bold blue]\[Discovery][/bold blue] Seed set to: [bold cyan]{seed}[/bold cyan]")

def save_results(params, error, elapsed_time, seed, Rj_final, target_scaled, output_dir, optimizer_name, bounds):
    """
    Saves optimization results and metrics to a JSON file and final simulation to NPY.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        console.print(fr"[bold blue]\[Discovery][/bold blue] Created output directory: [italic]{output_dir}[/italic]")

    # Calculate metrics and handle NaNs for JSON compliance
    raw_metrics = calculate_errors(Rj_final, target_scaled)
    clean_metrics = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in raw_metrics.items()
    }

    # Save metrics and params to JSON
    results_data = {
        "optimizer": optimizer_name,
        "seed": seed,
        "bounds": bounds,
        "best_params": params if isinstance(params, list) else params.tolist(),
        "best_error": float(error),
        "elapsed_time_seconds": float(elapsed_time),
        "metrics": clean_metrics
    }

    json_path = os.path.join(output_dir, "best_results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=4)
    
    # Save final simulation Rj
    npy_path = os.path.join(output_dir, "Rj_final.npy")
    np.save(npy_path, Rj_final)

    console.print(fr"[bold blue]\[Discovery][/bold blue] Results saved to: [bold green]{output_dir}[/bold green]")

def calculate_errors(predicted, real):
    """
    Calculates error metrics between predicted and real values.
    """
    predicted = predicted.flatten()
    real = real.flatten()

    mse = mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real, predicted)
    r2 = r2_score(real, predicted)

    # MAPE with zero handling
    non_zero_mask = (real != 0)
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((real[non_zero_mask] - predicted[non_zero_mask]) / real[non_zero_mask])) * 100
    else:
        mape = np.nan

    # SMAPE
    epsilon = 1e-10
    smape = np.mean(2 * np.abs(real - predicted) / (np.abs(real) + np.abs(predicted) + epsilon)) * 100

    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(real, predicted)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "smape": float(smape),
        "r2": float(r2),
        "max_error": float(np.max(np.abs(real - predicted))),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p_value)
    }

    return metrics

def print_metrics(predicted, real):
    """
    Prints detailed error metrics using a rich Table.
    """
    metrics = calculate_errors(predicted, real)

    table = Table(title="[bold blue]Error Metrics[/bold blue]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("RMSE", f"{metrics['rmse']:.4f}")
    table.add_row("MAE", f"{metrics['mae']:.4f}")
    table.add_row("MAPE", f"{metrics['mape']:.2f}%")
    table.add_row("SMAPE", f"{metrics['smape']:.2f}%")
    table.add_row("R² Score", f"{metrics['r2']:.4f}")
    table.add_row("Max Error", f"{metrics['max_error']:.4f}")
    table.add_row("KS Stat", f"{metrics['ks_stat']:.4f}")
    table.add_row("KS p-value", f"{metrics['ks_p_value']:.4f}")

    console.print(table)

def plot_results(real, predicted, title="Comparison", output_path=None):
    """
    Plots the real vs predicted values.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, y=1.05)

    x1 = np.arange(len(real))
    ax1.plot(x1, real, label='Real')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Real Data')
    ax1.grid(True, linestyle='--', alpha=0.5)

    x2 = np.arange(len(predicted))
    ax2.plot(x2, predicted, label='Predicted', color='orange')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Predicted Data')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        console.print(fr"[bold blue]\[Discovery][/bold blue] Plot saved to: [bold green]{output_path}[/bold green]")
    
    plt.show()
