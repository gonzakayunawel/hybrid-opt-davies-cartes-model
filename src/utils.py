import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import json
import os
import platform
import psutil
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def get_system_info():
    """
    Detects hardware characteristics: CPU, RAM, and GPU.
    """
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown CPU",
        "cores": psutil.cpu_count(logical=True),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "gpu": "None"
    }
    
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    
    return info

def print_system_info():
    """
    Prints system hardware information in a rich Panel.
    """
    info = get_system_info()
    content = (
        f"[bold cyan]OS:[/bold cyan] {info['os']}\n"
        f"[bold cyan]CPU:[/bold cyan] {info['cpu']} ({info['cores']} cores)\n"
        f"[bold cyan]RAM:[/bold cyan] {info['ram']}\n"
        f"[bold cyan]GPU:[/bold cyan] {info['gpu']}"
    )
    console.print(Panel(content, title=r"[bold blue]\[AGS-S 201 Discovery] Hardware Diagnostics[/bold blue]", expand=False))

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
    console.print(r"[bold blue]\[Discovery][/bold blue] Seed set to: [bold cyan]" + str(seed) + r"[/bold cyan]")

def save_results(params, error: float, elapsed_time: float, seed: int, Rj_final: np.ndarray, target_scaled: np.ndarray, output_dir: str, optimizer_name: str, bounds: dict, mission_id: str):
    """
    Saves optimization results and metrics to a JSON file and final simulation to NPY.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        console.print(r"[bold blue]\[Discovery][/bold blue] Created mission directory: [italic]" + output_dir + r"[/italic]")

    # Calculate metrics and handle NaNs for JSON compliance
    raw_metrics = calculate_errors(Rj_final, target_scaled)
    clean_metrics = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in raw_metrics.items()
    }

    # Save metrics and params to JSON
    results_data = {
        "mission_id": mission_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimizer": optimizer_name,
        "seed": seed,
        "bounds": bounds,
        "best_params": np.asarray(params).tolist(),
        "best_error": float(error),
        "elapsed_time_seconds": float(elapsed_time),
        "metrics": clean_metrics
    }

    json_path = os.path.join(output_dir, f"{optimizer_name}_best_results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=4)
    
    # Save final simulation Rj
    npy_path = os.path.join(output_dir, f"{optimizer_name}_Rj_final.npy")
    np.save(npy_path, Rj_final)

    console.print(r"[bold blue]\[Discovery][/bold blue] Results saved to: [bold green]" + output_dir + r"[/bold green]")

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

def plot_results(real, predicted, title="Optimization Analysis", output_dir=None, optimizer_name=None):
    """
    Plots the real vs predicted values as separate files: ranked activity, histograms, ECDF, and residuals.
    """
    prefix = f"{optimizer_name}_" if optimizer_name else ""

    # Shared data preparation
    sort_idx = np.argsort(real)[::-1]
    real_sorted = real[sort_idx]
    pred_sorted = predicted[sort_idx]
    x_rank = np.arange(len(real))

    # 1. Ranked Activity Comparison
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.fill_between(x_rank, real_sorted, color='blue', alpha=0.3, label='Real Intensity (SOSAFE)')
    ax1.plot(x_rank, pred_sorted, color='orange', linewidth=1.5, label='Predicted Intensity (Model)')
    ax1.set_title(f'Riot Activity Level per Site (Ranked) - {title}', fontsize=14)
    ax1.set_xlabel('Riot Site Rank (Ordered by Real Intensity)', fontsize=12)
    ax1.set_ylabel('Normalized Attack Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}ranked_activity.png")
        fig1.savefig(path, bbox_inches='tight')
        console.print(r"[bold blue]\[Discovery][/bold blue] Ranked activity plot saved to: [bold green]" + path + r"[/bold green]")
    plt.show()
    plt.close(fig1)

    # 2. Histogram / Density Comparison
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.hist(real, bins=30, alpha=0.5, label='Real Attacks', color='blue', density=True)
    ax2.hist(predicted, bins=30, alpha=0.5, label='Predicted Activity', color='orange', density=True)
    ax2.set_title(f'Attack Density Distribution - {title}', fontsize=14)
    ax2.set_xlabel('Activity Level', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}density_distribution.png")
        fig2.savefig(path, bbox_inches='tight')
        console.print(r"[bold blue]\[Discovery][/bold blue] Density distribution plot saved to: [bold green]" + path + r"[/bold green]")
    plt.show()
    plt.close(fig2)

    # 3. ECDF Comparison (Kolmogorov-Smirnov basis)
    def ecdf(data):
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return x, y

    x_real, y_real = ecdf(real)
    x_pred, y_pred = ecdf(predicted)

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.step(x_real, y_real, label='Real ECDF', color='blue', where='post')
    ax3.step(x_pred, y_pred, label='Predicted ECDF', color='orange', where='post')
    ax3.set_title(f'Cumulative Activity Distribution (ECDF) - {title}', fontsize=14)
    ax3.set_xlabel('Activity Level', fontsize=12)
    ax3.set_ylabel('Cumulative Probability F(x)', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}ecdf.png")
        fig3.savefig(path, bbox_inches='tight')
        console.print(r"[bold blue]\[Discovery][/bold blue] ECDF plot saved to: [bold green]" + path + r"[/bold green]")
    plt.show()
    plt.close(fig3)

    # 4. Residuals Plot (Spatial Error)
    residuals = real - predicted
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.scatter(x_rank, residuals, alpha=0.5, color='purple', s=10)
    ax4.axhline(0, color='black', linestyle='--')
    ax4.set_title(f'Residuals (Real - Predicted Intensity) - {title}', fontsize=14)
    ax4.set_xlabel('Site Index', fontsize=12)
    ax4.set_ylabel('Intensity Error', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}residuals.png")
        fig4.savefig(path, bbox_inches='tight')
        console.print(r"[bold blue]\[Discovery][/bold blue] Residuals plot saved to: [bold green]" + path + r"[/bold green]")
    plt.show()
    plt.close(fig4)

def plot_heatmaps(real_v, predicted_v, Zj, title="Spatial Distribution Comparison", output_dir=None, optimizer_name=None):
    """
    Creates side-by-side heatmaps for real vs predicted data with interpolation and smoothing.
    """
    nlat, nlon = 78, 71
    
    def map_to_grid(values, Zj_coords):
        grid = np.zeros((nlat, nlon))
        lat_min, lat_max = Zj_coords[:, 0].min(), Zj_coords[:, 0].max()
        lon_min, lon_max = Zj_coords[:, 1].min(), Zj_coords[:, 1].max()
        lat_unique = np.linspace(lat_min, lat_max, nlat)
        lon_unique = np.linspace(lon_min, lon_max, nlon)
        
        for i in range(len(values)):
            lat_idx = np.abs(lat_unique - Zj_coords[i, 0]).argmin()
            lon_idx = np.abs(lon_unique - Zj_coords[i, 1]).argmin()
            grid[lat_idx, lon_idx] += values[i]  # FIXED: #4 — accumulate; last-write-wins silently drops colliding zones
        return grid

    real_grid = map_to_grid(real_v, Zj)
    pred_grid = map_to_grid(predicted_v, Zj)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=20, y=1.05)

    datasets = [("Predicted Data (Simulated)", pred_grid), ("Real Data (SOSAFE)", real_grid)]

    for i, (d_title, data) in enumerate(datasets):
        ax = axes[i]
        rows, cols = data.shape
        x = np.arange(cols)
        y = np.arange(rows)
        
        # Interpolation to double resolution
        f = RegularGridInterpolator((y, x), data)
        x_new = np.linspace(0, cols - 1, cols * 2)
        y_new = np.linspace(0, rows - 1, rows * 2)
        xx, yy = np.meshgrid(x_new, y_new)
        pts = np.array([yy.flatten(), xx.flatten()]).T
        z = f(pts).reshape(xx.shape)
        
        # Gaussian smoothing
        z_smooth = gaussian_filter(z, sigma=1.0)  # FIXED: #5 — sigma=0.05 was sub-pixel (no-op); 1.0 gives visible smoothing
        
        im = ax.imshow(z_smooth, cmap='magma_r', extent=[0, cols, 0, rows], origin='lower')
        ax.set_title(d_title, fontsize=14)
        ax.set_xlabel('Longitude Index', fontsize=12)
        ax.set_ylabel('Latitude Index', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.colorbar(im, ax=ax, shrink=0.7, label='Intensity')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # FIXED: #6 — reserve top 5% for suptitle so it isn't clipped
    prefix = f"{optimizer_name}_" if optimizer_name else ""
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}spatial_heatmap.png")
        plt.savefig(path, bbox_inches='tight')
        console.print(r"[bold blue]\[Discovery][/bold blue] Heatmap saved to: [bold green]" + path + r"[/bold green]")
    plt.show()
