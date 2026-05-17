import numpy as np
import time
import torch
import os
import sys
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import DaviesModel

console = Console()

def run_simulation_numpy(distances, Ii, Zj, beta_r, gamma_r, alpha_p, gamma_p, Nt=500, Ntt=10):
    """
    Davies Model Simulation implemented with NumPy (Original logic).
    """
    nlat, nlon = 78, 71
    nz = 500
    Lr, Lp = 6, 12
    eta, tau, Ptotal = 0.005, 0.75, 500.0
    dt = 0.1
    
    Rj = np.zeros(nz, dtype=np.float64)
    Ai = np.zeros((nlat, nlon), dtype=np.float64)
    Pj = np.zeros(nz, dtype=np.float64)
    fjdel = np.zeros((nz, Lr), dtype=np.float64)
    rho = np.ones((nlat, nlon), dtype=np.float64)
    Ddel = np.zeros((nz, Lp), dtype=np.float64)
    Dj = np.zeros(nz, dtype=np.float64)
    
    Ii_v = Ii.copy()
    
    auxij1 = np.exp(-beta_r * distances)
    target_values = Zj[:, 2]
    dij = 1.0 * target_values * auxij1 / np.max(target_values)
    
    counter = 0
    start_time = time.time()
    
    for nn in range(Nt):
        for mm in range(Ntt):
            fj = np.exp(-np.floor(gamma_r * Pj / (Rj + 1.0e-20)))
            Wij = fj * dij
            Wi = np.sum(Wij, axis=2)
            P_off = rho * Wi / (1.0 + Wi)
            
            idxr = counter % Lr
            fjdel[:, idxr] = fj
            dnm_r = Lr if counter >= Lr else counter + 1
            We_ij = np.sum(fjdel, axis=1) * dij / dnm_r
            
            auxw = Ai / (np.sum(We_ij, axis=2) + 1.0e-20)
            Sij = auxw[:, :, np.newaxis] * We_ij
            Rj = np.sum(np.sum(Sij, axis=1), axis=0)
            
            Dj[:] = target_values ** (alpha_p) * np.exp(gamma_p * Rj[:])
            idxp = counter % Lp
            Ddel[:, idxp] = Dj[:]
            dnm_p = Lp if counter >= Lp else counter + 1
            Dej = np.sum(Ddel, axis=1) / dnm_p
            
            Pj = Ptotal * Dej / (np.sum(Dej) + 1.0e-20)
            counter += 1
            
            fj_cap = 1.0 - np.exp(-np.floor(Pj / (Rj + 1.0e-20)))
            Ci = tau * np.sum(Sij * fj_cap, axis=2)
            
            Ai += dt * (eta * P_off * Ii_v - Ci)
            Ii_v += -dt * eta * P_off * Ii_v
            
    elapsed = time.time() - start_time
    return Rj, elapsed

def run_simulation_cupy(distances, Ii, Zj, beta_r, gamma_r, alpha_p, gamma_p, Nt=500, Ntt=10):
    """
    Davies Model Simulation implemented with CuPy (GPU acceleration).
    """
    try:
        import cupy as cp
    except ImportError:
        return None, None
        
    nlat, nlon = 78, 71
    nz = 500
    Lr, Lp = 6, 12
    eta, tau, Ptotal = 0.005, 0.75, 500.0
    dt = 0.1
    
    # Move to GPU
    distances_cp = cp.asarray(distances)
    Ii_cp = cp.asarray(Ii)
    Zj_cp = cp.asarray(Zj)
    
    Rj = cp.zeros(nz, dtype=cp.float64)
    Ai = cp.zeros((nlat, nlon), dtype=cp.float64)
    Pj = cp.zeros(nz, dtype=cp.float64)
    fjdel = cp.zeros((nz, Lr), dtype=cp.float64)
    rho = cp.ones((nlat, nlon), dtype=cp.float64)
    Ddel = cp.zeros((nz, Lp), dtype=cp.float64)
    Dj = cp.zeros(nz, dtype=cp.float64)
    
    Ii_v = Ii_cp.copy()
    
    auxij1 = cp.exp(-beta_r * distances_cp)
    target_values = Zj_cp[:, 2]
    dij = 1.0 * target_values * auxij1 / cp.max(target_values)
    
    counter = 0
    # Sync and time
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    
    for nn in range(Nt):
        for mm in range(Ntt):
            fj = cp.exp(-cp.floor(gamma_r * Pj / (Rj + 1.0e-20)))
            Wij = fj * dij
            Wi = cp.sum(Wij, axis=2)
            P_off = rho * Wi / (1.0 + Wi)
            
            idxr = counter % Lr
            fjdel[:, idxr] = fj
            dnm_r = Lr if counter >= Lr else counter + 1
            We_ij = cp.sum(fjdel, axis=1) * dij / dnm_r
            
            auxw = Ai / (cp.sum(We_ij, axis=2) + 1.0e-20)
            Sij = auxw[:, :, cp.newaxis] * We_ij
            Rj = cp.sum(cp.sum(Sij, axis=1), axis=0)
            
            Dj[:] = target_values ** (alpha_p) * cp.exp(gamma_p * Rj[:])
            idxp = counter % Lp
            Ddel[:, idxp] = Dj[:]
            dnm_p = Lp if counter >= Lp else counter + 1
            Dej = cp.sum(Ddel, axis=1) / dnm_p
            
            Pj = Ptotal * Dej / (cp.sum(Dej) + 1.0e-20)
            counter += 1
            
            fj_cap = 1.0 - cp.exp(-cp.floor(Pj / (Rj + 1.0e-20)))
            Ci = tau * cp.sum(Sij * fj_cap, axis=2)
            
            Ai += dt * (eta * P_off * Ii_v - Ci)
            Ii_v += -dt * eta * P_off * Ii_v
            
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time
    return Rj.get(), elapsed

def perform_benchmark(data_dir, params):
    """
    Executes a performance benchmark comparing NumPy, CuPy, and PyTorch,
    including numerical equivalence validation.
    """
    from src.main import load_data
    
    with console.status(r"[bold blue]\[Benchmark][/bold blue] Loading data..."):
        origin, destination, targets, distances = load_data(data_dir)
        Ii = origin + destination
    
    beta_r, gamma_r, alpha_p, gamma_p = params
    
    # 1. NumPy Benchmark (Ground Truth)
    with console.status("Running [bold yellow]NumPy (Ground Truth)[/bold yellow] simulation..."):
        Rj_np, time_np = run_simulation_numpy(distances, Ii, targets, beta_r, gamma_r, alpha_p, gamma_p)
    
    # 2. CuPy Benchmark
    with console.status("Running [bold green]CuPy[/bold green] simulation..."):
        Rj_cp, time_cp = run_simulation_cupy(distances, Ii, targets, beta_r, gamma_r, alpha_p, gamma_p)
    
    # 3. PyTorch Benchmark (CPU/GPU)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    with console.status(fr"Running [bold cyan]PyTorch ({device_name})[/bold cyan] simulation..."):
        model = DaviesModel(distances, Ii, targets)
        start_pt = time.time()
        Rj_pt = model.run_simulation(beta_r, gamma_r, alpha_p, gamma_p)
        time_pt = time.time() - start_pt
    
    # Validation Logic
    def get_errors(ground_truth, predicted):
        if predicted is None:
            return "N/A", "N/A"
        max_err = np.max(np.abs(ground_truth - predicted))
        rmse = np.sqrt(np.mean((ground_truth - predicted)**2))
        return f"{max_err:.2e}", f"{rmse:.2e}"

    cp_max, cp_rmse = get_errors(Rj_np, Rj_cp)
    pt_max, pt_rmse = get_errors(Rj_np, Rj_pt)

    # Report
    table = Table(title="[bold blue]Performance & Numerical Equivalence Benchmark[/bold blue]")
    table.add_column("Engine", style="cyan")
    table.add_column("Time (s)", style="magenta")
    table.add_column("Speedup", style="green")
    table.add_column("Max Error", style="red")
    table.add_column("RMSE", style="red")
    
    table.add_row("NumPy (CPU)", f"{time_np:.4f}", "1.00x", "0.00e+00", "0.00e+00")
    
    if time_cp:
        table.add_row("CuPy (GPU)", f"{time_cp:.4f}", f"{time_np/time_cp:.2f}x", cp_max, cp_rmse)
    else:
        table.add_row("CuPy (GPU)", "N/A", "N/A", "N/A", "N/A")
        
    table.add_row(f"PyTorch ({'GPU' if torch.cuda.is_available() else 'CPU'})", f"{time_pt:.4f}", f"{time_np/time_pt:.2f}x", pt_max, pt_rmse)
    
    console.print(table)
    
    if pt_rmse != "N/A" and float(pt_rmse) < 1e-4:
        console.print(r"[bold green]\[Discovery][/bold green] Numerical equivalence [bold green]VALIDATED[/bold green] (Errors within tolerance).")
    else:
        console.print(r"[bold red]\[Discovery][/bold red] Numerical equivalence [bold red]WARNING[/bold red] (Significant discrepancy detected).")
