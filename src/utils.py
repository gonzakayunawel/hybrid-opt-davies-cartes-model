import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp

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
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "max_error": np.max(np.abs(real - predicted)),
        "ks_stat": ks_stat,
        "ks_p_value": ks_p_value
    }

    return metrics

def print_metrics(predicted, real):
    """
    Prints detailed error metrics.
    """
    metrics = calculate_errors(predicted, real)

    print("Error Metrics:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"SMAPE: {metrics['smape']:.2f}%")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Max Error: {metrics['max_error']:.4f}")
    print(f"KS Stat: {metrics['ks_stat']:.4f}, p-value: {metrics['ks_p_value']:.4f}")

def plot_results(real, predicted, title="Comparison"):
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
    plt.show()
