import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

def scale_data(data):
    """
    Scales data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return scaled_data, scaler

def process_simulation_output(Rj_t, method='linear', prominence=0.05, lowess_frac=0.1):
    """
    Processes the simulation output (Rj_t) by finding peaks and interpolating/smoothing.

    Args:
        Rj_t (np.ndarray): The simulation output array.
        method (str): 'linear' or 'lowess'.
        prominence (float): Prominence for peak detection.
        lowess_frac (float): Fraction for LOWESS smoothing.

    Returns:
        np.ndarray: The processed array.
    """

    x = np.arange(len(Rj_t))
    y = Rj_t

    # Normalize for peak detection stability if needed, or just use raw if scaled?
    # In DE notebook, Rj_t is scaled first then peaks found.
    # In PyTorch notebook, y is min-max normalized manually then peaks found.

    # Let's handle scaling inside or outside?
    # Usually it's better to work with what we have.
    # If we assume input is raw Rj_t, we might want to normalize for peak detection.

    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) if (np.max(y) - np.min(y)) > 0 else y

    peaks, _ = find_peaks(y_norm, prominence=prominence)

    # Ensure start and end points are included
    if len(peaks) == 0 or peaks[0] != 0:
        peaks = np.insert(peaks, 0, 0)
    if peaks[-1] != len(y) - 1:
        peaks = np.append(peaks, len(y) - 1)

    x_peaks = x[peaks]
    y_peaks = y[peaks] # Use original values at peaks

    if method == 'linear':
        processed_curve = np.interp(x, x_peaks, y_peaks)
    elif method == 'lowess':
        lowess = sm.nonparametric.lowess(y_peaks, x_peaks, frac=lowess_frac)
        lowess_x = lowess[:, 0]
        lowess_y = lowess[:, 1]
        processed_curve = np.interp(x, lowess_x, lowess_y)
    else:
        raise ValueError(f"Unknown processing method: {method}")

    return processed_curve
