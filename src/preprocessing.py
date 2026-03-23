import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks

class NumPyMinMaxScaler:
    """
    A lightweight replacement for sklearn's MinMaxScaler using NumPy.
    Provides fit, transform, and inverse_transform for 1D arrays.
    """
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.feature_range = feature_range

    def fit(self, data):
        self.data_min_ = np.min(data)
        self.data_max_ = np.max(data)
        data_range = self.data_max_ - self.data_min_

        if data_range == 0:
            self.scale_ = 0.0
        else:
            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range

        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, data):
        return data * self.scale_ + self.min_

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        if self.scale_ == 0:
            return np.full_like(data, self.data_min_)
        return (data - self.min_) / self.scale_

def scale_data(data):
    """
    Scales data using a lightweight NumPy implementation of MinMaxScaler.
    """
    scaler = NumPyMinMaxScaler()
    scaled_data = scaler.fit_transform(data)
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
