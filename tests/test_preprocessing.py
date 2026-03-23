import sys
from unittest.mock import MagicMock, patch

# Mocking modules that are missing in the environment before importing src.preprocessing
# Using sys.modules is necessary because we cannot import src.preprocessing otherwise
# if its dependencies are missing.
mock_np = MagicMock()
sys.modules['numpy'] = mock_np
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.api'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()

import pytest
from src.preprocessing import process_simulation_output

def test_process_simulation_output_invalid_method():
    """
    Test that process_simulation_output raises ValueError for unknown methods.
    """
    # Setup minimal mocks for the function preamble
    mock_np.min.return_value = 0
    mock_np.max.return_value = 1
    mock_np.arange.return_value = [0, 1, 2]

    from scipy.signal import find_peaks
    find_peaks.return_value = ([1], {}) # One peak

    # We use a mock for Rj_t as well
    Rj_t = MagicMock()
    Rj_t.__len__.return_value = 3

    invalid_method = 'unsupported_method'
    with pytest.raises(ValueError) as excinfo:
        process_simulation_output(Rj_t, method=invalid_method)

    assert f"Unknown processing method: {invalid_method}" in str(excinfo.value)
