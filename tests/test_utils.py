import sys
import math
from unittest.mock import MagicMock, patch

# Utility tests for src/utils.py

def test_calculate_errors_structure():
    """
    Verify the dictionary returned by calculate_errors contains all expected keys.
    """
    mock_np = MagicMock()
    mock_metrics = MagicMock()
    mock_stats = MagicMock()

    mock_metrics.mean_squared_error.return_value = 1.0
    mock_metrics.mean_absolute_error.return_value = 1.0
    mock_metrics.r2_score.return_value = 1.0
    mock_stats.ks_2samp.return_value = (0.5, 0.05)

    mock_np.any.return_value = True
    mock_np.mean.return_value = 10.0
    mock_np.max.return_value = 1.0
    mock_np.sqrt.side_effect = lambda x: math.sqrt(x)

    with patch.dict(sys.modules, {
        'numpy': mock_np,
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.metrics': mock_metrics,
        'scipy': MagicMock(),
        'scipy.stats': mock_stats
    }):
        if 'src.utils' in sys.modules:
            del sys.modules['src.utils']
        import src.utils
        from src.utils import calculate_errors

        predicted = MagicMock()
        real = MagicMock()
        metrics = calculate_errors(predicted, real)

        expected_keys = {"rmse", "mae", "mape", "smape", "r2", "max_error", "ks_stat", "ks_p_value"}
        assert set(metrics.keys()) == expected_keys

def test_calculate_errors_mape_zero_handling():
    """
    Verify that mape is np.nan when real values are all zero.
    """
    mock_np = MagicMock()
    mock_np.nan = float('nan')
    mock_np.any.return_value = False
    mock_stats = MagicMock()
    mock_stats.ks_2samp.return_value = (0.1, 0.9)

    with patch.dict(sys.modules, {
        'numpy': mock_np,
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.metrics': MagicMock(),
        'scipy': MagicMock(),
        'scipy.stats': mock_stats
    }):
        if 'src.utils' in sys.modules:
            del sys.modules['src.utils']
        import src.utils
        from src.utils import calculate_errors

        predicted = MagicMock()
        real = MagicMock()
        metrics = calculate_errors(predicted, real)
        assert math.isnan(metrics['mape'])

def test_print_metrics_output():
    """
    Verify that print_metrics calls calculate_errors and print.
    """
    with patch.dict(sys.modules, {
        'numpy': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.metrics': MagicMock(),
        'scipy': MagicMock(),
        'scipy.stats': MagicMock()
    }):
        if 'src.utils' in sys.modules:
            del sys.modules['src.utils']
        import src.utils
        from src.utils import print_metrics

        predicted = MagicMock()
        real = MagicMock()

        with patch('src.utils.calculate_errors') as mock_calc:
            mock_calc.return_value = {
                "rmse": 0.1, "mae": 0.1, "mape": 1.0, "smape": 1.0,
                "r2": 0.9, "max_error": 0.2, "ks_stat": 0.05, "ks_p_value": 0.99
            }
            with patch('builtins.print') as mock_print:
                print_metrics(predicted, real)
                mock_calc.assert_called_once_with(predicted, real)
                assert mock_print.call_count >= 8

def test_plot_results_calls():
    """
    Verify that plot_results calls matplotlib functions correctly.
    """
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

    # We must patch src.utils.plt AFTER the module is imported but during the test
    with patch.dict(sys.modules, {
        'numpy': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': mock_plt,
        'sklearn': MagicMock(),
        'sklearn.metrics': MagicMock(),
        'scipy': MagicMock(),
        'scipy.stats': MagicMock()
    }):
        if 'src.utils' in sys.modules:
            del sys.modules['src.utils']
        import src.utils
        # Inject the mock directly to ensure it is used
        src.utils.plt = mock_plt
        from src.utils import plot_results

        real = MagicMock()
        real.__len__.return_value = 3
        predicted = MagicMock()
        predicted.__len__.return_value = 3

        plot_results(real, predicted)

        mock_plt.subplots.assert_called_once()
        assert mock_ax1.plot.called
        assert mock_ax2.plot.called
        mock_plt.show.assert_called_once()
