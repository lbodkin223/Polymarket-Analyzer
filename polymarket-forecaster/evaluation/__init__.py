"""
Evaluation module for forecasting performance.

Implements metrics compatible with:
- ForecastBench (forecastbench.org)
- PrediBench (predibench.com)
- Prediction Arena

Key metrics:
- Brier Score (primary accuracy metric)
- ECE (Expected Calibration Error)
- Log Loss
- ROI (Return on Investment simulation)
"""

from .metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    log_loss,
    accuracy,
    reliability_diagram_data,
    compute_all_metrics,
    ForecastMetrics
)

from .benchmark import (
    Benchmark,
    ForecastBenchEvaluator,
    BacktestEvaluator,
    run_backtest
)

__all__ = [
    'brier_score',
    'expected_calibration_error',
    'maximum_calibration_error',
    'log_loss',
    'accuracy',
    'reliability_diagram_data',
    'compute_all_metrics',
    'ForecastMetrics',
    'Benchmark',
    'ForecastBenchEvaluator',
    'BacktestEvaluator',
    'run_backtest',
]
