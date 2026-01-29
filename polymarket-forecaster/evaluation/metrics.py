"""
Evaluation Metrics - Standard forecasting performance metrics.

Implements metrics used by:
- ForecastBench: Brier score (difficulty-adjusted)
- Outcome-RL paper: ECE of 0.042 (beats o1)
- AIA Forecaster: Expert-level benchmarking

Primary Metrics:
- Brier Score: Mean squared error of probability predictions
- ECE: Expected Calibration Error
- Log Loss: Cross-entropy loss
"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ForecastMetrics:
    """Complete set of forecasting metrics."""
    brier_score: float  # Lower is better (0 = perfect)
    ece: float  # Expected Calibration Error (lower is better)
    mce: float  # Maximum Calibration Error
    log_loss: float  # Cross-entropy loss
    accuracy: float  # Binary accuracy at 0.5 threshold
    n_samples: int
    mean_confidence: float
    calibration_slope: float  # 1.0 = perfect calibration


def brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate Brier Score.

    Brier Score = (1/n) * Σ(prediction - outcome)²

    This is the primary metric used by ForecastBench.
    Range: 0 (perfect) to 1 (worst)

    Reference values:
    - Random guessing (always 0.5): 0.25
    - Outcome-RL paper achieves: 0.193
    - Perfect forecaster: 0.0

    Args:
        predictions: Probability predictions (0-1)
        outcomes: Actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    if len(predictions) == 0:
        return 0.0

    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")

    squared_errors = [(p - o) ** 2 for p, o in zip(predictions, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def expected_calibration_error(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE = Σ (|bin_accuracy - bin_confidence| * bin_size / total_size)

    This measures how well-calibrated the predictions are.
    A prediction of 70% should resolve YES 70% of the time.

    Reference values:
    - Outcome-RL paper achieves: 0.042 (beats o1 which has higher ECE)
    - Perfect calibration: 0.0
    - Typical LLM: 0.10-0.20

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes
        n_bins: Number of calibration bins

    Returns:
        ECE value
    """
    if len(predictions) == 0:
        return 0.0

    bins = [[] for _ in range(n_bins)]

    for pred, outcome in zip(predictions, outcomes):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, outcome))

    ece = 0.0
    total = len(predictions)

    for bin_data in bins:
        if bin_data:
            avg_confidence = sum(p for p, _ in bin_data) / len(bin_data)
            accuracy = sum(o for _, o in bin_data) / len(bin_data)
            ece += len(bin_data) / total * abs(avg_confidence - accuracy)

    return ece


def maximum_calibration_error(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE = max(|bin_accuracy - bin_confidence|) across all bins

    This is the worst-case calibration error.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes
        n_bins: Number of calibration bins

    Returns:
        MCE value
    """
    if len(predictions) == 0:
        return 0.0

    bins = [[] for _ in range(n_bins)]

    for pred, outcome in zip(predictions, outcomes):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, outcome))

    max_error = 0.0

    for bin_data in bins:
        if bin_data:
            avg_confidence = sum(p for p, _ in bin_data) / len(bin_data)
            accuracy = sum(o for _, o in bin_data) / len(bin_data)
            max_error = max(max_error, abs(avg_confidence - accuracy))

    return max_error


def log_loss(predictions: List[float], outcomes: List[int],
             eps: float = 1e-15) -> float:
    """
    Calculate Log Loss (Cross-Entropy Loss).

    Log Loss = -(1/n) * Σ(outcome * log(pred) + (1-outcome) * log(1-pred))

    This penalizes confident wrong predictions more heavily.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes
        eps: Small value to avoid log(0)

    Returns:
        Log loss value
    """
    if len(predictions) == 0:
        return 0.0

    total = 0.0
    for pred, outcome in zip(predictions, outcomes):
        # Clamp to avoid log(0)
        pred = max(eps, min(1 - eps, pred))

        if outcome == 1:
            total += -math.log(pred)
        else:
            total += -math.log(1 - pred)

    return total / len(predictions)


def accuracy(predictions: List[float], outcomes: List[int],
             threshold: float = 0.5) -> float:
    """
    Calculate binary accuracy.

    Accuracy = correct_predictions / total_predictions

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes
        threshold: Classification threshold (default 0.5)

    Returns:
        Accuracy (0-1)
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, o in zip(predictions, outcomes)
                  if (p >= threshold) == (o == 1))

    return correct / len(predictions)


def reliability_diagram_data(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Generate data for reliability diagram.

    A reliability diagram plots predicted probability vs observed frequency.
    Perfect calibration = diagonal line.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes
        n_bins: Number of bins

    Returns:
        Dict with 'bin_centers', 'accuracies', 'counts'
    """
    bins = [[] for _ in range(n_bins)]

    for pred, outcome in zip(predictions, outcomes):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, outcome))

    bin_centers = []
    accuracies = []
    counts = []

    for i, bin_data in enumerate(bins):
        center = (i + 0.5) / n_bins
        bin_centers.append(center)

        if bin_data:
            acc = sum(o for _, o in bin_data) / len(bin_data)
            accuracies.append(acc)
            counts.append(len(bin_data))
        else:
            accuracies.append(0.0)
            counts.append(0)

    return {
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "counts": counts
    }


def calibration_slope(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate calibration slope via linear regression.

    Slope of 1.0 indicates perfect calibration.
    Slope < 1.0 indicates overconfidence.
    Slope > 1.0 indicates underconfidence.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes

    Returns:
        Calibration slope
    """
    if len(predictions) < 2:
        return 1.0

    x = np.array(predictions)
    y = np.array(outcomes)

    # Simple linear regression: y = slope * x + intercept
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 1.0

    slope = numerator / denominator
    return float(slope)


def compute_all_metrics(predictions: List[float],
                         outcomes: List[int]) -> ForecastMetrics:
    """
    Compute all forecasting metrics.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes

    Returns:
        ForecastMetrics with all computed values
    """
    n = len(predictions)

    if n == 0:
        return ForecastMetrics(
            brier_score=0.0,
            ece=0.0,
            mce=0.0,
            log_loss=0.0,
            accuracy=0.0,
            n_samples=0,
            mean_confidence=0.0,
            calibration_slope=1.0
        )

    return ForecastMetrics(
        brier_score=brier_score(predictions, outcomes),
        ece=expected_calibration_error(predictions, outcomes),
        mce=maximum_calibration_error(predictions, outcomes),
        log_loss=log_loss(predictions, outcomes),
        accuracy=accuracy(predictions, outcomes),
        n_samples=n,
        mean_confidence=sum(predictions) / n,
        calibration_slope=calibration_slope(predictions, outcomes)
    )


def difficulty_adjusted_brier(
    predictions: List[float],
    outcomes: List[int],
    baseline_predictions: Optional[List[float]] = None
) -> float:
    """
    Calculate difficulty-adjusted Brier score.

    From ForecastBench: "Enables fair comparison across different
    question subsets by accounting for inherent difficulty."

    Difficulty adjustment = Brier / baseline_Brier

    Args:
        predictions: Model predictions
        outcomes: Actual outcomes
        baseline_predictions: Baseline model predictions (default: market prices)

    Returns:
        Difficulty-adjusted Brier score
    """
    model_brier = brier_score(predictions, outcomes)

    if baseline_predictions is None:
        # Use 0.5 as baseline (uninformative prior)
        baseline_predictions = [0.5] * len(predictions)

    baseline_brier = brier_score(baseline_predictions, outcomes)

    if baseline_brier == 0:
        return model_brier

    # Skill score: improvement over baseline
    return model_brier / baseline_brier


def roi_simulation(
    predictions: List[float],
    outcomes: List[int],
    market_prices: List[float],
    bankroll: float = 10000,
    max_bet_fraction: float = 0.25,
    min_edge: float = 0.05
) -> Dict[str, float]:
    """
    Simulate ROI from trading on predictions.

    From Outcome-RL paper: ">10% ROI in Polymarket simulation"

    Args:
        predictions: AI probability predictions
        outcomes: Actual outcomes
        market_prices: Market prices at time of prediction
        bankroll: Starting bankroll
        max_bet_fraction: Maximum bet as fraction of bankroll
        min_edge: Minimum edge to place bet

    Returns:
        Dict with ROI metrics
    """
    current_bankroll = bankroll
    total_bets = 0
    winning_bets = 0
    total_wagered = 0

    for pred, outcome, market_price in zip(predictions, outcomes, market_prices):
        # Calculate edge
        yes_edge = pred - market_price
        no_edge = (1 - pred) - (1 - market_price)

        if abs(yes_edge) >= min_edge:
            if yes_edge > 0:
                side = "YES"
                edge = yes_edge
                bet_price = market_price
            else:
                side = "NO"
                edge = no_edge
                bet_price = 1 - market_price

            # Kelly sizing (capped)
            if bet_price > 0 and bet_price < 1:
                p = pred if side == "YES" else (1 - pred)
                b = (1 - bet_price) / bet_price
                kelly = (b * p - (1 - p)) / b
                kelly = max(0, min(max_bet_fraction, kelly))

                bet_amount = current_bankroll * kelly
                total_bets += 1
                total_wagered += bet_amount

                # Resolve bet
                won = (side == "YES" and outcome == 1) or (side == "NO" and outcome == 0)

                if won:
                    winning_bets += 1
                    profit = bet_amount * (1 - bet_price) / bet_price
                    current_bankroll += profit
                else:
                    current_bankroll -= bet_amount

    final_pnl = current_bankroll - bankroll
    roi = final_pnl / bankroll if bankroll > 0 else 0

    return {
        "final_bankroll": current_bankroll,
        "total_pnl": final_pnl,
        "roi": roi,
        "roi_percent": roi * 100,
        "total_bets": total_bets,
        "winning_bets": winning_bets,
        "win_rate": winning_bets / total_bets if total_bets > 0 else 0,
        "total_wagered": total_wagered
    }
