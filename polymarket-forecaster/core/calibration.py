"""
Calibration Layer - Statistical calibration for LLM probability outputs.

Implements techniques from:
- AIA Forecaster: Platt scaling for LLM hedging/overconfidence correction
- Outcome-RL paper: ECE optimization achieving 0.042

Key insight: LLMs tend to be overconfident or hedge toward 50%.
Calibration learns a correction function from historical accuracy.
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import json

from config.settings import CONFIG_DIR


@dataclass
class CalibrationResult:
    """Result of applying calibration."""
    raw_probability: float
    calibrated_probability: float
    method: str
    confidence: float  # How confident we are in the calibration


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    accuracy: float
    n_samples: int


class PlattScaling:
    """
    Platt scaling: Learn sigmoid correction from historical forecasts.

    calibrated = 1 / (1 + exp(A * raw + B))

    Parameters A and B are learned from (prediction, outcome) pairs.
    This addresses LLM overconfidence by learning a smooth correction.
    """

    def __init__(self, A: float = 0.0, B: float = 0.0):
        """
        Initialize with optional pre-learned parameters.

        Args:
            A: Slope parameter (learned from data)
            B: Intercept parameter (learned from data)
        """
        self.A = A
        self.B = B
        self._fitted = A != 0.0 or B != 0.0

    def fit(self, predictions: List[float], outcomes: List[int],
            max_iter: int = 100, lr: float = 0.1) -> Tuple[float, float]:
        """
        Learn Platt scaling parameters from historical data.

        Uses gradient descent to minimize log loss.

        Args:
            predictions: Raw probability predictions (0-1)
            outcomes: Actual outcomes (0 or 1)
            max_iter: Maximum iterations
            lr: Learning rate

        Returns:
            Tuple of (A, B) parameters
        """
        if len(predictions) < 10:
            # Not enough data, use identity transform
            self.A, self.B = 0.0, 0.0
            return (0.0, 0.0)

        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        # Initialize parameters
        A, B = 0.0, 0.0

        for _ in range(max_iter):
            # Forward pass
            logits = A * predictions + B
            calibrated = 1 / (1 + np.exp(-logits))

            # Compute gradients (log loss)
            error = calibrated - outcomes
            grad_A = np.mean(error * predictions)
            grad_B = np.mean(error)

            # Update
            A -= lr * grad_A
            B -= lr * grad_B

        self.A = float(A)
        self.B = float(B)
        self._fitted = True

        return (self.A, self.B)

    def calibrate(self, raw_prob: float) -> float:
        """
        Apply Platt scaling to a raw probability.

        Args:
            raw_prob: Raw probability (0-1)

        Returns:
            Calibrated probability
        """
        if not self._fitted:
            return raw_prob

        logit = self.A * raw_prob + self.B
        calibrated = 1 / (1 + math.exp(-logit))

        # Clamp to valid range
        return max(0.001, min(0.999, calibrated))

    def save(self, filepath: Optional[Path] = None):
        """Save parameters to file."""
        filepath = filepath or CONFIG_DIR / "platt_params.json"
        with open(filepath, 'w') as f:
            json.dump({"A": self.A, "B": self.B}, f)

    def load(self, filepath: Optional[Path] = None):
        """Load parameters from file."""
        filepath = filepath or CONFIG_DIR / "platt_params.json"
        if filepath.exists():
            with open(filepath) as f:
                params = json.load(f)
                self.A = params["A"]
                self.B = params["B"]
                self._fitted = True


class IsotonicCalibration:
    """
    Isotonic regression calibration.

    Non-parametric approach that learns a monotonic mapping.
    More flexible than Platt scaling but requires more data.
    """

    def __init__(self):
        self._bins: List[Tuple[float, float, float]] = []  # (lower, upper, calibrated)
        self._fitted = False

    def fit(self, predictions: List[float], outcomes: List[int],
            n_bins: int = 10) -> None:
        """
        Fit isotonic calibration from historical data.

        Args:
            predictions: Raw probability predictions
            outcomes: Actual outcomes
            n_bins: Number of calibration bins
        """
        if len(predictions) < n_bins * 5:
            return

        # Sort by prediction
        sorted_pairs = sorted(zip(predictions, outcomes))
        predictions = [p for p, _ in sorted_pairs]
        outcomes = [o for _, o in sorted_pairs]

        # Create bins
        bin_size = len(predictions) // n_bins
        self._bins = []

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(predictions)

            bin_preds = predictions[start:end]
            bin_outcomes = outcomes[start:end]

            lower = min(bin_preds)
            upper = max(bin_preds)
            calibrated = sum(bin_outcomes) / len(bin_outcomes)

            self._bins.append((lower, upper, calibrated))

        # Ensure monotonicity (pool adjacent violators)
        self._enforce_monotonicity()
        self._fitted = True

    def _enforce_monotonicity(self):
        """Pool adjacent violators to ensure monotonic mapping."""
        i = 0
        while i < len(self._bins) - 1:
            if self._bins[i][2] > self._bins[i + 1][2]:
                # Pool these bins
                lower = self._bins[i][0]
                upper = self._bins[i + 1][1]
                avg = (self._bins[i][2] + self._bins[i + 1][2]) / 2
                self._bins[i] = (lower, upper, avg)
                self._bins.pop(i + 1)
            else:
                i += 1

    def calibrate(self, raw_prob: float) -> float:
        """Apply isotonic calibration."""
        if not self._fitted:
            return raw_prob

        for lower, upper, calibrated in self._bins:
            if lower <= raw_prob <= upper:
                return calibrated

        # Extrapolate
        if raw_prob < self._bins[0][0]:
            return self._bins[0][2]
        return self._bins[-1][2]


class TemperatureScaling:
    """
    Temperature scaling: Simple single-parameter calibration.

    calibrated_logit = logit / temperature

    Temperature > 1 reduces confidence (spreads probabilities toward 0.5)
    Temperature < 1 increases confidence (pushes toward 0 or 1)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._fitted = temperature != 1.0

    def fit(self, predictions: List[float], outcomes: List[int],
            max_iter: int = 100) -> float:
        """Learn optimal temperature from data."""
        if len(predictions) < 10:
            return 1.0

        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        # Grid search for optimal temperature
        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.linspace(0.5, 2.0, 31):
            # Apply temperature
            logits = np.log(predictions / (1 - predictions + 1e-10))
            scaled_logits = logits / temp
            calibrated = 1 / (1 + np.exp(-scaled_logits))

            # Compute ECE
            ece = self._compute_ece(calibrated.tolist(), outcomes.tolist())

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        self._fitted = True
        return best_temp

    def _compute_ece(self, predictions: List[float], outcomes: List[int],
                     n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bins = [[] for _ in range(n_bins)]

        for pred, outcome in zip(predictions, outcomes):
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append((pred, outcome))

        ece = 0.0
        total = len(predictions)

        for bin_data in bins:
            if bin_data:
                avg_pred = sum(p for p, _ in bin_data) / len(bin_data)
                avg_outcome = sum(o for _, o in bin_data) / len(bin_data)
                ece += len(bin_data) / total * abs(avg_pred - avg_outcome)

        return ece

    def calibrate(self, raw_prob: float) -> float:
        """Apply temperature scaling."""
        if not self._fitted or self.temperature == 1.0:
            return raw_prob

        # Convert to logit, scale, convert back
        raw_prob = max(0.001, min(0.999, raw_prob))
        logit = math.log(raw_prob / (1 - raw_prob))
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + math.exp(-scaled_logit))

        return max(0.001, min(0.999, calibrated))


class EnsembleCalibrator:
    """
    Ensemble calibration combining multiple methods.

    From Outcome-RL paper: median of ensemble predictions reduces variance.
    """

    def __init__(self):
        self.platt = PlattScaling()
        self.isotonic = IsotonicCalibration()
        self.temperature = TemperatureScaling()
        self._fitted = False

    def fit(self, predictions: List[float], outcomes: List[int]):
        """Fit all calibration methods."""
        self.platt.fit(predictions, outcomes)
        self.isotonic.fit(predictions, outcomes)
        self.temperature.fit(predictions, outcomes)
        self._fitted = True

    def calibrate(self, raw_prob: float, method: str = "ensemble") -> CalibrationResult:
        """
        Apply calibration.

        Args:
            raw_prob: Raw probability
            method: "platt", "isotonic", "temperature", or "ensemble"

        Returns:
            CalibrationResult with calibrated probability
        """
        if method == "platt":
            calibrated = self.platt.calibrate(raw_prob)
        elif method == "isotonic":
            calibrated = self.isotonic.calibrate(raw_prob)
        elif method == "temperature":
            calibrated = self.temperature.calibrate(raw_prob)
        elif method == "ensemble":
            # Median of all methods (from Outcome-RL paper)
            values = [
                self.platt.calibrate(raw_prob),
                self.isotonic.calibrate(raw_prob),
                self.temperature.calibrate(raw_prob)
            ]
            calibrated = float(np.median(values))
        else:
            calibrated = raw_prob

        return CalibrationResult(
            raw_probability=raw_prob,
            calibrated_probability=calibrated,
            method=method,
            confidence=0.8 if self._fitted else 0.3
        )


def compute_ece(predictions: List[float], outcomes: List[int],
                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = sum(|accuracy(bin) - confidence(bin)| * count(bin) / total)

    From Outcome-RL paper: They achieve ECE of 0.042, beating o1.

    Args:
        predictions: Probability predictions
        outcomes: Actual outcomes (0 or 1)
        n_bins: Number of calibration bins

    Returns:
        ECE value (lower is better, 0 is perfect calibration)
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


def compute_mce(predictions: List[float], outcomes: List[int],
                n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error.

    MCE = max(|accuracy(bin) - confidence(bin)|) across all bins
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


def compute_calibration_metrics(predictions: List[float],
                                 outcomes: List[int]) -> CalibrationMetrics:
    """Compute all calibration metrics."""
    n = len(predictions)
    if n == 0:
        return CalibrationMetrics(0, 0, 0, 0, 0)

    # Brier score
    brier = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n

    # Accuracy (using 0.5 threshold)
    correct = sum(1 for p, o in zip(predictions, outcomes)
                  if (p >= 0.5) == (o == 1))
    accuracy = correct / n

    return CalibrationMetrics(
        ece=compute_ece(predictions, outcomes),
        mce=compute_mce(predictions, outcomes),
        brier_score=brier,
        accuracy=accuracy,
        n_samples=n
    )


# Default calibrator instance
_default_calibrator: Optional[EnsembleCalibrator] = None


def get_calibrator() -> EnsembleCalibrator:
    """Get or create the default calibrator."""
    global _default_calibrator
    if _default_calibrator is None:
        _default_calibrator = EnsembleCalibrator()
        # Try to load pre-trained parameters
        try:
            _default_calibrator.platt.load()
        except:
            pass
    return _default_calibrator


def calibrate(raw_prob: float, method: str = "ensemble") -> float:
    """Convenience function to calibrate a probability."""
    calibrator = get_calibrator()
    result = calibrator.calibrate(raw_prob, method)
    return result.calibrated_probability
