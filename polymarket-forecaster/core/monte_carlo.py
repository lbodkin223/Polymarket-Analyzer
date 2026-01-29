"""
Monte Carlo Probability Engine - Core simulation logic for probability forecasting.

This module converts evidence effects to probability multipliers and runs
Monte Carlo simulations to generate probability estimates with confidence intervals.
"""
import math
from typing import List, Dict, Optional, Tuple
import numpy as np

from . import Evidence, MCResult
from config.settings import (
    DEFAULT_SIMULATIONS,
    MULTIPLIER_VARIANCE,
    EVIDENCE_TYPE_WEIGHTS,
    MIN_PROBABILITY,
    MAX_PROBABILITY
)


def logit(p: float) -> float:
    """Convert probability to logit (log-odds) space."""
    p = max(MIN_PROBABILITY, min(MAX_PROBABILITY, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    """Convert logit back to probability."""
    return 1 / (1 + math.exp(-x))


def apply_evidence_effects(evidence_items: List[Evidence]) -> Dict[str, float]:
    """
    Convert evidence items to probability multipliers.

    Evidence supporting YES with high strength -> multiplier > 1.0
    Evidence supporting NO with high strength -> multiplier < 1.0

    The multiplier is calculated by:
    1. Determining direction (+1 for YES, -1 for NO, 0 for NEUTRAL)
    2. Weighting by evidence type (A=1.0, B=0.8, C=0.5, D=0.2)
    3. Converting the effect to a multiplier range (0.25 to 2.0)

    Args:
        evidence_items: List of Evidence objects

    Returns:
        Dict mapping source names to multipliers
    """
    multipliers = {}

    for evidence in evidence_items:
        # Determine direction
        if evidence.supports == "YES":
            direction = 1.0
        elif evidence.supports == "NO":
            direction = -1.0
        else:
            direction = 0.0

        # Get type weight
        type_weight = EVIDENCE_TYPE_WEIGHTS.get(evidence.evidence_type, 0.3)

        # Calculate effect: direction * strength * type_weight
        # This gives a value in range [-1.0, +1.0]
        effect = direction * evidence.strength * type_weight

        # Convert effect to multiplier
        # effect = -1.0 -> multiplier = 0.25 (strong NO evidence)
        # effect = 0.0 -> multiplier = 1.0 (neutral)
        # effect = +1.0 -> multiplier = 2.0 (strong YES evidence)
        # Using: multiplier = 0.25 + (effect + 1.0) * 0.875
        multiplier = 0.25 + (effect + 1.0) * 0.875

        # Clamp to reasonable range
        multipliers[evidence.source] = max(0.1, min(10.0, multiplier))

    return multipliers


def calculate_combined_multiplier(multipliers: Dict[str, float]) -> float:
    """
    Combine multiple evidence multipliers into a single aggregate multiplier.

    Uses geometric mean to prevent any single piece of evidence
    from dominating the result.

    Args:
        multipliers: Dict of source -> multiplier

    Returns:
        Combined multiplier
    """
    if not multipliers:
        return 1.0

    values = list(multipliers.values())

    # Geometric mean: (x1 * x2 * ... * xn) ^ (1/n)
    product = 1.0
    for v in values:
        product *= v

    geometric_mean = product ** (1 / len(values))

    return geometric_mean


def apply_multiplier_to_probability(
    baseline: float,
    multiplier: float,
    in_logit_space: bool = True
) -> float:
    """
    Apply a multiplier to a baseline probability.

    Args:
        baseline: Starting probability (0-1)
        multiplier: Evidence multiplier
        in_logit_space: If True, apply in logit space for better behavior

    Returns:
        Updated probability
    """
    if in_logit_space:
        # Convert to logit, apply log(multiplier) as additive adjustment
        base_logit = logit(baseline)
        adjusted_logit = base_logit + math.log(multiplier)
        result = sigmoid(adjusted_logit)
    else:
        # Direct multiplication (simpler but less well-behaved at extremes)
        result = baseline * multiplier

    # Clamp to valid range
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, result))


def run_monte_carlo(
    baseline: float,
    multipliers: Dict[str, float],
    n_sims: int = DEFAULT_SIMULATIONS,
    variance: float = MULTIPLIER_VARIANCE,
    seed: Optional[int] = None
) -> MCResult:
    """
    Run Monte Carlo simulation to estimate probability with uncertainty.

    Algorithm:
    1. Start with baseline probability in logit space
    2. For each simulation:
       - Apply each multiplier with random variation (±variance)
       - Convert back to probability space
    3. Return median and confidence intervals

    Args:
        baseline: Starting probability (0-1)
        multipliers: Dict of source -> multiplier
        n_sims: Number of simulations to run
        variance: Random variation fraction for multipliers (e.g., 0.2 = ±20%)
        seed: Random seed for reproducibility

    Returns:
        MCResult with probability estimates and confidence intervals
    """
    if seed is not None:
        np.random.seed(seed)

    if not multipliers:
        # No evidence, return baseline with uncertainty
        return MCResult(
            median_probability=baseline,
            mean_probability=baseline,
            ci_lower=max(MIN_PROBABILITY, baseline - 0.1),
            ci_upper=min(MAX_PROBABILITY, baseline + 0.1),
            simulations=n_sims
        )

    # Run simulations
    probabilities = np.zeros(n_sims)
    base_logit = logit(baseline)

    for i in range(n_sims):
        logit_adjustment = 0.0

        for source, multiplier in multipliers.items():
            # Add random variation to multiplier
            # Variation is uniform in range [1-variance, 1+variance]
            variation = 1.0 + np.random.uniform(-variance, variance)
            varied_multiplier = multiplier * variation

            # Clamp varied multiplier
            varied_multiplier = max(0.1, min(10.0, varied_multiplier))

            # Add log(multiplier) to logit adjustment
            logit_adjustment += math.log(varied_multiplier)

        # Apply adjustment and convert back
        final_logit = base_logit + logit_adjustment
        probabilities[i] = sigmoid(final_logit)

    # Clamp all probabilities
    probabilities = np.clip(probabilities, MIN_PROBABILITY, MAX_PROBABILITY)

    # Calculate statistics
    median = float(np.median(probabilities))
    mean = float(np.mean(probabilities))
    ci_lower = float(np.percentile(probabilities, 5))
    ci_upper = float(np.percentile(probabilities, 95))

    return MCResult(
        median_probability=median,
        mean_probability=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        simulations=n_sims,
        all_probabilities=probabilities.tolist() if n_sims <= 1000 else None
    )


def run_sensitivity_analysis(
    baseline: float,
    multipliers: Dict[str, float],
    n_sims: int = 1000
) -> Dict[str, float]:
    """
    Analyze which evidence sources have the most impact on the result.

    For each source, calculates the change in median probability
    when that source is removed.

    Args:
        baseline: Starting probability
        multipliers: Dict of source -> multiplier
        n_sims: Simulations per analysis

    Returns:
        Dict mapping source to impact score (absolute probability change)
    """
    if not multipliers:
        return {}

    # Get baseline result with all evidence
    full_result = run_monte_carlo(baseline, multipliers, n_sims)
    baseline_median = full_result.median_probability

    impacts = {}

    for source in multipliers:
        # Remove this source
        reduced_multipliers = {k: v for k, v in multipliers.items() if k != source}

        # Run simulation without this source
        reduced_result = run_monte_carlo(baseline, reduced_multipliers, n_sims)

        # Impact is the change in median probability
        impact = abs(baseline_median - reduced_result.median_probability)
        impacts[source] = impact

    # Sort by impact (highest first)
    return dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))


def simulate_with_scenarios(
    baseline: float,
    evidence_items: List[Evidence],
    scenarios: Optional[Dict[str, float]] = None,
    n_sims: int = DEFAULT_SIMULATIONS
) -> Dict[str, MCResult]:
    """
    Run simulations under different scenarios.

    Args:
        baseline: Starting probability
        evidence_items: List of Evidence objects
        scenarios: Dict of scenario_name -> baseline_adjustment
                   e.g., {"optimistic": 0.1, "pessimistic": -0.1}
        n_sims: Simulations per scenario

    Returns:
        Dict mapping scenario name to MCResult
    """
    multipliers = apply_evidence_effects(evidence_items)

    if scenarios is None:
        scenarios = {
            "base": 0.0,
            "optimistic": 0.1,
            "pessimistic": -0.1
        }

    results = {}

    for scenario_name, adjustment in scenarios.items():
        adjusted_baseline = max(MIN_PROBABILITY, min(MAX_PROBABILITY, baseline + adjustment))
        result = run_monte_carlo(adjusted_baseline, multipliers, n_sims)
        results[scenario_name] = result

    return results


def forecast_with_evidence(
    baseline: float,
    evidence_items: List[Evidence],
    n_sims: int = DEFAULT_SIMULATIONS
) -> Tuple[MCResult, Dict[str, float]]:
    """
    Complete forecasting pipeline: evidence -> multipliers -> simulation.

    Args:
        baseline: Starting probability
        evidence_items: List of Evidence objects
        n_sims: Number of simulations

    Returns:
        Tuple of (MCResult, multipliers dict)
    """
    multipliers = apply_evidence_effects(evidence_items)
    mc_result = run_monte_carlo(baseline, multipliers, n_sims)

    return mc_result, multipliers
