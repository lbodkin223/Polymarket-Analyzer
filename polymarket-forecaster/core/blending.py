"""
Probability Blending - Combine baseline priors with evidence-based projections.

This module provides methods to blend prior probabilities with
Monte Carlo projection results based on confidence levels.
"""
from typing import Optional, Dict
import json
from pathlib import Path

from config.settings import MIN_PROBABILITY, MAX_PROBABILITY, CONFIG_DIR


# Confidence-based blending weights
# Higher confidence = more weight on baseline (trust historical patterns)
# Lower confidence = more weight on projection (trust current evidence)
CONFIDENCE_WEIGHTS = {
    'high': 0.75,    # 75% baseline, 25% projection
    'medium': 0.50,  # 50/50 blend
    'low': 0.25,     # 25% baseline, 75% projection
}


def blend_probabilities(
    baseline: float,
    projection: float,
    confidence: str = "medium"
) -> float:
    """
    Blend baseline probability with evidence-based projection.

    The blend is deterministic based on confidence level:
    - High confidence: Trust baseline more (historical patterns reliable)
    - Low confidence: Trust projection more (current evidence is key)

    Args:
        baseline: Prior probability from domain knowledge
        projection: Probability from Monte Carlo evidence analysis
        confidence: Confidence level ["high", "medium", "low"]

    Returns:
        Blended probability, clamped to [0.001, 0.95]
    """
    weight = CONFIDENCE_WEIGHTS.get(confidence.lower(), 0.5)

    # Weighted average
    result = weight * baseline + (1 - weight) * projection

    # Clamp to valid range
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, result))


def blend_with_market(
    ai_prob: float,
    market_price: float,
    market_efficiency: float = 0.5
) -> float:
    """
    Blend AI probability with market price based on market efficiency.

    Higher volume/liquidity markets are more efficient and should
    be weighted more heavily.

    Args:
        ai_prob: AI's probability estimate
        market_price: Current market price
        market_efficiency: Weight given to market (0-1)

    Returns:
        Blended probability
    """
    result = (1 - market_efficiency) * ai_prob + market_efficiency * market_price
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, result))


def get_market_efficiency_weight(volume: float) -> float:
    """
    Determine market efficiency weight based on trading volume.

    Higher volume markets are more likely to be efficiently priced.

    Args:
        volume: Trading volume in dollars

    Returns:
        Efficiency weight (0-1)
    """
    # Load thresholds from priors.json
    priors_path = CONFIG_DIR / "priors.json"

    try:
        with open(priors_path) as f:
            priors = json.load(f)
            thresholds = priors.get("market_efficiency", {}).get("thresholds", {})
    except (FileNotFoundError, json.JSONDecodeError):
        # Default thresholds
        thresholds = {
            "very_high": {"min_volume": 1000000, "market_weight": 0.8},
            "high": {"min_volume": 100000, "market_weight": 0.6},
            "medium": {"min_volume": 10000, "market_weight": 0.4},
            "low": {"min_volume": 1000, "market_weight": 0.2}
        }

    # Find appropriate tier
    if volume >= thresholds["very_high"]["min_volume"]:
        return thresholds["very_high"]["market_weight"]
    elif volume >= thresholds["high"]["min_volume"]:
        return thresholds["high"]["market_weight"]
    elif volume >= thresholds["medium"]["min_volume"]:
        return thresholds["medium"]["market_weight"]
    else:
        return thresholds["low"]["market_weight"]


def load_domain_prior(
    category: str,
    subcategory: Optional[str] = None
) -> Optional[float]:
    """
    Load a baseline prior probability from domain configuration.

    Args:
        category: Main category (e.g., "economy", "politics")
        subcategory: Specific type (e.g., "fed_rate_cut", "election")

    Returns:
        Prior probability if found, None otherwise
    """
    priors_path = CONFIG_DIR / "priors.json"

    try:
        with open(priors_path) as f:
            priors = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    domain_priors = priors.get("domain_priors", {})

    if category not in domain_priors:
        return None

    category_priors = domain_priors[category]

    if subcategory and subcategory in category_priors:
        sub_data = category_priors[subcategory]
        # Look for probability keys
        for key in ["base_probability", "probability", "approval_rate",
                    "pass_rate", "monthly_probability", "annual_probability"]:
            if key in sub_data:
                return sub_data[key]

    return None


def apply_time_decay(
    probability: float,
    days_remaining: int,
    direction: str = "YES"
) -> float:
    """
    Adjust probability based on time remaining until resolution.

    Events that haven't happened yet are less likely as time runs out.
    Events that have started are more likely to complete.

    Args:
        probability: Current probability estimate
        days_remaining: Days until market resolution
        direction: "YES" if event needs to happen, "NO" if needs to not happen

    Returns:
        Time-adjusted probability
    """
    priors_path = CONFIG_DIR / "priors.json"

    try:
        with open(priors_path) as f:
            priors = json.load(f)
            time_factors = priors.get("time_decay", {}).get("factors", {})
    except (FileNotFoundError, json.JSONDecodeError):
        time_factors = {
            "immediate": {"days": 7, "confidence_multiplier": 1.2},
            "short_term": {"days": 30, "confidence_multiplier": 1.0},
            "medium_term": {"days": 90, "confidence_multiplier": 0.8},
            "long_term": {"days": 365, "confidence_multiplier": 0.6}
        }

    # Determine time horizon
    if days_remaining <= time_factors["immediate"]["days"]:
        multiplier = time_factors["immediate"]["confidence_multiplier"]
    elif days_remaining <= time_factors["short_term"]["days"]:
        multiplier = time_factors["short_term"]["confidence_multiplier"]
    elif days_remaining <= time_factors["medium_term"]["days"]:
        multiplier = time_factors["medium_term"]["confidence_multiplier"]
    else:
        multiplier = time_factors["long_term"]["confidence_multiplier"]

    # Apply multiplier in logit space for better behavior at extremes
    import math

    def logit(p):
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    base_logit = logit(probability)

    # Adjust based on direction
    if direction == "YES":
        # If event needs to happen and we're running out of time,
        # probability should decrease (unless multiplier > 1)
        adjusted_logit = base_logit * multiplier
    else:
        # If event needs to NOT happen, more time = more chances to fail
        adjusted_logit = base_logit / multiplier if multiplier != 0 else base_logit

    result = sigmoid(adjusted_logit)
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, result))


def ensemble_blend(
    probabilities: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Blend multiple probability estimates with optional weights.

    Useful for combining different models or methods.

    Args:
        probabilities: Dict of source_name -> probability
        weights: Optional dict of source_name -> weight (default: equal weights)

    Returns:
        Weighted average probability
    """
    if not probabilities:
        return 0.5

    if weights is None:
        # Equal weights
        weights = {k: 1.0 for k in probabilities}

    total_weight = sum(weights.get(k, 0) for k in probabilities)
    if total_weight == 0:
        return sum(probabilities.values()) / len(probabilities)

    weighted_sum = sum(
        p * weights.get(k, 0)
        for k, p in probabilities.items()
    )

    result = weighted_sum / total_weight
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, result))


def calculate_confidence_from_ci(
    ci_lower: float,
    ci_upper: float
) -> str:
    """
    Determine confidence level from confidence interval width.

    Narrow intervals = high confidence
    Wide intervals = low confidence

    Args:
        ci_lower: Lower bound of 90% CI
        ci_upper: Upper bound of 90% CI

    Returns:
        Confidence level: "high", "medium", or "low"
    """
    width = ci_upper - ci_lower

    if width < 0.15:  # Less than 15 percentage points
        return "high"
    elif width < 0.30:  # Less than 30 percentage points
        return "medium"
    else:
        return "low"
