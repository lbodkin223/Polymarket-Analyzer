"""
Analyst Agent - Bayesian probability aggregation.

From Polyseer: "Applies Bayesian mathematics with correlation adjustments"
Uses Log-Likelihood Ratios: LLR = log(P(evidence|YES) / P(evidence|NO))

From Outcome-RL paper: "Median prediction sampling across ensemble of 7 predictions"
"""
import math
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .base import BaseAgent, AgentConfig, AgentResponse
from core.monte_carlo import (
    run_monte_carlo, apply_evidence_effects, logit, sigmoid
)
from core.calibration import calibrate, compute_ece
from config.settings import DEFAULT_LLM, DEFAULT_SIMULATIONS


@dataclass
class AnalysisResult:
    """Output from the Analyst Agent."""
    raw_probability: float
    calibrated_probability: float
    confidence_interval: Tuple[float, float]
    yes_llr: float  # Total log-likelihood ratio for YES
    no_llr: float   # Total log-likelihood ratio for NO
    evidence_weights: Dict[str, float]
    ensemble_probabilities: List[float]  # From multiple runs
    reasoning: str


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Aggregates evidence into probability estimates.

    Implements:
    1. Bayesian Log-Likelihood Ratio aggregation (from Polyseer)
    2. Monte Carlo simulation with evidence multipliers
    3. Ensemble predictions (median of N runs, from Outcome-RL)
    4. Calibration layer (from AIA Forecaster)

    This is where the core probability math happens.
    """

    def __init__(self, config: Optional[AgentConfig] = None,
                 n_ensemble: int = 7,
                 n_simulations: int = DEFAULT_SIMULATIONS):
        if config is None:
            config = AgentConfig(
                name="AnalystAgent",
                model=DEFAULT_LLM,
                temperature=0.2,  # Low temp for analytical work
                system_prompt=self._get_system_prompt()
            )
        super().__init__(config)
        self.n_ensemble = n_ensemble
        self.n_simulations = n_simulations

    def _get_system_prompt(self) -> str:
        return """You are a quantitative analyst for prediction market forecasting.

Your job is to aggregate evidence into probability estimates using Bayesian methods.

Key techniques:
1. Log-Likelihood Ratios (LLR): log(P(evidence|YES) / P(evidence|NO))
2. Evidence weights based on source quality (A=1.0, B=0.8, C=0.5, D=0.2)
3. Correlation adjustments for dependent evidence
4. Ensemble prediction aggregation

Be precise and quantitative. Show your reasoning.

Always respond in valid JSON format."""

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Aggregate evidence into probability estimates.

        Args:
            input_data: Must contain validated evidence and baseline probability

        Returns:
            AgentResponse with AnalysisResult
        """
        validated_evidence = input_data.get("validated_evidence", [])
        baseline = input_data.get("baseline_probability", 0.5)
        question = input_data.get("question", "")

        if not validated_evidence:
            # No evidence - return baseline with wide CI
            return self.create_response(
                success=True,
                data={
                    "raw_probability": baseline,
                    "calibrated_probability": calibrate(baseline),
                    "confidence_interval": (max(0.1, baseline - 0.2),
                                             min(0.9, baseline + 0.2)),
                    "reasoning": "No validated evidence, returning baseline"
                },
                reasoning="No evidence to analyze",
                confidence=0.3
            )

        # Calculate Log-Likelihood Ratios
        yes_llr, no_llr, evidence_weights = self._calculate_llr(validated_evidence)

        # Convert evidence to Monte Carlo multipliers
        from core import Evidence
        evidence_objects = [
            Evidence(
                source=e.get("source", "Unknown"),
                content=e.get("content", ""),
                supports=e.get("supports", "NEUTRAL"),
                strength=e.get("strength", 0.5),
                evidence_type=e.get("evidence_type", "C"),
                reasoning=e.get("reasoning", "")
            )
            for e in validated_evidence
        ]
        multipliers = apply_evidence_effects(evidence_objects)

        # Run ensemble of Monte Carlo simulations (from Outcome-RL paper)
        ensemble_probs = []
        for i in range(self.n_ensemble):
            mc_result = run_monte_carlo(
                baseline=baseline,
                multipliers=multipliers,
                n_sims=self.n_simulations,
                seed=i * 42  # Different seed for each ensemble member
            )
            ensemble_probs.append(mc_result.median_probability)

        # Take median of ensemble (from Outcome-RL paper)
        raw_probability = float(np.median(ensemble_probs))

        # Calculate confidence interval from ensemble spread
        ci_lower = float(np.percentile(ensemble_probs, 5))
        ci_upper = float(np.percentile(ensemble_probs, 95))

        # Apply calibration (from AIA Forecaster)
        calibrated_probability = calibrate(raw_probability)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            baseline, raw_probability, calibrated_probability,
            yes_llr, no_llr, len(validated_evidence)
        )

        result = AnalysisResult(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            confidence_interval=(ci_lower, ci_upper),
            yes_llr=yes_llr,
            no_llr=no_llr,
            evidence_weights=evidence_weights,
            ensemble_probabilities=ensemble_probs,
            reasoning=reasoning
        )

        return self.create_response(
            success=True,
            data={
                "raw_probability": result.raw_probability,
                "calibrated_probability": result.calibrated_probability,
                "confidence_interval": result.confidence_interval,
                "yes_llr": result.yes_llr,
                "no_llr": result.no_llr,
                "evidence_weights": result.evidence_weights,
                "ensemble_probabilities": result.ensemble_probabilities,
                "reasoning": result.reasoning
            },
            reasoning=reasoning,
            confidence=min(0.9, 0.5 + len(validated_evidence) * 0.05)
        )

    def _calculate_llr(self, evidence_list: List[Dict]) -> Tuple[float, float, Dict]:
        """
        Calculate Log-Likelihood Ratios for evidence.

        From Polyseer: LLR = log(P(evidence|YES) / P(evidence|NO))

        Returns:
            Tuple of (total_yes_llr, total_no_llr, evidence_weights)
        """
        yes_llr = 0.0
        no_llr = 0.0
        weights = {}

        type_weights = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.2}

        for evidence in evidence_list:
            source = evidence.get("source", "Unknown")
            supports = evidence.get("supports", "NEUTRAL")
            strength = evidence.get("strength", 0.5)
            etype = evidence.get("evidence_type", "C")

            weight = type_weights.get(etype, 0.5) * strength
            weights[source] = weight

            if supports == "YES":
                # Evidence supports YES: P(evidence|YES) > P(evidence|NO)
                # LLR is positive
                llr = weight * 2  # Scale factor
                yes_llr += llr
            elif supports == "NO":
                # Evidence supports NO: P(evidence|NO) > P(evidence|YES)
                # LLR is negative
                llr = weight * 2
                no_llr += llr

        return yes_llr, no_llr, weights

    def _apply_correlation_adjustment(self, evidence_list: List[Dict]) -> float:
        """
        Adjust for correlated evidence.

        Multiple pieces from the same source or about the same event
        should not be counted independently.

        Returns:
            Correlation discount factor (0-1)
        """
        sources = [e.get("source", "") for e in evidence_list]
        unique_sources = set(sources)

        if len(sources) == 0:
            return 1.0

        # Discount for repeated sources
        diversity_ratio = len(unique_sources) / len(sources)

        # Apply square root to soften the penalty
        return math.sqrt(diversity_ratio)

    def _generate_reasoning(self, baseline: float, raw: float, calibrated: float,
                            yes_llr: float, no_llr: float, n_evidence: int) -> str:
        """Generate human-readable reasoning."""
        net_llr = yes_llr - no_llr
        direction = "YES" if net_llr > 0 else "NO" if net_llr < 0 else "neutral"

        parts = [
            f"Starting from baseline: {baseline:.1%}",
            f"Analyzed {n_evidence} pieces of validated evidence",
            f"YES LLR: {yes_llr:.2f}, NO LLR: {no_llr:.2f}, Net: {net_llr:+.2f}",
            f"Evidence direction: {direction}",
            f"Raw probability (ensemble median): {raw:.1%}",
            f"Calibrated probability: {calibrated:.1%}",
        ]

        if abs(calibrated - raw) > 0.05:
            parts.append(f"Calibration adjusted by {(calibrated - raw):+.1%}")

        return " | ".join(parts)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response."""
        return json.dumps({
            "probability": 0.55,
            "confidence": 0.7
        })


class BayesianAnalystAgent(AnalystAgent):
    """
    Enhanced analyst using full Bayesian updating.

    P(H|E) = P(E|H) * P(H) / P(E)

    Where:
    - H = hypothesis (outcome is YES)
    - E = evidence
    - P(H) = prior probability
    - P(E|H) = likelihood of evidence given YES
    - P(E) = marginal probability of evidence
    """

    def bayesian_update(self, prior: float, evidence_list: List[Dict]) -> float:
        """
        Apply sequential Bayesian updates.

        Each piece of evidence updates the probability:
        posterior = likelihood_ratio * prior / (likelihood_ratio * prior + (1 - prior))
        """
        probability = prior

        for evidence in evidence_list:
            supports = evidence.get("supports", "NEUTRAL")
            strength = evidence.get("strength", 0.5)
            etype = evidence.get("evidence_type", "C")

            type_weights = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.2}
            weight = type_weights.get(etype, 0.5)

            if supports == "NEUTRAL":
                continue

            # Calculate likelihood ratio
            if supports == "YES":
                # Strong YES evidence: P(E|YES) / P(E|NO) > 1
                lr = 1 + (strength * weight * 3)  # LR from 1 to 4
            else:
                # Strong NO evidence: P(E|YES) / P(E|NO) < 1
                lr = 1 / (1 + (strength * weight * 3))  # LR from 0.25 to 1

            # Bayesian update
            probability = (lr * probability) / (lr * probability + (1 - probability))

            # Clamp to avoid extremes
            probability = max(0.01, min(0.99, probability))

        return probability

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Run Bayesian analysis."""
        validated_evidence = input_data.get("validated_evidence", [])
        baseline = input_data.get("baseline_probability", 0.5)

        # Apply Bayesian updates
        bayesian_prob = self.bayesian_update(baseline, validated_evidence)

        # Also run parent's ensemble method
        parent_result = super().run(input_data)

        # Combine both estimates (weighted average)
        ensemble_prob = parent_result.data.get("raw_probability", baseline)
        combined_prob = 0.5 * bayesian_prob + 0.5 * ensemble_prob

        # Update the response
        parent_result.data["bayesian_probability"] = bayesian_prob
        parent_result.data["combined_probability"] = combined_prob
        parent_result.reasoning += f" | Bayesian: {bayesian_prob:.1%}"

        return parent_result
