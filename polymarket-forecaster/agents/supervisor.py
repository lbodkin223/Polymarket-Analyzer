"""
Supervisor Agent - Reconciles conflicts and orchestrates re-research.

From AIA Forecaster: "Supervisor Agent reconciles disparate forecasts
from individual agents, performs additional corrective searches,
resolves conflicting perspectives"
"""
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentResponse
from config.settings import EXPENSIVE_LLM


@dataclass
class SupervisorDecision:
    """Decision from the Supervisor Agent."""
    final_probability: float
    confidence: float
    action: str  # "accept", "re_research", "reject"
    conflicts_resolved: List[str]
    adjustments_made: List[str]
    reasoning: str


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Final quality control and conflict resolution.

    Responsibilities:
    1. Review outputs from Planner, Researcher, Critic, Analyst
    2. Identify and resolve conflicting evidence
    3. Trigger additional research if needed
    4. Make final probability adjustments
    5. Decide whether the forecast is ready

    This is the "meta" agent that oversees the entire pipeline.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="SupervisorAgent",
                model=EXPENSIVE_LLM,  # Use expensive model for supervision
                temperature=0.3,
                system_prompt=self._get_system_prompt()
            )
        super().__init__(config)
        self.max_research_iterations = 3

    def _get_system_prompt(self) -> str:
        return """You are the supervising agent for a prediction market forecasting system.

Your job is to:
1. Review the entire analysis pipeline
2. Identify contradictions or conflicts in evidence
3. Decide if the forecast is reliable enough to act on
4. Suggest corrections or additional research if needed

You have authority to:
- Accept the forecast and proceed to trading decision
- Request additional research on specific topics
- Reject the forecast if evidence is insufficient
- Adjust the final probability based on meta-analysis

Be thorough but efficient. Don't request unnecessary re-research.

Always respond in valid JSON format."""

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Review and reconcile the full analysis.

        Args:
            input_data: Contains outputs from all previous agents:
                - plan: From PlannerAgent
                - research: From ResearcherAgent
                - critique: From CriticAgent
                - analysis: From AnalystAgent
                - iteration: Current iteration number

        Returns:
            AgentResponse with SupervisorDecision
        """
        plan = input_data.get("plan", {})
        research = input_data.get("research", {})
        critique = input_data.get("critique", {})
        analysis = input_data.get("analysis", {})
        iteration = input_data.get("iteration", 1)
        question = input_data.get("question", "")

        # Check for critical issues
        issues = self._identify_issues(plan, research, critique, analysis)

        # Identify conflicts
        conflicts = self._identify_conflicts(research, analysis)

        # Decide on action
        action, adjustments = self._decide_action(
            issues, conflicts, critique, analysis, iteration
        )

        # Calculate final probability
        raw_prob = analysis.get("calibrated_probability",
                                analysis.get("raw_probability", 0.5))
        final_prob, prob_adjustments = self._adjust_probability(
            raw_prob, issues, conflicts, critique
        )
        adjustments.extend(prob_adjustments)

        # Calculate confidence
        confidence = self._calculate_confidence(critique, analysis, issues)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            action, issues, conflicts, adjustments, final_prob
        )

        decision = SupervisorDecision(
            final_probability=final_prob,
            confidence=confidence,
            action=action,
            conflicts_resolved=conflicts,
            adjustments_made=adjustments,
            reasoning=reasoning
        )

        return self.create_response(
            success=True,
            data={
                "final_probability": decision.final_probability,
                "confidence": decision.confidence,
                "action": decision.action,
                "conflicts_resolved": decision.conflicts_resolved,
                "adjustments_made": decision.adjustments_made,
                "reasoning": decision.reasoning,
                "ready_for_trading": action == "accept"
            },
            reasoning=reasoning,
            confidence=confidence
        )

    def _identify_issues(self, plan: Dict, research: Dict,
                         critique: Dict, analysis: Dict) -> List[str]:
        """Identify issues across the pipeline."""
        issues = []

        # Check research quality
        quality_score = critique.get("quality_score", 0)
        if quality_score < 0.4:
            issues.append(f"Low research quality score: {quality_score:.2f}")

        # Check for research gaps
        gaps = critique.get("research_gaps", [])
        if gaps:
            issues.append(f"Research gaps identified: {len(gaps)}")

        # Check evidence balance
        yes_count = len(research.get("yes_evidence", []))
        no_count = len(research.get("no_evidence", []))
        total = yes_count + no_count

        if total > 0:
            balance = min(yes_count, no_count) / max(yes_count, no_count) if max(yes_count, no_count) > 0 else 0
            if balance < 0.2:
                issues.append(f"Evidence imbalance: {yes_count} YES vs {no_count} NO")

        # Check confidence interval width
        ci = analysis.get("confidence_interval", (0, 1))
        ci_width = ci[1] - ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else 1
        if ci_width > 0.4:
            issues.append(f"Wide confidence interval: {ci_width:.1%}")

        return issues

    def _identify_conflicts(self, research: Dict, analysis: Dict) -> List[str]:
        """Identify conflicting evidence."""
        conflicts = []

        yes_evidence = research.get("yes_evidence", [])
        no_evidence = research.get("no_evidence", [])

        # Look for high-strength conflicting evidence
        strong_yes = [e for e in yes_evidence if e.get("strength", 0) > 0.7]
        strong_no = [e for e in no_evidence if e.get("strength", 0) > 0.7]

        if strong_yes and strong_no:
            conflicts.append(
                f"Strong conflicting evidence: {len(strong_yes)} high-strength YES, "
                f"{len(strong_no)} high-strength NO"
            )

        # Check for same-source conflicts
        yes_sources = {e.get("source", "").lower() for e in yes_evidence}
        no_sources = {e.get("source", "").lower() for e in no_evidence}
        overlap = yes_sources & no_sources

        if overlap:
            conflicts.append(f"Same sources appear on both sides: {overlap}")

        return conflicts

    def _decide_action(self, issues: List[str], conflicts: List[str],
                       critique: Dict, analysis: Dict, iteration: int) -> Tuple[str, List[str]]:
        """Decide what action to take."""
        adjustments = []

        # Check iteration limit
        if iteration >= self.max_research_iterations:
            adjustments.append("Max iterations reached, accepting current result")
            return "accept", adjustments

        # Check critic recommendation
        recommendation = critique.get("recommendation", "proceed")
        if recommendation == "insufficient" and iteration < 2:
            adjustments.append("Insufficient evidence, requesting more research")
            return "re_research", adjustments

        if recommendation == "needs_more_research" and iteration < 2:
            adjustments.append("Gaps identified, requesting targeted research")
            return "re_research", adjustments

        # Check for severe issues
        severe_issues = [i for i in issues if "Low research quality" in i or
                         "Evidence imbalance" in i]
        if severe_issues and iteration < 2:
            adjustments.append(f"Severe issues found: {severe_issues}")
            return "re_research", adjustments

        # Default: accept
        adjustments.append("Quality checks passed")
        return "accept", adjustments

    def _adjust_probability(self, raw_prob: float, issues: List[str],
                            conflicts: List[str], critique: Dict) -> Tuple[float, List[str]]:
        """Make final probability adjustments."""
        prob = raw_prob
        adjustments = []

        # Shrink toward 0.5 if there are issues
        shrinkage_factor = 1.0

        if issues:
            shrinkage_factor *= 0.9
            adjustments.append("Applied 10% shrinkage due to issues")

        if conflicts:
            shrinkage_factor *= 0.85
            adjustments.append("Applied 15% shrinkage due to conflicts")

        # Apply shrinkage (move toward 0.5)
        if shrinkage_factor < 1.0:
            prob = 0.5 + (prob - 0.5) * shrinkage_factor

        # Clamp
        prob = max(0.01, min(0.99, prob))

        if prob != raw_prob:
            adjustments.append(f"Adjusted {raw_prob:.1%} → {prob:.1%}")

        return prob, adjustments

    def _calculate_confidence(self, critique: Dict, analysis: Dict,
                               issues: List[str]) -> float:
        """Calculate final confidence level."""
        base_confidence = analysis.get("confidence", 0.5)
        quality_score = critique.get("quality_score", 0.5)

        # Average of base confidence and quality
        confidence = (base_confidence + quality_score) / 2

        # Penalize for issues
        confidence -= len(issues) * 0.05

        return max(0.1, min(0.95, confidence))

    def _generate_reasoning(self, action: str, issues: List[str],
                            conflicts: List[str], adjustments: List[str],
                            final_prob: float) -> str:
        """Generate supervisor reasoning."""
        parts = [f"Action: {action.upper()}"]

        if issues:
            parts.append(f"Issues: {len(issues)}")
        if conflicts:
            parts.append(f"Conflicts: {len(conflicts)}")
        if adjustments:
            parts.append(f"Adjustments: {len(adjustments)}")

        parts.append(f"Final probability: {final_prob:.1%}")

        return " | ".join(parts)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response."""
        return json.dumps({
            "action": "accept",
            "confidence": 0.7
        })


class ReconciliationSupervisor(SupervisorAgent):
    """
    Enhanced supervisor with explicit conflict reconciliation.

    Uses LLM to reason about conflicting evidence and make
    informed decisions about which to trust.
    """

    def reconcile_conflict(self, yes_evidence: List[Dict],
                           no_evidence: List[Dict], question: str) -> Dict:
        """
        Use LLM to reconcile conflicting evidence.

        Returns:
            Dict with reconciliation result
        """
        prompt = f"""Analyze these conflicting pieces of evidence for a prediction market.

QUESTION: {question}

EVIDENCE SUPPORTING YES:
{json.dumps(yes_evidence[:3], indent=2)}

EVIDENCE SUPPORTING NO:
{json.dumps(no_evidence[:3], indent=2)}

Reconcile this conflict:
1. Which evidence is more credible and why?
2. Are there any misinterpretations?
3. What is the most likely outcome given all evidence?

Respond in JSON:
{{
    "winner": "YES" or "NO" or "UNCERTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "probability_adjustment": -0.2 to 0.2
}}"""

        response = self.call_llm(prompt)
        return self.parse_json_response(response)
