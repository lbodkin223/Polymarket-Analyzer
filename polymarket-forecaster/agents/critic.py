"""
Critic Agent - Validates evidence quality and identifies research gaps.

From Polyseer: "Validates evidence quality and identifies research gaps"

This is a key differentiator from simple pipelines - the Critic
ensures evidence quality before it reaches the Analyst.
"""
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentResponse
from core import Evidence
from config.settings import DEFAULT_LLM


@dataclass
class CriticReport:
    """Output from the Critic Agent."""
    validated_evidence: List[Dict]
    rejected_evidence: List[Dict]
    quality_issues: List[str]
    research_gaps: List[str]
    additional_queries_needed: List[str]
    overall_quality_score: float  # 0-1
    recommendation: str  # "proceed", "needs_more_research", "insufficient"


class CriticAgent(BaseAgent):
    """
    Critic Agent - Quality control for the research pipeline.

    Responsibilities:
    1. Validate evidence quality and relevance
    2. Identify weak or unreliable sources
    3. Detect research gaps (missing perspectives)
    4. Flag contradictory evidence
    5. Recommend additional research if needed
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="CriticAgent",
                model=DEFAULT_LLM,
                temperature=0.3,  # Lower temp for critical analysis
                system_prompt=self._get_system_prompt()
            )
        super().__init__(config)
        self.min_evidence_count = 3
        self.min_quality_score = 0.5

    def _get_system_prompt(self) -> str:
        return """You are a critical analysis agent for prediction market forecasting.

Your job is to validate evidence quality and identify gaps in research.

Evaluation criteria:
1. SOURCE QUALITY
   - A: Official documents, press releases, regulatory filings (2.0x weight)
   - B: Major news outlets (Reuters, Bloomberg, WSJ) (0.8x weight)
   - C: Standard news, reputable blogs (0.5x weight)
   - D: Social media, unverified claims, rumors (0.2x weight)

2. RELEVANCE
   - Does the evidence directly address the question?
   - Is the information current and timely?
   - Are there caveats or conditions mentioned?

3. BALANCE
   - Is there evidence for both YES and NO outcomes?
   - Are any important perspectives missing?
   - Is the research one-sided?

4. CONTRADICTIONS
   - Do any pieces of evidence contradict each other?
   - If so, which is more credible?

Be skeptical and thorough. Flag any issues you find.

Always respond in valid JSON format."""

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Validate evidence and identify research gaps.

        Args:
            input_data: Must contain evidence lists from Researcher

        Returns:
            AgentResponse with CriticReport
        """
        yes_evidence = input_data.get("yes_evidence", [])
        no_evidence = input_data.get("no_evidence", [])
        neutral_evidence = input_data.get("neutral_evidence", [])
        question = input_data.get("question", "")

        all_evidence = yes_evidence + no_evidence + neutral_evidence

        if not all_evidence:
            return self.create_response(
                success=False,
                data={"recommendation": "insufficient"},
                reasoning="No evidence provided to validate",
                confidence=0.0,
                error="No evidence to validate"
            )

        # Validate each piece of evidence
        validated = []
        rejected = []
        issues = []

        for evidence in all_evidence:
            is_valid, reason = self._validate_evidence(evidence, question)
            if is_valid:
                validated.append(evidence)
            else:
                rejected.append({**evidence, "rejection_reason": reason})
                issues.append(reason)

        # Check for research gaps
        gaps = self._identify_gaps(yes_evidence, no_evidence, question)

        # Generate additional query suggestions
        additional_queries = self._suggest_queries(gaps, question)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            validated, rejected, yes_evidence, no_evidence
        )

        # Determine recommendation
        if len(validated) < self.min_evidence_count:
            recommendation = "insufficient"
        elif gaps and quality_score < self.min_quality_score:
            recommendation = "needs_more_research"
        else:
            recommendation = "proceed"

        report = CriticReport(
            validated_evidence=validated,
            rejected_evidence=rejected,
            quality_issues=issues,
            research_gaps=gaps,
            additional_queries_needed=additional_queries,
            overall_quality_score=quality_score,
            recommendation=recommendation
        )

        return self.create_response(
            success=True,
            data={
                "validated_evidence": report.validated_evidence,
                "rejected_evidence": report.rejected_evidence,
                "quality_issues": report.quality_issues,
                "research_gaps": report.research_gaps,
                "additional_queries": report.additional_queries_needed,
                "quality_score": report.overall_quality_score,
                "recommendation": report.recommendation
            },
            reasoning=f"Validated {len(validated)}/{len(all_evidence)} evidence. "
                      f"Quality: {quality_score:.2f}. Recommendation: {recommendation}",
            confidence=quality_score
        )

    def _validate_evidence(self, evidence: Dict, question: str) -> Tuple[bool, str]:
        """
        Validate a single piece of evidence.

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check for minimum content
        content = evidence.get("content", "")
        if len(content) < 20:
            return False, "Content too short to be meaningful"

        # Check source quality
        evidence_type = evidence.get("evidence_type", "D")
        if evidence_type == "D" and evidence.get("strength", 0) > 0.7:
            return False, "High strength claim from weak source (D-type)"

        # Check for obvious issues in content
        content_lower = content.lower()
        red_flags = [
            "unconfirmed", "rumor", "allegedly", "sources say",
            "could not be verified", "fake news"
        ]
        for flag in red_flags:
            if flag in content_lower:
                # Don't reject, but reduce confidence
                evidence["strength"] = evidence.get("strength", 0.5) * 0.7

        # Use LLM for deeper validation
        prompt = f"""Validate this evidence for the prediction market question.

QUESTION: {question}

EVIDENCE:
Source: {evidence.get('source', 'Unknown')}
Type: {evidence.get('evidence_type', 'Unknown')}
Supports: {evidence.get('supports', 'Unknown')}
Content: {content[:300]}

Is this evidence:
1. Relevant to the question?
2. From a credible source?
3. Current and timely?

Respond in JSON:
{{
    "is_valid": true or false,
    "reason": "explanation if invalid",
    "adjusted_strength": 0.0-1.0
}}"""

        response = self.call_llm(prompt)
        data = self.parse_json_response(response)

        is_valid = data.get("is_valid", True)
        reason = data.get("reason", "")

        # Adjust strength if suggested
        if "adjusted_strength" in data:
            evidence["strength"] = data["adjusted_strength"]

        return is_valid, reason

    def _identify_gaps(self, yes_evidence: List, no_evidence: List,
                       question: str) -> List[str]:
        """Identify gaps in the research."""
        gaps = []

        # Check for balance
        yes_count = len(yes_evidence)
        no_count = len(no_evidence)

        if yes_count == 0:
            gaps.append("No evidence found supporting YES outcome")
        if no_count == 0:
            gaps.append("No evidence found supporting NO outcome")

        if yes_count > 0 and no_count > 0:
            ratio = max(yes_count, no_count) / min(yes_count, no_count)
            if ratio > 3:
                weaker = "YES" if yes_count < no_count else "NO"
                gaps.append(f"Evidence heavily skewed, need more {weaker} research")

        # Check for source diversity
        all_sources = set()
        for e in yes_evidence + no_evidence:
            all_sources.add(e.get("source", "").lower())

        if len(all_sources) < 3:
            gaps.append("Limited source diversity - need more varied sources")

        # Check for high-quality sources
        high_quality = [e for e in yes_evidence + no_evidence
                        if e.get("evidence_type") in ["A", "B"]]
        if len(high_quality) < 2:
            gaps.append("Lacking high-quality (A/B tier) sources")

        return gaps

    def _suggest_queries(self, gaps: List[str], question: str) -> List[str]:
        """Suggest additional queries based on identified gaps."""
        if not gaps:
            return []

        prompt = f"""Based on these research gaps, suggest additional search queries.

QUESTION: {question}

GAPS:
{chr(10).join('- ' + g for g in gaps)}

Generate 3 specific search queries that would help fill these gaps.
Respond as a JSON array: ["query1", "query2", "query3"]"""

        response = self.call_llm(prompt)
        try:
            queries = json.loads(response)
            if isinstance(queries, list):
                return queries[:3]
        except:
            pass

        # Fallback queries
        fallback = []
        for gap in gaps[:3]:
            if "YES" in gap:
                fallback.append(f"{question} evidence supporting")
            elif "NO" in gap:
                fallback.append(f"{question} evidence against")
            else:
                fallback.append(f"{question} analysis")

        return fallback

    def _calculate_quality_score(self, validated: List, rejected: List,
                                  yes_evidence: List, no_evidence: List) -> float:
        """Calculate overall research quality score."""
        if not validated:
            return 0.0

        # Validation ratio (40% weight)
        total = len(validated) + len(rejected)
        validation_score = len(validated) / total if total > 0 else 0

        # Balance score (30% weight)
        yes_count = len([e for e in validated if e.get("supports") == "YES"])
        no_count = len([e for e in validated if e.get("supports") == "NO"])
        if yes_count + no_count > 0:
            balance = min(yes_count, no_count) / max(yes_count, no_count)
        else:
            balance = 0

        # Source quality score (30% weight)
        type_scores = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.2}
        quality_sum = sum(type_scores.get(e.get("evidence_type", "C"), 0.5)
                          for e in validated)
        quality_score = quality_sum / len(validated) if validated else 0

        # Weighted average
        final_score = (
            0.4 * validation_score +
            0.3 * balance +
            0.3 * quality_score
        )

        return min(1.0, max(0.0, final_score))

    def _mock_response(self, prompt: str) -> str:
        """Generate mock validation response."""
        return json.dumps({
            "is_valid": True,
            "reason": "",
            "adjusted_strength": 0.6
        })
