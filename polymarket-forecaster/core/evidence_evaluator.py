"""
Evidence Evaluator - Classify evidence as supporting YES/NO and assess strength.
"""
import json
from typing import List, Optional
from datetime import datetime

from . import Evidence
from config.settings import DEFAULT_LLM, EVIDENCE_TYPE_WEIGHTS


def call_llm(prompt: str, model: str = DEFAULT_LLM) -> str:
    """
    STUB: Replace with OpenAI/Anthropic API call.
    API_KEY: OPENAI_API_KEY or ANTHROPIC_API_KEY
    """
    # Mock response based on content analysis
    # In real implementation, this would call the LLM API
    return json.dumps({
        "supports": "YES",
        "strength": 0.7,
        "evidence_type": "B",
        "reasoning": "This evidence suggests a positive outcome based on the stated facts."
    })


class EvidenceEvaluator:
    """Evaluate evidence for prediction markets."""

    def __init__(self, model: str = DEFAULT_LLM):
        """
        Initialize the evaluator.

        Args:
            model: LLM model to use for evaluation
        """
        self.model = model

    def evaluate(
        self,
        evidence_text: str,
        market_question: str,
        source: Optional[str] = None,
        url: Optional[str] = None
    ) -> Evidence:
        """
        Evaluate a piece of evidence against a market question.

        Args:
            evidence_text: The evidence content to evaluate
            market_question: The market question to evaluate against
            source: Source name (e.g., "Reuters", "Twitter")
            url: URL of the evidence source

        Returns:
            Evidence object with classification and strength
        """
        prompt = f"""Analyze this evidence in the context of a prediction market question.

Market Question: {market_question}

Evidence:
{evidence_text}

Evaluate this evidence and respond in JSON format:
{{
    "supports": "YES" or "NO" or "NEUTRAL",
    "strength": 0.0-1.0 (how strongly does this support the conclusion),
    "evidence_type": "A" (primary/official source), "B" (high-quality secondary), "C" (standard news), or "D" (weak/unverified),
    "reasoning": "Brief explanation of your assessment"
}}

Consider:
- Does this directly address the market question?
- Is the source reliable and authoritative?
- How recent and relevant is this information?
- Are there caveats or uncertainties mentioned?

JSON response:"""

        response = call_llm(prompt, self.model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback for unparseable response
            data = {
                "supports": "NEUTRAL",
                "strength": 0.3,
                "evidence_type": "D",
                "reasoning": "Could not evaluate evidence"
            }

        return Evidence(
            source=source or "Unknown",
            content=evidence_text[:500],  # Truncate for storage
            supports=data.get("supports", "NEUTRAL"),
            strength=float(data.get("strength", 0.5)),
            evidence_type=data.get("evidence_type", "C"),
            reasoning=data.get("reasoning", ""),
            url=url,
            timestamp=datetime.now()
        )

    def evaluate_batch(
        self,
        evidence_items: List[dict],
        market_question: str
    ) -> List[Evidence]:
        """
        Evaluate multiple pieces of evidence.

        Args:
            evidence_items: List of dicts with 'text', 'source', 'url' keys
            market_question: The market question

        Returns:
            List of Evidence objects
        """
        results = []
        for item in evidence_items:
            evidence = self.evaluate(
                evidence_text=item.get("text", item.get("content", "")),
                market_question=market_question,
                source=item.get("source", "Unknown"),
                url=item.get("url")
            )
            results.append(evidence)
        return results

    def classify_source_type(self, source: str, url: Optional[str] = None) -> str:
        """
        Classify a source into evidence type categories.

        Args:
            source: Source name
            url: Optional URL for additional context

        Returns:
            Evidence type: "A", "B", "C", or "D"
        """
        source_lower = source.lower()

        # Type A: Primary/Official sources
        primary_sources = [
            'whitehouse.gov', 'federalreserve.gov', 'sec.gov',
            'official', 'press release', 'government', '.gov'
        ]
        if any(s in source_lower for s in primary_sources):
            return "A"

        # Type B: High-quality secondary sources
        quality_sources = [
            'reuters', 'associated press', 'ap news', 'bloomberg',
            'wall street journal', 'wsj', 'financial times', 'ft',
            'new york times', 'nyt', 'washington post', 'economist'
        ]
        if any(s in source_lower for s in quality_sources):
            return "B"

        # Type D: Weak sources
        weak_sources = [
            'twitter', 'x.com', 'reddit', 'facebook', 'tiktok',
            'blog', 'forum', 'anonymous', 'rumor'
        ]
        if any(s in source_lower for s in weak_sources):
            return "D"

        # Type C: Standard sources (default)
        return "C"

    def aggregate_evidence(self, evidence_list: List[Evidence]) -> dict:
        """
        Aggregate multiple pieces of evidence into a summary.

        Args:
            evidence_list: List of Evidence objects

        Returns:
            Dict with aggregated statistics
        """
        if not evidence_list:
            return {
                "total_count": 0,
                "yes_count": 0,
                "no_count": 0,
                "neutral_count": 0,
                "weighted_score": 0.0,
                "average_strength": 0.0,
                "dominant_direction": "NEUTRAL"
            }

        yes_evidence = [e for e in evidence_list if e.supports == "YES"]
        no_evidence = [e for e in evidence_list if e.supports == "NO"]
        neutral_evidence = [e for e in evidence_list if e.supports == "NEUTRAL"]

        # Calculate weighted scores
        yes_score = sum(e.weight for e in yes_evidence)
        no_score = sum(e.weight for e in no_evidence)
        total_weight = yes_score + no_score

        if total_weight > 0:
            weighted_score = (yes_score - no_score) / total_weight  # -1 to +1
        else:
            weighted_score = 0.0

        avg_strength = sum(e.strength for e in evidence_list) / len(evidence_list)

        # Determine dominant direction
        if yes_score > no_score * 1.2:  # 20% threshold
            dominant = "YES"
        elif no_score > yes_score * 1.2:
            dominant = "NO"
        else:
            dominant = "NEUTRAL"

        return {
            "total_count": len(evidence_list),
            "yes_count": len(yes_evidence),
            "no_count": len(no_evidence),
            "neutral_count": len(neutral_evidence),
            "yes_weighted_score": yes_score,
            "no_weighted_score": no_score,
            "weighted_score": weighted_score,
            "average_strength": avg_strength,
            "dominant_direction": dominant
        }

    def generate_evidence_summary(self, evidence_list: List[Evidence]) -> str:
        """
        Generate a human-readable summary of the evidence.

        Args:
            evidence_list: List of Evidence objects

        Returns:
            Summary string
        """
        agg = self.aggregate_evidence(evidence_list)

        if agg["total_count"] == 0:
            return "No evidence collected."

        summary_parts = [
            f"Analyzed {agg['total_count']} pieces of evidence:",
            f"  - {agg['yes_count']} supporting YES",
            f"  - {agg['no_count']} supporting NO",
            f"  - {agg['neutral_count']} neutral",
            f"Weighted score: {agg['weighted_score']:+.2f} (range: -1 to +1)",
            f"Overall direction: {agg['dominant_direction']}"
        ]

        return "\n".join(summary_parts)
