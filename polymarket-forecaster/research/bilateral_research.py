"""
Bilateral Research - Gather evidence for BOTH sides of a prediction market.

This prevents confirmation bias by explicitly researching YES and NO cases.
"""
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

from core import Evidence
from core.evidence_evaluator import EvidenceEvaluator
from .news_search import search_news, NewsResult
from config.settings import DEFAULT_LLM


def call_llm(prompt: str, model: str = DEFAULT_LLM) -> str:
    """
    STUB: Replace with OpenAI/Anthropic API call.
    """
    # Mock response for query generation
    return json.dumps({
        "yes_queries": [
            "evidence supporting outcome",
            "reasons for yes",
            "bullish case analysis"
        ],
        "no_queries": [
            "evidence against outcome",
            "reasons for no",
            "bearish case analysis"
        ]
    })


@dataclass
class ResearchResult:
    """Complete bilateral research result."""
    pro_evidence: List[Evidence]
    con_evidence: List[Evidence]
    yes_query_count: int
    no_query_count: int
    total_sources: int
    summary: str


class BilateralResearcher:
    """
    Research both sides of a prediction market question.

    This class ensures balanced evidence gathering to prevent
    confirmation bias in probability estimates.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        num_queries_per_side: int = 3,
        results_per_query: int = 5
    ):
        """
        Initialize the researcher.

        Args:
            model: LLM model for evidence evaluation
            num_queries_per_side: Number of search queries per side
            results_per_query: Number of results to fetch per query
        """
        self.model = model
        self.num_queries_per_side = num_queries_per_side
        self.results_per_query = results_per_query
        self.evaluator = EvidenceEvaluator(model)

    def generate_search_queries(
        self,
        market_question: str
    ) -> Tuple[List[str], List[str]]:
        """
        Generate search queries for both YES and NO cases.

        Args:
            market_question: The prediction market question

        Returns:
            Tuple of (yes_queries, no_queries)
        """
        prompt = f"""Generate search queries to research BOTH sides of this prediction market question.

Question: {market_question}

Generate {self.num_queries_per_side} search queries that would find evidence SUPPORTING a YES outcome,
and {self.num_queries_per_side} queries that would find evidence SUPPORTING a NO outcome.

Respond in JSON format:
{{
    "yes_queries": ["query1", "query2", "query3"],
    "no_queries": ["query1", "query2", "query3"]
}}

Make queries specific and likely to find relevant news/analysis."""

        response = call_llm(prompt, self.model)

        try:
            data = json.loads(response)
            yes_queries = data.get("yes_queries", [])
            no_queries = data.get("no_queries", [])
        except json.JSONDecodeError:
            # Fallback: generate simple queries
            yes_queries = [
                f"{market_question} likely yes",
                f"{market_question} evidence supporting",
                f"{market_question} bullish case"
            ]
            no_queries = [
                f"{market_question} unlikely no",
                f"{market_question} evidence against",
                f"{market_question} bearish case"
            ]

        return (
            yes_queries[:self.num_queries_per_side],
            no_queries[:self.num_queries_per_side]
        )

    def research_side(
        self,
        queries: List[str],
        market_question: str,
        side: str
    ) -> List[Evidence]:
        """
        Research one side of the question.

        Args:
            queries: Search queries to use
            market_question: Original market question
            side: "YES" or "NO" - the side we're researching FOR

        Returns:
            List of Evidence objects
        """
        all_results: List[NewsResult] = []

        for query in queries:
            results = search_news(query, num_results=self.results_per_query)
            all_results.extend(results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Convert to evidence items for evaluation
        evidence_items = []
        for result in unique_results:
            evidence_dict = {
                "text": f"{result.title}\n\n{result.snippet}",
                "source": result.source,
                "url": result.url
            }
            evidence_items.append(evidence_dict)

        # Evaluate all evidence
        evaluated = self.evaluator.evaluate_batch(evidence_items, market_question)

        return evaluated

    def research(
        self,
        market_question: str
    ) -> Tuple[List[Evidence], List[Evidence]]:
        """
        Research BOTH sides of the question.

        This is the main entry point for bilateral research.

        Args:
            market_question: The prediction market question

        Returns:
            Tuple of (pro_evidence, con_evidence)
            - pro_evidence: Evidence that supports YES
            - con_evidence: Evidence that supports NO
        """
        # Generate queries for both sides
        yes_queries, no_queries = self.generate_search_queries(market_question)

        # Research each side
        yes_side_evidence = self.research_side(
            yes_queries, market_question, "YES"
        )
        no_side_evidence = self.research_side(
            no_queries, market_question, "NO"
        )

        # Combine and categorize all evidence
        all_evidence = yes_side_evidence + no_side_evidence

        # Separate by what the evidence actually supports
        pro_evidence = [e for e in all_evidence if e.supports == "YES"]
        con_evidence = [e for e in all_evidence if e.supports == "NO"]

        return (pro_evidence, con_evidence)

    def full_research(
        self,
        market_question: str
    ) -> ResearchResult:
        """
        Perform full bilateral research with summary.

        Args:
            market_question: The prediction market question

        Returns:
            ResearchResult with all evidence and summary
        """
        yes_queries, no_queries = self.generate_search_queries(market_question)
        pro_evidence, con_evidence = self.research(market_question)

        # Generate summary
        summary = self._generate_summary(pro_evidence, con_evidence)

        return ResearchResult(
            pro_evidence=pro_evidence,
            con_evidence=con_evidence,
            yes_query_count=len(yes_queries),
            no_query_count=len(no_queries),
            total_sources=len(pro_evidence) + len(con_evidence),
            summary=summary
        )

    def _generate_summary(
        self,
        pro_evidence: List[Evidence],
        con_evidence: List[Evidence]
    ) -> str:
        """Generate a summary of the bilateral research."""
        pro_strength = sum(e.weight for e in pro_evidence)
        con_strength = sum(e.weight for e in con_evidence)

        total_strength = pro_strength + con_strength

        if total_strength == 0:
            balance = "No significant evidence found for either side."
        elif pro_strength > con_strength * 1.5:
            balance = "Evidence leans toward YES outcome."
        elif con_strength > pro_strength * 1.5:
            balance = "Evidence leans toward NO outcome."
        else:
            balance = "Evidence is relatively balanced between YES and NO."

        summary = f"""Bilateral Research Summary:
- {len(pro_evidence)} pieces of evidence supporting YES (weighted strength: {pro_strength:.2f})
- {len(con_evidence)} pieces of evidence supporting NO (weighted strength: {con_strength:.2f})

{balance}

Top YES evidence:
{self._format_top_evidence(pro_evidence)}

Top NO evidence:
{self._format_top_evidence(con_evidence)}"""

        return summary

    def _format_top_evidence(
        self,
        evidence: List[Evidence],
        top_n: int = 3
    ) -> str:
        """Format top evidence items for display."""
        if not evidence:
            return "  (No evidence found)"

        sorted_evidence = sorted(evidence, key=lambda e: e.weight, reverse=True)
        top = sorted_evidence[:top_n]

        lines = []
        for e in top:
            lines.append(
                f"  - [{e.evidence_type}] {e.source}: {e.reasoning[:100]}..."
            )

        return "\n".join(lines) if lines else "  (No evidence found)"


def quick_research(
    market_question: str,
    num_results: int = 10
) -> Tuple[List[Evidence], List[Evidence]]:
    """
    Quick bilateral research with default settings.

    Args:
        market_question: The prediction market question
        num_results: Total number of results to fetch

    Returns:
        Tuple of (pro_evidence, con_evidence)
    """
    researcher = BilateralResearcher(
        num_queries_per_side=2,
        results_per_query=num_results // 4
    )
    return researcher.research(market_question)
