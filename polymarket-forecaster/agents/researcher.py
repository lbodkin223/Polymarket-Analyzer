"""
Researcher Agent - Gathers bilateral evidence from multiple sources.

From Polyseer: "Gathers bilateral evidence from Valyu API
(academic papers, news, regulatory filings)"

From AIA: "Agentic search with discretion over search queries,
adaptive conditioning on previous query results"
"""
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .base import BaseAgent, AgentConfig, AgentResponse
from core import Evidence
from research.news_search import search_news, NewsResult
from config.settings import DEFAULT_LLM


@dataclass
class ResearchOutput:
    """Output from a research session."""
    query: str
    side: str  # "YES" or "NO"
    results: List[NewsResult]
    evaluated_evidence: List[Evidence]
    total_sources: int


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Gathers evidence from multiple sources.

    Key features from AIA Forecaster:
    1. Agentic search - agent decides on queries
    2. Adaptive conditioning - adjusts based on previous results
    3. High-quality source prioritization

    Responsibilities:
    1. Execute search queries from the Planner
    2. Fetch and parse news/data from multiple sources
    3. Classify evidence by source quality (A/B/C/D)
    4. Return structured evidence for the Analyst
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ResearcherAgent",
                model=DEFAULT_LLM,
                temperature=0.5,
                system_prompt=self._get_system_prompt()
            )
        super().__init__(config)
        self.results_per_query = 5
        self.max_queries_per_side = 5

    def _get_system_prompt(self) -> str:
        return """You are a research agent that gathers evidence for prediction market forecasting.

Your job is to:
1. Execute search queries and analyze results
2. Evaluate each piece of evidence for relevance and quality
3. Classify evidence as supporting YES, NO, or NEUTRAL
4. Rate source quality: A (official/primary), B (major news), C (standard), D (weak)

Be objective and evaluate evidence on its merits, not on what outcome you expect.

Always respond in valid JSON format."""

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Execute research based on the plan.

        Args:
            input_data: Must contain 'plan' with yes_queries and no_queries

        Returns:
            AgentResponse with evidence lists
        """
        plan = input_data.get("plan", {})
        question = plan.get("original_question", input_data.get("question", ""))

        yes_queries = plan.get("yes_research_queries", plan.get("yes_queries", []))
        no_queries = plan.get("no_research_queries", plan.get("no_queries", []))

        if not yes_queries and not no_queries:
            return self.create_response(
                success=False,
                data={},
                reasoning="No research queries provided",
                confidence=0.0,
                error="Missing research queries"
            )

        # Execute searches
        all_results: List[NewsResult] = []
        query_outputs: List[ResearchOutput] = []

        # Search for YES evidence
        for query in yes_queries[:self.max_queries_per_side]:
            results = search_news(query, num_results=self.results_per_query)
            all_results.extend(results)
            query_outputs.append(ResearchOutput(
                query=query,
                side="YES",
                results=results,
                evaluated_evidence=[],
                total_sources=len(results)
            ))

        # Search for NO evidence
        for query in no_queries[:self.max_queries_per_side]:
            results = search_news(query, num_results=self.results_per_query)
            all_results.extend(results)
            query_outputs.append(ResearchOutput(
                query=query,
                side="NO",
                results=results,
                evaluated_evidence=[],
                total_sources=len(results)
            ))

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Evaluate each result
        all_evidence = []
        for result in unique_results:
            evidence = self._evaluate_result(result, question)
            all_evidence.append(evidence)

        # Separate by what the evidence supports
        yes_evidence = [e for e in all_evidence if e.supports == "YES"]
        no_evidence = [e for e in all_evidence if e.supports == "NO"]
        neutral_evidence = [e for e in all_evidence if e.supports == "NEUTRAL"]

        return self.create_response(
            success=True,
            data={
                "yes_evidence": [self._evidence_to_dict(e) for e in yes_evidence],
                "no_evidence": [self._evidence_to_dict(e) for e in no_evidence],
                "neutral_evidence": [self._evidence_to_dict(e) for e in neutral_evidence],
                "total_sources": len(unique_results),
                "queries_executed": len(query_outputs)
            },
            reasoning=f"Found {len(yes_evidence)} YES, {len(no_evidence)} NO, "
                      f"{len(neutral_evidence)} NEUTRAL from {len(unique_results)} sources",
            confidence=0.7
        )

    def _evaluate_result(self, result: NewsResult, question: str) -> Evidence:
        """
        Evaluate a search result as evidence.

        Uses LLM to determine:
        - Does it support YES, NO, or NEUTRAL?
        - How strong is the evidence?
        - What type of source is it?
        """
        prompt = f"""Evaluate this search result as evidence for the prediction market question.

QUESTION: {question}

SEARCH RESULT:
Title: {result.title}
Source: {result.source}
Snippet: {result.snippet}

Respond in JSON:
{{
    "supports": "YES" or "NO" or "NEUTRAL",
    "strength": 0.0-1.0,
    "evidence_type": "A" (official/primary) or "B" (major news) or "C" (standard) or "D" (weak/social),
    "reasoning": "Brief explanation"
}}"""

        response = self.call_llm(prompt)
        data = self.parse_json_response(response)

        return Evidence(
            source=result.source,
            content=f"{result.title}\n{result.snippet}"[:500],
            supports=data.get("supports", "NEUTRAL"),
            strength=float(data.get("strength", 0.5)),
            evidence_type=data.get("evidence_type", "C"),
            reasoning=data.get("reasoning", ""),
            url=result.url,
            timestamp=datetime.now()
        )

    def _evidence_to_dict(self, evidence: Evidence) -> Dict[str, Any]:
        """Convert Evidence to dictionary."""
        return {
            "source": evidence.source,
            "content": evidence.content,
            "supports": evidence.supports,
            "strength": evidence.strength,
            "evidence_type": evidence.evidence_type,
            "reasoning": evidence.reasoning,
            "url": evidence.url,
            "weight": evidence.weight
        }

    def _mock_response(self, prompt: str) -> str:
        """Generate mock evaluation response."""
        # Simple heuristic for mock
        prompt_lower = prompt.lower()

        if "positive" in prompt_lower or "likely" in prompt_lower or "supporting" in prompt_lower:
            supports = "YES"
            strength = 0.7
        elif "negative" in prompt_lower or "unlikely" in prompt_lower or "against" in prompt_lower:
            supports = "NO"
            strength = 0.6
        else:
            supports = "YES" if hash(prompt) % 2 == 0 else "NO"
            strength = 0.5

        return json.dumps({
            "supports": supports,
            "strength": strength,
            "evidence_type": "B",
            "reasoning": f"Mock evaluation - evidence appears to support {supports}"
        })


class AdaptiveResearcherAgent(ResearcherAgent):
    """
    Enhanced researcher with adaptive query generation.

    From AIA Forecaster: "Agents have discretion over search queries,
    adaptive conditioning on previous query results"
    """

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Execute adaptive research with query refinement.

        If initial results are sparse or one-sided, generate
        additional targeted queries.
        """
        # First pass - standard research
        result = super().run(input_data)

        if not result.success:
            return result

        data = result.data
        yes_count = len(data.get("yes_evidence", []))
        no_count = len(data.get("no_evidence", []))

        # Check for imbalance
        total = yes_count + no_count
        if total < 5 or (total > 0 and min(yes_count, no_count) / total < 0.2):
            # Generate additional queries for underrepresented side
            weak_side = "NO" if yes_count > no_count else "YES"
            self.log(f"Evidence imbalance detected, searching more for {weak_side}")

            additional = self._generate_additional_queries(
                input_data.get("plan", {}).get("original_question", ""),
                weak_side
            )

            # Execute additional searches
            for query in additional[:3]:
                results = search_news(query, num_results=3)
                for r in results:
                    evidence = self._evaluate_result(
                        r,
                        input_data.get("plan", {}).get("original_question", "")
                    )
                    if evidence.supports == weak_side:
                        data[f"{weak_side.lower()}_evidence"].append(
                            self._evidence_to_dict(evidence)
                        )

            result.data = data
            result.reasoning += f" (Added {len(additional)} adaptive queries for {weak_side})"

        return result

    def _generate_additional_queries(self, question: str, target_side: str) -> List[str]:
        """Generate additional queries targeting a specific side."""
        prompt = f"""Generate 3 search queries that would find evidence
{"SUPPORTING" if target_side == "YES" else "AGAINST"} this outcome:

Question: {question}

Respond as a JSON array of strings:
["query1", "query2", "query3"]"""

        response = self.call_llm(prompt)
        try:
            queries = json.loads(response)
            if isinstance(queries, list):
                return queries
        except:
            pass

        # Fallback
        if target_side == "YES":
            return [
                f"{question} evidence supporting",
                f"{question} reasons will happen",
                f"{question} positive indicators"
            ]
        else:
            return [
                f"{question} evidence against",
                f"{question} reasons will not happen",
                f"{question} obstacles barriers"
            ]
