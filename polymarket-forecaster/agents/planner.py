"""
Planner Agent - Decomposes questions into research pathways.

From Polyseer: "Decomposes questions into subclaims and research pathways"

This agent analyzes the market question and creates a structured
research plan that the Researcher agents will execute.
"""
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentResponse
from config.settings import DEFAULT_LLM


@dataclass
class ResearchPlan:
    """A structured research plan for a market question."""
    original_question: str
    category: str
    subclaims: List[str]  # Key claims that need to be verified
    yes_research_queries: List[str]  # Queries to find YES evidence
    no_research_queries: List[str]  # Queries to find NO evidence
    key_entities: List[str]  # People, orgs, events to research
    data_sources: List[str]  # Recommended sources (news, official, academic)
    time_sensitivity: str  # "high", "medium", "low"
    baseline_probability: float
    reasoning: str


class PlannerAgent(BaseAgent):
    """
    Planner Agent - First stage of the multi-agent pipeline.

    Responsibilities:
    1. Parse and understand the market question
    2. Identify key claims that need verification
    3. Generate balanced research queries (YES and NO)
    4. Recommend data sources
    5. Estimate baseline probability
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="PlannerAgent",
                model=DEFAULT_LLM,
                temperature=0.3,  # Lower temp for structured output
                system_prompt=self._get_system_prompt()
            )
        super().__init__(config)

    def _get_system_prompt(self) -> str:
        return """You are a research planning agent for prediction market forecasting.

Your job is to analyze market questions and create structured research plans.

For each question, you must:
1. Identify the core claim and any subclaims
2. Generate search queries that would find evidence FOR the outcome (YES)
3. Generate search queries that would find evidence AGAINST the outcome (NO)
4. Identify key entities (people, organizations, events) relevant to the question
5. Recommend data sources (news outlets, official sources, academic papers)
6. Assess time sensitivity (how quickly the situation might change)
7. Estimate a baseline probability based on historical patterns

IMPORTANT: You must research BOTH sides equally to avoid confirmation bias.

Always respond in valid JSON format."""

    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Create a research plan for a market question.

        Args:
            input_data: Must contain 'question', optionally 'description', 'category'

        Returns:
            AgentResponse with ResearchPlan data
        """
        question = input_data.get("question", "")
        description = input_data.get("description", "")
        category = input_data.get("category", "")

        if not question:
            return self.create_response(
                success=False,
                data={},
                reasoning="No question provided",
                confidence=0.0,
                error="Missing required field: question"
            )

        prompt = self._build_prompt(question, description, category)
        response = self.call_llm(prompt, self.config.system_prompt)
        plan_data = self.parse_json_response(response)

        # Build ResearchPlan
        plan = ResearchPlan(
            original_question=question,
            category=plan_data.get("category", category or "general"),
            subclaims=plan_data.get("subclaims", []),
            yes_research_queries=plan_data.get("yes_queries", []),
            no_research_queries=plan_data.get("no_queries", []),
            key_entities=plan_data.get("key_entities", []),
            data_sources=plan_data.get("data_sources", ["news", "official"]),
            time_sensitivity=plan_data.get("time_sensitivity", "medium"),
            baseline_probability=plan_data.get("baseline_probability", 0.5),
            reasoning=plan_data.get("reasoning", "")
        )

        return self.create_response(
            success=True,
            data={
                "plan": plan.__dict__,
                "raw_response": plan_data
            },
            reasoning=plan.reasoning,
            confidence=0.8
        )

    def _build_prompt(self, question: str, description: str, category: str) -> str:
        return f"""Analyze this prediction market question and create a research plan.

QUESTION: {question}

{f'DESCRIPTION: {description}' if description else ''}
{f'CATEGORY: {category}' if category else ''}

Create a comprehensive research plan. Respond in JSON format:

{{
    "category": "politics|economy|crypto|sports|tech|geopolitics|entertainment|general",
    "subclaims": [
        "First key claim that needs to be verified",
        "Second key claim",
        "..."
    ],
    "yes_queries": [
        "Search query that would find evidence FOR the outcome",
        "Another YES-supporting query",
        "Third YES query"
    ],
    "no_queries": [
        "Search query that would find evidence AGAINST the outcome",
        "Another NO-supporting query",
        "Third NO query"
    ],
    "key_entities": ["Entity1", "Entity2", "..."],
    "data_sources": ["reuters", "bloomberg", "official_gov", "academic"],
    "time_sensitivity": "high|medium|low",
    "baseline_probability": 0.5,
    "reasoning": "Brief explanation of the baseline probability"
}}

IMPORTANT:
- Generate at least 3 queries for each side (YES and NO)
- Queries should be specific and likely to return relevant news
- Consider multiple angles and subclaims
- Base probability on historical patterns in this category"""

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        # Extract question from prompt
        if "QUESTION:" in prompt:
            q_start = prompt.find("QUESTION:") + 9
            q_end = prompt.find("\n", q_start)
            question = prompt[q_start:q_end].strip()
        else:
            question = "Unknown question"

        # Detect category
        q_lower = question.lower()
        if any(w in q_lower for w in ["fed", "rate", "inflation", "recession"]):
            category = "economy"
            baseline = 0.35
        elif any(w in q_lower for w in ["bitcoin", "crypto", "btc", "eth"]):
            category = "crypto"
            baseline = 0.45
        elif any(w in q_lower for w in ["election", "president", "vote"]):
            category = "politics"
            baseline = 0.50
        else:
            category = "general"
            baseline = 0.50

        return json.dumps({
            "category": category,
            "subclaims": [
                "Primary claim from the question",
                "Supporting condition that must be true",
                "Timeline requirement"
            ],
            "yes_queries": [
                f"{question} likely to happen",
                f"{question} supporting evidence",
                f"{question} positive indicators"
            ],
            "no_queries": [
                f"{question} unlikely",
                f"{question} evidence against",
                f"{question} obstacles challenges"
            ],
            "key_entities": ["Entity1", "Entity2"],
            "data_sources": ["reuters", "bloomberg", "official"],
            "time_sensitivity": "medium",
            "baseline_probability": baseline,
            "reasoning": f"Based on historical patterns in {category} category"
        })
