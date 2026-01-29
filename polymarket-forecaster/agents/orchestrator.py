"""
Agent Orchestrator - Coordinates the multi-agent pipeline.

This is the main entry point for the multi-agent forecasting system.
It runs the full pipeline: Planner → Researcher → Critic → Analyst → Supervisor
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .planner import PlannerAgent
from .researcher import ResearcherAgent, AdaptiveResearcherAgent
from .critic import CriticAgent
from .analyst import AnalystAgent, BayesianAnalystAgent
from .supervisor import SupervisorAgent, ReconciliationSupervisor
from .base import AgentResponse

from core import Market, ForecastResult, Evidence
from core.edge_calculator import full_edge_analysis
from core.calibration import calibrate


@dataclass
class PipelineResult:
    """Complete result from the multi-agent pipeline."""
    market: Market
    final_probability: float
    calibrated_probability: float
    confidence_interval: Tuple[float, float]
    edge: float
    side: str
    should_trade: bool
    kelly_fraction: float
    agent_outputs: Dict[str, AgentResponse]
    iterations: int
    total_evidence: int
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class AgentOrchestrator:
    """
    Orchestrates the multi-agent forecasting pipeline.

    Pipeline:
    1. PlannerAgent: Decompose question into research plan
    2. ResearcherAgent: Gather bilateral evidence
    3. CriticAgent: Validate evidence, identify gaps
    4. (Optional) Loop back to Researcher if gaps found
    5. AnalystAgent: Aggregate into probability
    6. SupervisorAgent: Final review and reconciliation

    This architecture is based on:
    - Polyseer's 5-agent pipeline
    - AIA Forecaster's supervisor reconciliation
    - Outcome-RL's ensemble approach
    """

    def __init__(
        self,
        use_adaptive_researcher: bool = True,
        use_bayesian_analyst: bool = True,
        use_reconciliation_supervisor: bool = True,
        max_iterations: int = 3,
        verbose: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            use_adaptive_researcher: Use adaptive query generation
            use_bayesian_analyst: Use full Bayesian updating
            use_reconciliation_supervisor: Use conflict reconciliation
            max_iterations: Maximum research iterations
            verbose: Print progress
        """
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize agents
        self.planner = PlannerAgent()

        if use_adaptive_researcher:
            self.researcher = AdaptiveResearcherAgent()
        else:
            self.researcher = ResearcherAgent()

        self.critic = CriticAgent()

        if use_bayesian_analyst:
            self.analyst = BayesianAnalystAgent()
        else:
            self.analyst = AnalystAgent()

        if use_reconciliation_supervisor:
            self.supervisor = ReconciliationSupervisor()
        else:
            self.supervisor = SupervisorAgent()

    def log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(f"[Orchestrator] {message}")

    def forecast(self, market: Market) -> PipelineResult:
        """
        Run the full forecasting pipeline for a market.

        Args:
            market: Market to forecast

        Returns:
            PipelineResult with complete forecast
        """
        self.log(f"Starting forecast for: {market.question}")
        agent_outputs: Dict[str, AgentResponse] = {}
        reasoning: List[str] = []
        iteration = 0

        # Step 1: Planning
        self.log("Step 1: Planning research strategy...")
        plan_input = {
            "question": market.question,
            "description": market.description,
            "category": market.category
        }
        plan_response = self.planner.run(plan_input)
        agent_outputs["planner"] = plan_response

        if not plan_response.success:
            self.log(f"Planning failed: {plan_response.error}")
            return self._create_fallback_result(market, agent_outputs, reasoning)

        plan = plan_response.data.get("plan", {})
        baseline = plan.get("baseline_probability", 0.5)
        reasoning.append(f"Baseline probability: {baseline:.1%}")

        # Research loop
        research_data = {}
        critique_data = {}

        while iteration < self.max_iterations:
            iteration += 1
            self.log(f"Step 2.{iteration}: Conducting research (iteration {iteration})...")

            # Step 2: Research
            research_input = {
                "plan": plan,
                "question": market.question,
                "iteration": iteration
            }
            research_response = self.researcher.run(research_input)
            agent_outputs[f"researcher_{iteration}"] = research_response

            if not research_response.success:
                self.log(f"Research failed: {research_response.error}")
                break

            research_data = research_response.data
            yes_count = len(research_data.get("yes_evidence", []))
            no_count = len(research_data.get("no_evidence", []))
            reasoning.append(f"Research iteration {iteration}: {yes_count} YES, {no_count} NO evidence")

            # Step 3: Critique
            self.log(f"Step 3.{iteration}: Validating evidence...")
            critique_input = {
                "yes_evidence": research_data.get("yes_evidence", []),
                "no_evidence": research_data.get("no_evidence", []),
                "neutral_evidence": research_data.get("neutral_evidence", []),
                "question": market.question
            }
            critique_response = self.critic.run(critique_input)
            agent_outputs[f"critic_{iteration}"] = critique_response

            critique_data = critique_response.data
            quality = critique_data.get("quality_score", 0)
            recommendation = critique_data.get("recommendation", "proceed")

            reasoning.append(f"Critique: quality={quality:.2f}, recommendation={recommendation}")

            # Check if we should continue researching
            if recommendation == "proceed" or iteration >= self.max_iterations:
                break

            # Update plan with additional queries from critic
            additional_queries = critique_data.get("additional_queries", [])
            if additional_queries:
                plan["yes_research_queries"] = additional_queries[:len(additional_queries)//2]
                plan["no_research_queries"] = additional_queries[len(additional_queries)//2:]
                self.log(f"Adding {len(additional_queries)} queries for next iteration")
            else:
                break

        # Step 4: Analysis
        self.log("Step 4: Analyzing evidence...")
        validated_evidence = critique_data.get("validated_evidence", [])
        if not validated_evidence:
            # Fall back to all evidence if none validated
            validated_evidence = (
                research_data.get("yes_evidence", []) +
                research_data.get("no_evidence", [])
            )

        analysis_input = {
            "validated_evidence": validated_evidence,
            "baseline_probability": baseline,
            "question": market.question
        }
        analysis_response = self.analyst.run(analysis_input)
        agent_outputs["analyst"] = analysis_response

        analysis_data = analysis_response.data
        raw_prob = analysis_data.get("raw_probability", baseline)
        calibrated_prob = analysis_data.get("calibrated_probability", raw_prob)
        ci = analysis_data.get("confidence_interval", (raw_prob - 0.1, raw_prob + 0.1))

        reasoning.append(f"Analysis: raw={raw_prob:.1%}, calibrated={calibrated_prob:.1%}")

        # Step 5: Supervisor review
        self.log("Step 5: Supervisor review...")
        supervisor_input = {
            "plan": plan,
            "research": research_data,
            "critique": critique_data,
            "analysis": analysis_data,
            "question": market.question,
            "iteration": iteration
        }
        supervisor_response = self.supervisor.run(supervisor_input)
        agent_outputs["supervisor"] = supervisor_response

        supervisor_data = supervisor_response.data
        final_prob = supervisor_data.get("final_probability", calibrated_prob)
        confidence = supervisor_data.get("confidence", 0.5)

        reasoning.append(f"Supervisor: final={final_prob:.1%}, confidence={confidence:.1%}")

        # Step 6: Calculate trading decision
        self.log("Step 6: Calculating edge and position sizing...")
        edge_analysis = full_edge_analysis(
            ai_prob=final_prob,
            market_price=market.price
        )

        reasoning.append(f"Edge: {edge_analysis.edge:+.1%} on {edge_analysis.side}")

        # Build result
        result = PipelineResult(
            market=market,
            final_probability=final_prob,
            calibrated_probability=calibrated_prob,
            confidence_interval=ci if isinstance(ci, tuple) else tuple(ci),
            edge=edge_analysis.edge,
            side=edge_analysis.side,
            should_trade=edge_analysis.should_trade,
            kelly_fraction=edge_analysis.kelly_fraction,
            agent_outputs=agent_outputs,
            iterations=iteration,
            total_evidence=len(validated_evidence),
            reasoning=reasoning
        )

        self.log(f"Forecast complete: {final_prob:.1%} ({edge_analysis.side})")
        return result

    def _create_fallback_result(self, market: Market,
                                 agent_outputs: Dict[str, AgentResponse],
                                 reasoning: List[str]) -> PipelineResult:
        """Create a fallback result when pipeline fails."""
        return PipelineResult(
            market=market,
            final_probability=0.5,
            calibrated_probability=0.5,
            confidence_interval=(0.3, 0.7),
            edge=0.0,
            side="NONE",
            should_trade=False,
            kelly_fraction=0.0,
            agent_outputs=agent_outputs,
            iterations=0,
            total_evidence=0,
            reasoning=reasoning + ["Pipeline failed, returning neutral forecast"]
        )

    def to_forecast_result(self, pipeline_result: PipelineResult) -> ForecastResult:
        """Convert PipelineResult to ForecastResult for compatibility."""
        # Extract evidence from agent outputs
        researcher_output = None
        for key, output in pipeline_result.agent_outputs.items():
            if key.startswith("researcher"):
                researcher_output = output
                break

        pro_evidence = []
        con_evidence = []

        if researcher_output and researcher_output.data:
            for e in researcher_output.data.get("yes_evidence", []):
                pro_evidence.append(Evidence(
                    source=e.get("source", "Unknown"),
                    content=e.get("content", ""),
                    supports="YES",
                    strength=e.get("strength", 0.5),
                    evidence_type=e.get("evidence_type", "C"),
                    reasoning=e.get("reasoning", "")
                ))
            for e in researcher_output.data.get("no_evidence", []):
                con_evidence.append(Evidence(
                    source=e.get("source", "Unknown"),
                    content=e.get("content", ""),
                    supports="NO",
                    strength=e.get("strength", 0.5),
                    evidence_type=e.get("evidence_type", "C"),
                    reasoning=e.get("reasoning", "")
                ))

        return ForecastResult(
            probability=pipeline_result.final_probability,
            confidence_interval=pipeline_result.confidence_interval,
            edge=pipeline_result.edge,
            side=pipeline_result.side,
            should_trade=pipeline_result.should_trade,
            kelly_fraction=pipeline_result.kelly_fraction,
            evidence_summary="\n".join(pipeline_result.reasoning),
            reasoning=pipeline_result.reasoning,
            pro_evidence=pro_evidence,
            con_evidence=con_evidence
        )


def run_multi_agent_forecast(market: Market, verbose: bool = True) -> ForecastResult:
    """
    Convenience function to run multi-agent forecast.

    Args:
        market: Market to forecast
        verbose: Print progress

    Returns:
        ForecastResult
    """
    orchestrator = AgentOrchestrator(verbose=verbose)
    pipeline_result = orchestrator.forecast(market)
    return orchestrator.to_forecast_result(pipeline_result)
