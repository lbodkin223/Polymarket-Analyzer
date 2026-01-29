"""
Multi-Agent Forecasting System

Architecture based on Polyseer and AIA Forecaster:

Polyseer 5-agent pipeline:
    Planner → Researcher → Critic → Analyst → Reporter

AIA Forecaster:
    Multiple Search Agents → Supervisor → Calibration

Our hybrid implementation:
    PlannerAgent: Decomposes question into research pathways
    ResearcherAgent: Gathers bilateral evidence (YES/NO)
    CriticAgent: Validates evidence, identifies gaps
    AnalystAgent: Bayesian probability aggregation
    SupervisorAgent: Reconciles conflicts, triggers re-research
"""

from .base import BaseAgent, AgentConfig, AgentResponse
from .planner import PlannerAgent
from .researcher import ResearcherAgent
from .critic import CriticAgent
from .analyst import AnalystAgent
from .supervisor import SupervisorAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'AgentResponse',
    'PlannerAgent',
    'ResearcherAgent',
    'CriticAgent',
    'AnalystAgent',
    'SupervisorAgent',
    'AgentOrchestrator',
]
