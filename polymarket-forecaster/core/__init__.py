"""
Core module for Polymarket Forecaster.
Contains data classes and core forecasting logic.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime


@dataclass
class Market:
    """Represents a Polymarket prediction market."""
    id: str
    question: str
    price: float  # Current YES price (0-1)
    volume: float
    category: str
    end_date: str
    description: Optional[str] = None
    outcome_prices: Optional[Dict[str, float]] = None
    liquidity: Optional[float] = None

    @property
    def no_price(self) -> float:
        """Price of NO shares."""
        return 1.0 - self.price


@dataclass
class MarketAnalysis:
    """Structured analysis of a market question."""
    original_question: str
    category: str
    subcategory: Optional[str]
    key_entities: List[str]
    resolution_criteria: str
    time_horizon: str  # "immediate", "short_term", "medium_term", "long_term"
    baseline_probability: float
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class Evidence:
    """A piece of evidence for or against a market outcome."""
    source: str
    content: str
    supports: str  # "YES", "NO", "NEUTRAL"
    strength: float  # 0.0-1.0
    evidence_type: str  # "A" (primary), "B" (high-quality), "C" (standard), "D" (weak)
    reasoning: str
    url: Optional[str] = None
    timestamp: Optional[datetime] = None

    @property
    def weight(self) -> float:
        """Calculate weighted strength based on evidence type."""
        type_weights = {'A': 1.0, 'B': 0.8, 'C': 0.5, 'D': 0.2}
        return self.strength * type_weights.get(self.evidence_type, 0.3)


@dataclass
class MCResult:
    """Result from Monte Carlo simulation."""
    median_probability: float
    mean_probability: float
    ci_lower: float  # 5th percentile
    ci_upper: float  # 95th percentile
    simulations: int
    all_probabilities: Optional[List[float]] = None

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Return 90% confidence interval."""
        return (self.ci_lower, self.ci_upper)


@dataclass
class ForecastResult:
    """Complete forecast result for a market."""
    probability: float
    confidence_interval: Tuple[float, float]
    edge: float
    side: str  # "YES" or "NO"
    should_trade: bool
    kelly_fraction: float
    evidence_summary: str
    reasoning: List[str]
    pro_evidence: List[Evidence] = field(default_factory=list)
    con_evidence: List[Evidence] = field(default_factory=list)
    market_analysis: Optional[MarketAnalysis] = None
    mc_result: Optional[MCResult] = None


@dataclass
class TradeDecision:
    """A trading decision with sizing."""
    market_id: str
    side: str  # "YES" or "NO"
    amount: float
    expected_value: float
    ai_probability: float
    market_price: float
    kelly_fraction: float
    confidence: str


@dataclass
class Position:
    """A position in a market."""
    market_id: str
    side: str
    shares: float
    avg_price: float
    current_price: float
    unrealized_pnl: float

    @property
    def value(self) -> float:
        """Current position value."""
        return self.shares * self.current_price


@dataclass
class Portfolio:
    """Portfolio of positions."""
    positions: List[Position]
    cash: float
    total_value: float
    total_pnl: float

    @property
    def exposure(self) -> float:
        """Total market exposure."""
        return sum(p.value for p in self.positions)


# Export all classes
__all__ = [
    'Market',
    'MarketAnalysis',
    'Evidence',
    'MCResult',
    'ForecastResult',
    'TradeDecision',
    'Position',
    'Portfolio',
]
