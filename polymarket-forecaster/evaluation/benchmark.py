"""
Benchmark Evaluation - Compare forecasts against standards.

Compatible with:
- ForecastBench (forecastbench.org)
- PrediBench (predibench.com)
- Prediction Arena

From ForecastBench:
- Balanced new rounds every 2 weeks
- 1,000 questions for LLM forecasters
- Nightly validation, resolution, leaderboard updates
- Difficulty-adjusted Brier score
"""
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .metrics import (
    compute_all_metrics,
    brier_score,
    expected_calibration_error,
    roi_simulation,
    difficulty_adjusted_brier,
    ForecastMetrics
)


@dataclass
class BenchmarkQuestion:
    """A single benchmark question."""
    id: str
    question: str
    category: str
    resolution_date: str
    outcome: Optional[int]  # None if unresolved
    market_price: Optional[float]  # Price at question creation
    difficulty: float = 0.5  # Estimated difficulty (0-1)


@dataclass
class ForecastRecord:
    """A single forecast record."""
    question_id: str
    prediction: float
    timestamp: datetime
    model_name: str
    confidence: float


@dataclass
class BenchmarkResult:
    """Result of benchmark evaluation."""
    model_name: str
    metrics: ForecastMetrics
    difficulty_adjusted_brier: float
    roi_metrics: Dict[str, float]
    n_questions: int
    n_resolved: int
    category_breakdown: Dict[str, ForecastMetrics]
    comparison_to_baseline: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class Benchmark:
    """
    Base benchmark class for evaluating forecasters.
    """

    def __init__(self, name: str, data_dir: Optional[Path] = None):
        self.name = name
        self.data_dir = data_dir or Path("./data/benchmarks")
        self.questions: List[BenchmarkQuestion] = []
        self.forecasts: List[ForecastRecord] = []

    def add_question(self, question: BenchmarkQuestion):
        """Add a benchmark question."""
        self.questions.append(question)

    def add_forecast(self, forecast: ForecastRecord):
        """Add a forecast record."""
        self.forecasts.append(forecast)

    def get_resolved_pairs(self, model_name: Optional[str] = None
                           ) -> Tuple[List[float], List[int], List[float]]:
        """
        Get (predictions, outcomes, market_prices) for resolved questions.

        Returns:
            Tuple of (predictions, outcomes, market_prices)
        """
        predictions = []
        outcomes = []
        market_prices = []

        resolved_questions = {q.id: q for q in self.questions if q.outcome is not None}

        for forecast in self.forecasts:
            if model_name and forecast.model_name != model_name:
                continue

            if forecast.question_id in resolved_questions:
                q = resolved_questions[forecast.question_id]
                predictions.append(forecast.prediction)
                outcomes.append(q.outcome)
                market_prices.append(q.market_price or 0.5)

        return predictions, outcomes, market_prices

    def evaluate(self, model_name: str) -> BenchmarkResult:
        """
        Evaluate a model's forecasts.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            BenchmarkResult with all metrics
        """
        predictions, outcomes, market_prices = self.get_resolved_pairs(model_name)

        if not predictions:
            return BenchmarkResult(
                model_name=model_name,
                metrics=ForecastMetrics(0, 0, 0, 0, 0, 0, 0, 1.0),
                difficulty_adjusted_brier=0,
                roi_metrics={},
                n_questions=len(self.questions),
                n_resolved=0,
                category_breakdown={},
                comparison_to_baseline={}
            )

        # Compute core metrics
        metrics = compute_all_metrics(predictions, outcomes)

        # Difficulty-adjusted Brier
        adj_brier = difficulty_adjusted_brier(predictions, outcomes, market_prices)

        # ROI simulation
        roi = roi_simulation(predictions, outcomes, market_prices)

        # Category breakdown
        category_breakdown = self._compute_category_breakdown(model_name)

        # Comparison to baseline (market prices)
        baseline_brier = brier_score(market_prices, outcomes)
        comparison = {
            "baseline_brier": baseline_brier,
            "model_brier": metrics.brier_score,
            "improvement": baseline_brier - metrics.brier_score,
            "improvement_percent": ((baseline_brier - metrics.brier_score) / baseline_brier * 100
                                   if baseline_brier > 0 else 0)
        }

        return BenchmarkResult(
            model_name=model_name,
            metrics=metrics,
            difficulty_adjusted_brier=adj_brier,
            roi_metrics=roi,
            n_questions=len(self.questions),
            n_resolved=len(predictions),
            category_breakdown=category_breakdown,
            comparison_to_baseline=comparison
        )

    def _compute_category_breakdown(self, model_name: str) -> Dict[str, ForecastMetrics]:
        """Compute metrics by category."""
        categories: Dict[str, Tuple[List[float], List[int]]] = {}

        resolved_questions = {q.id: q for q in self.questions if q.outcome is not None}

        for forecast in self.forecasts:
            if forecast.model_name != model_name:
                continue

            if forecast.question_id in resolved_questions:
                q = resolved_questions[forecast.question_id]
                cat = q.category

                if cat not in categories:
                    categories[cat] = ([], [])

                categories[cat][0].append(forecast.prediction)
                categories[cat][1].append(q.outcome)

        result = {}
        for cat, (preds, outs) in categories.items():
            result[cat] = compute_all_metrics(preds, outs)

        return result

    def save(self, filepath: Optional[Path] = None):
        """Save benchmark data to file."""
        filepath = filepath or self.data_dir / f"{self.name}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "category": q.category,
                    "resolution_date": q.resolution_date,
                    "outcome": q.outcome,
                    "market_price": q.market_price,
                    "difficulty": q.difficulty
                }
                for q in self.questions
            ],
            "forecasts": [
                {
                    "question_id": f.question_id,
                    "prediction": f.prediction,
                    "timestamp": f.timestamp.isoformat(),
                    "model_name": f.model_name,
                    "confidence": f.confidence
                }
                for f in self.forecasts
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[Path] = None):
        """Load benchmark data from file."""
        filepath = filepath or self.data_dir / f"{self.name}.json"

        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        self.name = data["name"]

        self.questions = [
            BenchmarkQuestion(
                id=q["id"],
                question=q["question"],
                category=q["category"],
                resolution_date=q["resolution_date"],
                outcome=q.get("outcome"),
                market_price=q.get("market_price"),
                difficulty=q.get("difficulty", 0.5)
            )
            for q in data.get("questions", [])
        ]

        self.forecasts = [
            ForecastRecord(
                question_id=f["question_id"],
                prediction=f["prediction"],
                timestamp=datetime.fromisoformat(f["timestamp"]),
                model_name=f["model_name"],
                confidence=f["confidence"]
            )
            for f in data.get("forecasts", [])
        ]


class ForecastBenchEvaluator(Benchmark):
    """
    Evaluator compatible with ForecastBench standards.

    From ForecastBench:
    - Difficulty-adjusted Brier Score for fair comparison
    - Equal-weighted average of dataset and market scores
    - Rankings stabilize within 50 days
    """

    def __init__(self):
        super().__init__("forecastbench")
        self.leaderboard: Dict[str, BenchmarkResult] = {}

    def evaluate_for_leaderboard(self, model_name: str) -> float:
        """
        Evaluate and return leaderboard score.

        Returns:
            Composite score (lower is better)
        """
        result = self.evaluate(model_name)
        self.leaderboard[model_name] = result

        # ForecastBench uses equal-weighted average
        # of difficulty-adjusted Brier and ECE
        composite = (result.difficulty_adjusted_brier + result.metrics.ece) / 2

        return composite

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get sorted leaderboard."""
        scores = []
        for model_name, result in self.leaderboard.items():
            score = (result.difficulty_adjusted_brier + result.metrics.ece) / 2
            scores.append((model_name, score))

        return sorted(scores, key=lambda x: x[1])


class BacktestEvaluator(Benchmark):
    """
    Evaluator for backtesting on historical Polymarket data.
    """

    def __init__(self):
        super().__init__("backtest")
        self.trade_history: List[Dict] = []

    def add_historical_market(
        self,
        market_id: str,
        question: str,
        category: str,
        resolution_date: str,
        outcome: int,
        prices_over_time: List[Tuple[str, float]]  # [(timestamp, price), ...]
    ):
        """Add a historical market for backtesting."""
        # Use final price before resolution
        final_price = prices_over_time[-1][1] if prices_over_time else 0.5

        self.add_question(BenchmarkQuestion(
            id=market_id,
            question=question,
            category=category,
            resolution_date=resolution_date,
            outcome=outcome,
            market_price=final_price
        ))

    def simulate_trading(
        self,
        model_name: str,
        bankroll: float = 10000,
        min_edge: float = 0.05
    ) -> Dict[str, float]:
        """
        Simulate trading based on forecasts.

        Returns:
            Trading performance metrics
        """
        predictions, outcomes, market_prices = self.get_resolved_pairs(model_name)

        if not predictions:
            return {"roi": 0, "total_bets": 0}

        return roi_simulation(
            predictions=predictions,
            outcomes=outcomes,
            market_prices=market_prices,
            bankroll=bankroll,
            min_edge=min_edge
        )

    def generate_report(self, model_name: str) -> str:
        """Generate a text report of backtest results."""
        result = self.evaluate(model_name)
        roi = self.simulate_trading(model_name)

        lines = [
            f"=== Backtest Report: {model_name} ===",
            "",
            f"Questions: {result.n_questions} total, {result.n_resolved} resolved",
            "",
            "--- Accuracy Metrics ---",
            f"Brier Score: {result.metrics.brier_score:.4f}",
            f"ECE: {result.metrics.ece:.4f}",
            f"Accuracy: {result.metrics.accuracy:.1%}",
            f"Log Loss: {result.metrics.log_loss:.4f}",
            "",
            "--- Calibration ---",
            f"Mean Confidence: {result.metrics.mean_confidence:.1%}",
            f"Calibration Slope: {result.metrics.calibration_slope:.2f}",
            "",
            "--- vs Baseline (Market) ---",
            f"Baseline Brier: {result.comparison_to_baseline.get('baseline_brier', 0):.4f}",
            f"Improvement: {result.comparison_to_baseline.get('improvement_percent', 0):.1f}%",
            "",
            "--- Trading Simulation ---",
            f"ROI: {roi.get('roi_percent', 0):.1f}%",
            f"Total Bets: {roi.get('total_bets', 0)}",
            f"Win Rate: {roi.get('win_rate', 0):.1%}",
            f"Final Bankroll: ${roi.get('final_bankroll', 10000):,.2f}",
        ]

        if result.category_breakdown:
            lines.extend(["", "--- By Category ---"])
            for cat, metrics in result.category_breakdown.items():
                lines.append(f"{cat}: Brier={metrics.brier_score:.4f}, "
                            f"ECE={metrics.ece:.4f}, n={metrics.n_samples}")

        return "\n".join(lines)


def run_backtest(
    forecaster_func,  # Function that takes question and returns probability
    historical_markets: List[Dict],
    model_name: str = "test_model"
) -> BenchmarkResult:
    """
    Convenience function to run a backtest.

    Args:
        forecaster_func: Function(question: str) -> float
        historical_markets: List of dicts with question, outcome, market_price
        model_name: Name for the model

    Returns:
        BenchmarkResult
    """
    evaluator = BacktestEvaluator()

    for i, market in enumerate(historical_markets):
        evaluator.add_question(BenchmarkQuestion(
            id=str(i),
            question=market["question"],
            category=market.get("category", "general"),
            resolution_date=market.get("resolution_date", ""),
            outcome=market["outcome"],
            market_price=market.get("market_price", 0.5)
        ))

        # Generate forecast
        prediction = forecaster_func(market["question"])

        evaluator.add_forecast(ForecastRecord(
            question_id=str(i),
            prediction=prediction,
            timestamp=datetime.now(),
            model_name=model_name,
            confidence=0.7
        ))

    return evaluator.evaluate(model_name)
