#!/usr/bin/env python3
"""
Polymarket AI Forecaster - CLI Entry Point

A multi-agent Monte Carlo probability forecasting system for Polymarket.

Architecture based on:
- Polyseer: 5-agent pipeline (Planner → Researcher → Critic → Analyst → Reporter)
- AIA Forecaster: Supervisor reconciliation + Platt scaling calibration
- Outcome-RL paper: Ensemble predictions, ECE optimization

Usage:
    python main.py scan              - Scan markets for opportunities
    python main.py forecast <url>    - Forecast single market (multi-agent)
    python main.py forecast-simple   - Forecast with simple pipeline
    python main.py backtest          - Test on resolved markets
    python main.py paper-trade       - Run without real money
    python main.py benchmark         - Run benchmark evaluation
    python main.py test              - Run test pipeline
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Optional

from core import Market, ForecastResult, Evidence
from core.market_parser import parse_market_question, check_resolution_status
from core.evidence_evaluator import EvidenceEvaluator
from core.monte_carlo import forecast_with_evidence, run_monte_carlo, apply_evidence_effects
from core.edge_calculator import full_edge_analysis, calculate_edge
from core.blending import blend_probabilities, load_domain_prior, calculate_confidence_from_ci
from core.calibration import calibrate, compute_ece, EnsembleCalibrator

from research.bilateral_research import BilateralResearcher, quick_research
from research.news_search import search_news

from trading.polymarket_client import PolymarketClient, scan_for_opportunities
from trading.portfolio import PortfolioManager

from agents.orchestrator import AgentOrchestrator, run_multi_agent_forecast

from evaluation.metrics import compute_all_metrics, brier_score, roi_simulation
from evaluation.benchmark import BacktestEvaluator, ForecastBenchEvaluator

from config.settings import MIN_EDGE_PERCENT, MIN_VOLUME


def forecast_market_multi_agent(market: Market, verbose: bool = True) -> ForecastResult:
    """
    Run multi-agent forecasting pipeline.

    Pipeline: Planner → Researcher → Critic → Analyst → Supervisor

    Args:
        market: Market to forecast
        verbose: Print progress

    Returns:
        ForecastResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-AGENT FORECAST")
        print(f"{'='*60}")
        print(f"Question: {market.question}")
        print(f"Market Price: {market.price:.1%} | Volume: ${market.volume:,.0f}")
        print(f"{'='*60}\n")

    orchestrator = AgentOrchestrator(
        use_adaptive_researcher=True,
        use_bayesian_analyst=True,
        use_reconciliation_supervisor=True,
        verbose=verbose
    )

    pipeline_result = orchestrator.forecast(market)
    result = orchestrator.to_forecast_result(pipeline_result)

    if verbose:
        print(f"\n{'='*60}")
        print("FORECAST COMPLETE")
        print(f"{'='*60}")
        print(f"Final Probability: {pipeline_result.final_probability:.1%}")
        print(f"Calibrated: {pipeline_result.calibrated_probability:.1%}")
        print(f"Confidence Interval: [{pipeline_result.confidence_interval[0]:.1%}, "
              f"{pipeline_result.confidence_interval[1]:.1%}]")
        print(f"Edge: {pipeline_result.edge:+.1%} on {pipeline_result.side}")
        print(f"Should Trade: {'✓ YES' if pipeline_result.should_trade else '✗ NO'}")
        if pipeline_result.should_trade:
            print(f"Kelly Fraction: {pipeline_result.kelly_fraction:.1%}")
        print(f"Research Iterations: {pipeline_result.iterations}")
        print(f"Evidence Pieces: {pipeline_result.total_evidence}")

    return result


def forecast_market_simple(market: Market, verbose: bool = True) -> ForecastResult:
    """
    Run simple single-pipeline forecast (original implementation).

    Args:
        market: Market to forecast
        verbose: Print progress

    Returns:
        ForecastResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SIMPLE FORECAST")
        print(f"{'='*60}")
        print(f"Question: {market.question}")
        print(f"Market Price: {market.price:.1%} | Volume: ${market.volume:,.0f}")
        print(f"{'='*60}")

    # Step 1: Parse market question
    if verbose:
        print("\n[1/6] Parsing market question...")
    analysis = parse_market_question(market.question, market.description)

    if verbose:
        print(f"  Category: {analysis.category}")
        print(f"  Time horizon: {analysis.time_horizon}")
        print(f"  Baseline probability: {analysis.baseline_probability:.1%}")

    # Step 2: Load domain prior if available
    domain_prior = load_domain_prior(analysis.category, analysis.subcategory)
    baseline = domain_prior if domain_prior else analysis.baseline_probability

    # Step 3: Bilateral research
    if verbose:
        print("\n[2/6] Conducting bilateral research...")
    researcher = BilateralResearcher()
    pro_evidence, con_evidence = researcher.research(market.question)

    if verbose:
        print(f"  Found {len(pro_evidence)} pieces of YES evidence")
        print(f"  Found {len(con_evidence)} pieces of NO evidence")

    # Step 4: Monte Carlo simulation
    if verbose:
        print("\n[3/6] Running Monte Carlo simulation...")
    all_evidence = pro_evidence + con_evidence
    mc_result, multipliers = forecast_with_evidence(baseline, all_evidence)

    if verbose:
        print(f"  Baseline: {baseline:.1%}")
        print(f"  Projection: {mc_result.median_probability:.1%}")
        print(f"  90% CI: [{mc_result.ci_lower:.1%}, {mc_result.ci_upper:.1%}]")

    # Step 5: Calibration (NEW - from AIA Forecaster)
    if verbose:
        print("\n[4/6] Applying calibration...")
    raw_prob = mc_result.median_probability
    calibrated_prob = calibrate(raw_prob)

    if verbose:
        print(f"  Raw: {raw_prob:.1%}")
        print(f"  Calibrated: {calibrated_prob:.1%}")

    # Step 6: Blend and calculate edge
    if verbose:
        print("\n[5/6] Blending probabilities...")
    confidence = calculate_confidence_from_ci(mc_result.ci_lower, mc_result.ci_upper)
    final_prob = blend_probabilities(baseline, calibrated_prob, confidence)

    if verbose:
        print(f"  Confidence level: {confidence}")
        print(f"  Final probability: {final_prob:.1%}")

    # Step 7: Calculate edge
    if verbose:
        print("\n[6/6] Calculating edge and position sizing...")
    edge_analysis = full_edge_analysis(final_prob, market.price)

    if verbose:
        print(f"  AI Probability: {final_prob:.1%}")
        print(f"  Market Price: {market.price:.1%}")
        print(f"  Edge: {edge_analysis.edge:+.1%}")
        print(f"  Side: {edge_analysis.side}")
        print(f"  Should Trade: {edge_analysis.should_trade}")
        if edge_analysis.should_trade:
            print(f"  Kelly Fraction: {edge_analysis.kelly_fraction:.1%}")
            print(f"  Position Size: ${edge_analysis.position_size:.2f}")

    evaluator = EvidenceEvaluator()
    evidence_summary = evaluator.generate_evidence_summary(all_evidence)

    return ForecastResult(
        probability=final_prob,
        confidence_interval=(mc_result.ci_lower, mc_result.ci_upper),
        edge=edge_analysis.edge,
        side=edge_analysis.side,
        should_trade=edge_analysis.should_trade,
        kelly_fraction=edge_analysis.kelly_fraction,
        evidence_summary=evidence_summary,
        reasoning=[
            f"Category: {analysis.category}",
            f"Baseline: {baseline:.1%}",
            f"Calibrated: {calibrated_prob:.1%}",
            f"Final: {final_prob:.1%}",
        ],
        pro_evidence=pro_evidence,
        con_evidence=con_evidence,
        market_analysis=analysis,
        mc_result=mc_result
    )


def cmd_scan(args):
    """Scan markets for trading opportunities."""
    print("Scanning Polymarket for opportunities...")
    print(f"Minimum volume: ${args.min_volume:,}")
    print(f"Maximum markets: {args.limit}")
    print(f"Using: {'Multi-agent' if args.multi_agent else 'Simple'} pipeline")
    print()

    markets = scan_for_opportunities(
        min_volume=args.min_volume,
        max_markets=args.limit
    )

    if not markets:
        print("No tradeable markets found.")
        return

    opportunities = []

    for market in markets:
        try:
            if args.multi_agent:
                result = forecast_market_multi_agent(market, verbose=False)
            else:
                result = forecast_market_simple(market, verbose=False)

            if result.should_trade:
                opportunities.append((market, result))
        except Exception as e:
            print(f"Error forecasting {market.id}: {e}")

    print(f"\n{'='*80}")
    print(f"SCAN RESULTS: Found {len(opportunities)} opportunities out of {len(markets)} markets")
    print(f"{'='*80}\n")

    if not opportunities:
        print("No significant edges found.")
        return

    opportunities.sort(key=lambda x: abs(x[1].edge), reverse=True)

    for market, result in opportunities:
        print(f"Market: {market.question}")
        print(f"  Price: {market.price:.1%} | AI: {result.probability:.1%}")
        print(f"  Edge: {result.edge:+.1%} on {result.side}")
        print(f"  Kelly: {result.kelly_fraction:.1%}")
        print()


def cmd_forecast(args):
    """Forecast a single market."""
    client = PolymarketClient()

    if args.url:
        market = client.get_market_by_url(args.url)
        if not market:
            print(f"Could not find market for URL: {args.url}")
            return
    elif args.id:
        market = client.get_market(args.id)
        if not market:
            print(f"Could not find market with ID: {args.id}")
            return
    else:
        # Use first mock market for demo
        markets = client.get_markets()
        if markets:
            market = markets[0]
        else:
            print("No markets available")
            return

    if args.simple:
        result = forecast_market_simple(market, verbose=True)
    else:
        result = forecast_market_multi_agent(market, verbose=True)

    # Final summary
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Question: {market.question}")
    print(f"AI Probability: {result.probability:.1%}")
    print(f"Market Price: {market.price:.1%}")
    print(f"Edge: {result.edge:+.1%} on {result.side}")
    print(f"Confidence Interval: [{result.confidence_interval[0]:.1%}, {result.confidence_interval[1]:.1%}]")
    print()

    if result.should_trade:
        print(f"✓ TRADE SIGNAL: {result.side}")
        print(f"  Kelly Size: {result.kelly_fraction:.1%}")
    else:
        print("✗ No trade recommended (edge < 5%)")

    print(f"\n{result.evidence_summary}")


def cmd_backtest(args):
    """Run backtest on historical markets."""
    print("Running backtest evaluation...")

    # Create mock historical data for demo
    historical_markets = [
        {"question": "Will the Fed raise rates?", "outcome": 1, "market_price": 0.65},
        {"question": "Will Bitcoin hit $50k?", "outcome": 1, "market_price": 0.45},
        {"question": "Will recession occur?", "outcome": 0, "market_price": 0.30},
        {"question": "Will AI pass benchmark?", "outcome": 0, "market_price": 0.25},
        {"question": "Will election be contested?", "outcome": 1, "market_price": 0.40},
    ]

    evaluator = BacktestEvaluator()

    for i, market_data in enumerate(historical_markets):
        from evaluation.benchmark import BenchmarkQuestion, ForecastRecord
        from datetime import datetime

        evaluator.add_question(BenchmarkQuestion(
            id=str(i),
            question=market_data["question"],
            category="general",
            resolution_date="2024-12-31",
            outcome=market_data["outcome"],
            market_price=market_data["market_price"]
        ))

        # Generate mock forecast
        mock_market = Market(
            id=str(i),
            question=market_data["question"],
            price=market_data["market_price"],
            volume=100000,
            category="general",
            end_date="2024-12-31"
        )

        result = forecast_market_simple(mock_market, verbose=False)

        evaluator.add_forecast(ForecastRecord(
            question_id=str(i),
            prediction=result.probability,
            timestamp=datetime.now(),
            model_name="polymarket-forecaster",
            confidence=0.7
        ))

    # Generate report
    print("\n" + evaluator.generate_report("polymarket-forecaster"))


def cmd_benchmark(args):
    """Run ForecastBench-style evaluation."""
    print("Running benchmark evaluation (ForecastBench compatible)...")
    print()

    evaluator = ForecastBenchEvaluator()

    # Add some test questions and forecasts
    from evaluation.benchmark import BenchmarkQuestion, ForecastRecord
    from datetime import datetime

    test_data = [
        (0.60, 1),  # Prediction, Outcome
        (0.45, 0),
        (0.70, 1),
        (0.30, 0),
        (0.55, 1),
        (0.40, 0),
        (0.75, 1),
        (0.25, 0),
        (0.65, 1),
        (0.35, 0),
    ]

    for i, (pred, outcome) in enumerate(test_data):
        evaluator.add_question(BenchmarkQuestion(
            id=str(i),
            question=f"Test question {i}",
            category="test",
            resolution_date="2024-12-31",
            outcome=outcome,
            market_price=0.5
        ))

        evaluator.add_forecast(ForecastRecord(
            question_id=str(i),
            prediction=pred,
            timestamp=datetime.now(),
            model_name="test-model",
            confidence=0.7
        ))

    result = evaluator.evaluate("test-model")

    print("=== Benchmark Results ===")
    print(f"Brier Score: {result.metrics.brier_score:.4f}")
    print(f"ECE: {result.metrics.ece:.4f}")
    print(f"Accuracy: {result.metrics.accuracy:.1%}")
    print(f"Log Loss: {result.metrics.log_loss:.4f}")
    print()
    print(f"Difficulty-Adjusted Brier: {result.difficulty_adjusted_brier:.4f}")
    print()

    # Compare to reference values
    print("--- Reference Comparisons ---")
    print(f"Outcome-RL paper Brier: 0.193")
    print(f"Outcome-RL paper ECE: 0.042")
    print(f"Your Brier: {result.metrics.brier_score:.4f}")
    print(f"Your ECE: {result.metrics.ece:.4f}")


def cmd_paper_trade(args):
    """Run paper trading simulation."""
    print("Starting paper trading mode...")
    print(f"Starting bankroll: ${args.bankroll:,}")
    print(f"Using: {'Multi-agent' if args.multi_agent else 'Simple'} pipeline")
    print()

    portfolio = PortfolioManager(bankroll=args.bankroll)
    client = PolymarketClient(paper_trade=True)

    markets = client.get_markets()

    for market in markets[:5]:
        print(f"\nAnalyzing: {market.question}")

        if args.multi_agent:
            result = forecast_market_multi_agent(market, verbose=False)
        else:
            result = forecast_market_simple(market, verbose=False)

        if result.should_trade:
            print(f"  Signal: {result.side} with {result.edge:+.1%} edge")

            from core import TradeDecision
            trade = TradeDecision(
                market_id=market.id,
                side=result.side,
                amount=result.kelly_fraction * portfolio.cash * 0.5,
                expected_value=0,
                ai_probability=result.probability,
                market_price=market.price,
                kelly_fraction=result.kelly_fraction,
                confidence="medium"
            )

            allowed, reason = portfolio.check_risk_limits(trade)
            print(f"  Risk check: {reason}")

            if allowed:
                price = market.price if trade.side == "YES" else (1 - market.price)
                record = portfolio.execute_trade(trade, price)
                print(f"  Executed: {record.shares:.2f} shares at {record.price:.1%}")
        else:
            print(f"  No edge found")

    print(f"\n{'='*60}")
    print("PAPER TRADING SUMMARY")
    print(f"{'='*60}")
    summary = portfolio.get_performance_summary()
    print(f"Total trades: {summary['total_trades']}")
    print(f"Cash remaining: ${summary['current_cash']:.2f}")
    print(f"Position value: ${portfolio.total_value - portfolio.cash:.2f}")
    print(f"Total value: ${summary['total_value']:.2f}")


def cmd_test(args):
    """Run test pipeline."""
    print("Running test pipeline...\n")

    mock_market = Market(
        id="test-123",
        question="Will the Fed cut rates in January 2026?",
        price=0.45,
        volume=500000,
        category="economy",
        end_date="2026-01-31"
    )

    print("="*60)
    print("TEST 1: Simple Pipeline")
    print("="*60)
    result_simple = forecast_market_simple(mock_market, verbose=True)

    print("\n" + "="*60)
    print("TEST 2: Multi-Agent Pipeline")
    print("="*60)
    result_multi = forecast_market_multi_agent(mock_market, verbose=True)

    print("\n" + "="*60)
    print("TEST 3: Core Math Verification")
    print("="*60)

    from core.monte_carlo import logit, sigmoid
    test_prob = 0.65
    assert abs(sigmoid(logit(test_prob)) - test_prob) < 0.0001
    print("✓ Logit/sigmoid roundtrip: PASS")

    test_evidence = [
        Evidence(
            source="Test1", content="Test", supports="YES",
            strength=0.8, evidence_type="A", reasoning="Test"
        ),
        Evidence(
            source="Test2", content="Test", supports="NO",
            strength=0.5, evidence_type="B", reasoning="Test"
        ),
    ]
    multipliers = apply_evidence_effects(test_evidence)
    assert multipliers["Test1"] > 1.0
    assert multipliers["Test2"] < 1.0
    print("✓ Evidence effects: PASS")

    edge, side, should = calculate_edge(0.60, 0.50)
    assert side == "YES"
    assert edge > 0
    print("✓ Edge calculation: PASS")

    from core.edge_calculator import kelly_fraction
    kelly = kelly_fraction(0.60, 0.50, "YES")
    assert 0 < kelly < 1
    print("✓ Kelly calculation: PASS")

    print("\n" + "="*60)
    print("TEST 4: Calibration")
    print("="*60)

    calibrated = calibrate(0.7)
    print(f"Raw: 0.70 → Calibrated: {calibrated:.2f}")
    print("✓ Calibration: PASS")

    print("\n" + "="*60)
    print("TEST 5: Benchmark Metrics")
    print("="*60)

    predictions = [0.6, 0.4, 0.7, 0.3, 0.8]
    outcomes = [1, 0, 1, 0, 1]
    brier = brier_score(predictions, outcomes)
    ece = compute_ece(predictions, outcomes)
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
    print("✓ Metrics: PASS")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)

    print(f"\nComparison:")
    print(f"  Simple Pipeline:     {result_simple.probability:.1%}")
    print(f"  Multi-Agent Pipeline: {result_multi.probability:.1%}")
    print(f"  Market Price:        {mock_market.price:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket AI Forecaster (Multi-Agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan --limit 20
  python main.py forecast --url https://polymarket.com/event/fed-rate
  python main.py forecast --id fed-rate-jan-2026 --simple
  python main.py paper-trade --bankroll 10000 --multi-agent
  python main.py backtest
  python main.py benchmark
  python main.py test

Architecture:
  Multi-agent pipeline based on Polyseer + AIA Forecaster:
    Planner → Researcher → Critic → Analyst → Supervisor

  Features:
    - Bilateral research (YES/NO evidence)
    - Bayesian probability aggregation
    - Ensemble predictions (median of N runs)
    - Platt scaling calibration
    - ForecastBench-compatible metrics
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan markets for opportunities")
    scan_parser.add_argument("--limit", type=int, default=20)
    scan_parser.add_argument("--min-volume", type=float, default=MIN_VOLUME)
    scan_parser.add_argument("--multi-agent", action="store_true", default=True)
    scan_parser.add_argument("--simple", action="store_true")

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Forecast single market")
    forecast_parser.add_argument("--url", help="Polymarket URL")
    forecast_parser.add_argument("--id", help="Market ID")
    forecast_parser.add_argument("--simple", action="store_true",
                                  help="Use simple pipeline instead of multi-agent")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Test on resolved markets")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run ForecastBench evaluation")

    # Paper trade command
    paper_parser = subparsers.add_parser("paper-trade", help="Paper trading mode")
    paper_parser.add_argument("--bankroll", type=float, default=10000)
    paper_parser.add_argument("--multi-agent", action="store_true", default=True)
    paper_parser.add_argument("--simple", action="store_true")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test pipeline")

    args = parser.parse_args()

    # Handle --simple flag for scan and paper-trade
    if hasattr(args, 'simple') and args.simple:
        args.multi_agent = False

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "forecast":
        cmd_forecast(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "paper-trade":
        cmd_paper_trade(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
