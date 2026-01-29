"""
Edge Calculator - Calculate trading edge and Kelly criterion position sizing.

This module compares AI probability estimates to market prices and
determines optimal position sizes using the Kelly criterion.
"""
from typing import Tuple, Optional
from dataclasses import dataclass

from config.settings import MIN_EDGE_PERCENT, MAX_KELLY_FRACTION, BANKROLL


@dataclass
class EdgeAnalysis:
    """Complete edge analysis for a market."""
    edge: float  # AI prob - market price (for YES side)
    edge_percent: float  # Edge as percentage
    side: str  # "YES" or "NO"
    should_trade: bool
    kelly_fraction: float
    expected_value: float
    position_size: float  # In dollars
    risk_reward_ratio: float


def calculate_edge(
    ai_prob: float,
    market_price: float,
    min_edge: float = MIN_EDGE_PERCENT
) -> Tuple[float, str, bool]:
    """
    Calculate the edge between AI probability and market price.

    The edge is the difference between our probability estimate and
    the market price. A positive edge on YES means we think YES is
    underpriced; a positive edge on NO means we think NO is underpriced.

    Args:
        ai_prob: AI's probability estimate for YES (0-1)
        market_price: Current market price for YES (0-1)
        min_edge: Minimum edge required to trade (default 5%)

    Returns:
        Tuple of (edge_value, side, should_trade)
        - edge_value: The magnitude of the edge
        - side: "YES" if betting on yes, "NO" if betting on no
        - should_trade: True if edge exceeds minimum threshold
    """
    # Calculate raw edge
    yes_edge = ai_prob - market_price
    no_edge = (1 - ai_prob) - (1 - market_price)  # = market_price - ai_prob

    # Determine which side has positive edge
    if yes_edge > 0:
        side = "YES"
        edge = yes_edge
    elif no_edge > 0:
        side = "NO"
        edge = no_edge
    else:
        # This shouldn't happen mathematically, but handle it
        side = "YES" if abs(yes_edge) > abs(no_edge) else "NO"
        edge = 0.0

    # Check if edge exceeds threshold
    should_trade = edge >= min_edge

    return (edge, side, should_trade)


def kelly_fraction(
    ai_prob: float,
    market_price: float,
    side: str,
    max_kelly: float = MAX_KELLY_FRACTION
) -> float:
    """
    Calculate Kelly criterion fraction for position sizing.

    Kelly formula: f* = (bp - q) / b
    Where:
        b = odds received on the bet (payout per $1 bet, minus the $1)
        p = probability of winning
        q = probability of losing (1 - p)

    For prediction markets:
        - YES at price P: you pay P, win 1 if YES happens
            - b = (1 - P) / P  (you win 1-P for a P bet)
        - NO at price (1-P): you pay (1-P), win 1 if NO happens
            - b = P / (1 - P)  (you win P for a 1-P bet)

    Args:
        ai_prob: AI's probability estimate for YES (0-1)
        market_price: Current market price for YES (0-1)
        side: "YES" or "NO"
        max_kelly: Maximum Kelly fraction to use (default 0.25)

    Returns:
        Kelly fraction (capped at max_kelly), 0 if negative edge
    """
    if side == "YES":
        # Betting YES: cost = market_price, win = 1
        # b = (1 - market_price) / market_price
        p = ai_prob  # Our probability YES happens
        q = 1 - p
        b = (1 - market_price) / market_price if market_price > 0 else 0
    else:
        # Betting NO: cost = (1 - market_price), win = 1
        # b = market_price / (1 - market_price)
        p = 1 - ai_prob  # Our probability NO happens
        q = 1 - p
        b = market_price / (1 - market_price) if market_price < 1 else 0

    if b <= 0:
        return 0.0

    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b

    # Never bet more than max_kelly fraction
    kelly = max(0.0, min(max_kelly, kelly))

    return kelly


def calculate_expected_value(
    ai_prob: float,
    market_price: float,
    side: str,
    bet_amount: float = 1.0
) -> float:
    """
    Calculate expected value of a bet.

    EV = (probability of win × profit if win) - (probability of loss × loss if lose)

    Args:
        ai_prob: AI's probability estimate for YES
        market_price: Current market price for YES
        side: "YES" or "NO"
        bet_amount: Amount to bet

    Returns:
        Expected value in dollars
    """
    if side == "YES":
        win_prob = ai_prob
        # Profit if win: bet_amount * (1 - market_price) / market_price
        # Using shares model: spend $bet_amount at price, get bet_amount/price shares
        # Each share pays $1 if YES, so profit = bet_amount/price - bet_amount
        profit_if_win = bet_amount * (1 - market_price) / market_price
        loss_if_lose = bet_amount
    else:
        win_prob = 1 - ai_prob
        # Betting NO at price (1 - market_price)
        profit_if_win = bet_amount * market_price / (1 - market_price)
        loss_if_lose = bet_amount

    ev = (win_prob * profit_if_win) - ((1 - win_prob) * loss_if_lose)

    return ev


def calculate_position_size(
    kelly: float,
    bankroll: float = BANKROLL,
    confidence_factor: float = 1.0
) -> float:
    """
    Calculate position size in dollars.

    Args:
        kelly: Kelly fraction
        bankroll: Total bankroll
        confidence_factor: Multiply Kelly by this (0-1) for more conservative sizing

    Returns:
        Position size in dollars
    """
    adjusted_kelly = kelly * confidence_factor
    return bankroll * adjusted_kelly


def full_edge_analysis(
    ai_prob: float,
    market_price: float,
    bankroll: float = BANKROLL,
    min_edge: float = MIN_EDGE_PERCENT,
    max_kelly: float = MAX_KELLY_FRACTION,
    confidence_factor: float = 1.0
) -> EdgeAnalysis:
    """
    Perform complete edge analysis for a market.

    Args:
        ai_prob: AI's probability estimate for YES
        market_price: Current market price for YES
        bankroll: Total bankroll
        min_edge: Minimum edge to trade
        max_kelly: Maximum Kelly fraction
        confidence_factor: Conservative sizing factor

    Returns:
        EdgeAnalysis with all calculations
    """
    edge, side, should_trade = calculate_edge(ai_prob, market_price, min_edge)

    kelly = kelly_fraction(ai_prob, market_price, side, max_kelly)
    kelly_adjusted = kelly * confidence_factor

    position = calculate_position_size(kelly, bankroll, confidence_factor)
    ev = calculate_expected_value(ai_prob, market_price, side, position)

    # Risk/reward ratio
    if side == "YES":
        potential_profit = position * (1 - market_price) / market_price
        risk = position
    else:
        potential_profit = position * market_price / (1 - market_price)
        risk = position

    rr_ratio = potential_profit / risk if risk > 0 else 0

    return EdgeAnalysis(
        edge=edge,
        edge_percent=edge * 100,
        side=side,
        should_trade=should_trade,
        kelly_fraction=kelly_adjusted,
        expected_value=ev,
        position_size=position,
        risk_reward_ratio=rr_ratio
    )


def compare_to_market(
    ai_prob: float,
    market_price: float
) -> str:
    """
    Generate human-readable comparison of AI vs market.

    Args:
        ai_prob: AI probability
        market_price: Market price

    Returns:
        Description string
    """
    edge, side, should_trade = calculate_edge(ai_prob, market_price)

    if not should_trade:
        return f"No significant edge. AI: {ai_prob:.1%}, Market: {market_price:.1%}, Edge: {edge:+.1%}"

    if side == "YES":
        return f"YES is underpriced. AI: {ai_prob:.1%} vs Market: {market_price:.1%}. Edge: +{edge:.1%}"
    else:
        return f"NO is underpriced. AI: {1-ai_prob:.1%} vs Market: {1-market_price:.1%}. Edge: +{edge:.1%}"


def fractional_kelly(
    kelly: float,
    fraction: float = 0.5
) -> float:
    """
    Apply fractional Kelly for more conservative sizing.

    Full Kelly is theoretically optimal but volatile.
    Half Kelly (fraction=0.5) is commonly used in practice.

    Args:
        kelly: Full Kelly fraction
        fraction: What fraction of Kelly to use

    Returns:
        Reduced Kelly fraction
    """
    return kelly * fraction
