"""
Portfolio Manager - Position tracking and risk management.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from core import Position, Portfolio, TradeDecision
from config.settings import BANKROLL, MAX_KELLY_FRACTION, MAX_POSITION_SIZE


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    market_id: str
    side: str
    shares: float
    price: float
    amount: float
    timestamp: datetime
    pnl: Optional[float] = None  # Filled when position closed


class PortfolioManager:
    """
    Manage portfolio positions and risk limits.
    """

    def __init__(
        self,
        bankroll: float = BANKROLL,
        max_position_pct: float = MAX_KELLY_FRACTION,
        max_position_size: float = MAX_POSITION_SIZE,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize portfolio manager.

        Args:
            bankroll: Total capital available
            max_position_pct: Maximum position as percentage of bankroll
            max_position_size: Maximum position in dollars
            data_dir: Directory for persisting portfolio data
        """
        self.bankroll = bankroll
        self.max_position_pct = max_position_pct
        self.max_position_size = max_position_size
        self.data_dir = data_dir or Path("./data")

        self._positions: Dict[str, Position] = {}
        self._trade_history: List[TradeRecord] = []
        self._cash = bankroll

    @property
    def positions(self) -> List[Position]:
        """Get all current positions."""
        return list(self._positions.values())

    @property
    def cash(self) -> float:
        """Get available cash."""
        return self._cash

    @property
    def total_value(self) -> float:
        """Get total portfolio value."""
        position_value = sum(p.value for p in self.positions)
        return self._cash + position_value

    @property
    def total_pnl(self) -> float:
        """Get total P&L."""
        return self.total_value - self.bankroll

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        return Portfolio(
            positions=self.positions,
            cash=self._cash,
            total_value=self.total_value,
            total_pnl=self.total_pnl
        )

    def check_risk_limits(self, trade: TradeDecision) -> tuple[bool, str]:
        """
        Check if a trade passes risk limits.

        Args:
            trade: Proposed trade

        Returns:
            Tuple of (allowed, reason)
        """
        # Check position size limits
        if trade.amount > self.max_position_size:
            return False, f"Trade exceeds max position size (${self.max_position_size})"

        if trade.amount > self.bankroll * self.max_position_pct:
            return False, f"Trade exceeds max position % ({self.max_position_pct:.0%})"

        # Check cash availability
        if trade.amount > self._cash:
            return False, f"Insufficient cash (${self._cash:.2f} available)"

        # Check existing position in same market
        if trade.market_id in self._positions:
            existing = self._positions[trade.market_id]
            if existing.side != trade.side:
                # Would need to close existing position first
                return True, "Will close existing opposite position"

        # Check total exposure limits
        current_exposure = sum(p.value for p in self.positions)
        max_exposure = self.bankroll * 0.8  # Max 80% deployed

        if current_exposure + trade.amount > max_exposure:
            return False, f"Would exceed max exposure ({max_exposure:.0%})"

        return True, "Trade approved"

    def apply_kelly_adjustment(
        self,
        kelly_fraction: float,
        confidence: str = "medium"
    ) -> float:
        """
        Adjust Kelly fraction based on confidence and portfolio state.

        Args:
            kelly_fraction: Raw Kelly fraction
            confidence: Confidence level

        Returns:
            Adjusted position size in dollars
        """
        # Scale Kelly by confidence
        confidence_multipliers = {
            "high": 1.0,
            "medium": 0.75,
            "low": 0.5
        }
        multiplier = confidence_multipliers.get(confidence, 0.75)

        adjusted_kelly = kelly_fraction * multiplier

        # Cap at maximum
        adjusted_kelly = min(adjusted_kelly, self.max_position_pct)

        # Convert to dollars
        position_size = self._cash * adjusted_kelly

        # Apply absolute cap
        position_size = min(position_size, self.max_position_size)

        return position_size

    def execute_trade(
        self,
        trade: TradeDecision,
        execution_price: float
    ) -> TradeRecord:
        """
        Record a trade execution.

        Args:
            trade: Trade decision
            execution_price: Actual execution price

        Returns:
            TradeRecord of the execution
        """
        shares = trade.amount / execution_price

        # Update cash
        self._cash -= trade.amount

        # Update or create position
        if trade.market_id in self._positions:
            existing = self._positions[trade.market_id]
            if existing.side == trade.side:
                # Add to position
                total_shares = existing.shares + shares
                avg_price = (
                    (existing.shares * existing.avg_price + shares * execution_price)
                    / total_shares
                )
                self._positions[trade.market_id] = Position(
                    market_id=trade.market_id,
                    side=trade.side,
                    shares=total_shares,
                    avg_price=avg_price,
                    current_price=execution_price,
                    unrealized_pnl=0  # Will be updated on mark-to-market
                )
            else:
                # Close existing position (simplified)
                self._cash += existing.shares * execution_price
                del self._positions[trade.market_id]

                # Open new position if trade is larger
                if shares > existing.shares:
                    remaining_shares = shares - existing.shares
                    self._positions[trade.market_id] = Position(
                        market_id=trade.market_id,
                        side=trade.side,
                        shares=remaining_shares,
                        avg_price=execution_price,
                        current_price=execution_price,
                        unrealized_pnl=0
                    )
        else:
            # New position
            self._positions[trade.market_id] = Position(
                market_id=trade.market_id,
                side=trade.side,
                shares=shares,
                avg_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=0
            )

        # Record trade
        record = TradeRecord(
            trade_id=f"trade-{len(self._trade_history)+1}",
            market_id=trade.market_id,
            side=trade.side,
            shares=shares,
            price=execution_price,
            amount=trade.amount,
            timestamp=datetime.now()
        )
        self._trade_history.append(record)

        return record

    def mark_to_market(self, market_prices: Dict[str, float]):
        """
        Update positions with current market prices.

        Args:
            market_prices: Dict of market_id -> current_price
        """
        for market_id, position in self._positions.items():
            if market_id in market_prices:
                current = market_prices[market_id]

                # Adjust for side
                if position.side == "NO":
                    current = 1 - current

                position.current_price = current
                position.unrealized_pnl = (
                    (current - position.avg_price) * position.shares
                )

    def close_position(
        self,
        market_id: str,
        settlement_price: float
    ) -> Optional[TradeRecord]:
        """
        Close a position at settlement.

        Args:
            market_id: Market ID
            settlement_price: Final settlement price (1.0 for YES, 0.0 for NO)

        Returns:
            TradeRecord of the close, or None if no position
        """
        if market_id not in self._positions:
            return None

        position = self._positions[market_id]

        # Calculate settlement value
        if position.side == "YES":
            payout = position.shares * settlement_price
        else:
            payout = position.shares * (1 - settlement_price)

        pnl = payout - (position.shares * position.avg_price)

        # Update cash
        self._cash += payout

        # Remove position
        del self._positions[market_id]

        # Record
        record = TradeRecord(
            trade_id=f"close-{len(self._trade_history)+1}",
            market_id=market_id,
            side="CLOSE",
            shares=-position.shares,
            price=settlement_price,
            amount=-payout,
            timestamp=datetime.now(),
            pnl=pnl
        )
        self._trade_history.append(record)

        return record

    def get_trade_history(self) -> List[TradeRecord]:
        """Get all trade history."""
        return self._trade_history

    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary."""
        realized_pnl = sum(
            t.pnl for t in self._trade_history
            if t.pnl is not None
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)

        winning_trades = [
            t for t in self._trade_history
            if t.pnl is not None and t.pnl > 0
        ]
        losing_trades = [
            t for t in self._trade_history
            if t.pnl is not None and t.pnl < 0
        ]

        return {
            "total_trades": len(self._trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "current_cash": self._cash,
            "total_value": self.total_value,
            "return_pct": (self.total_value - self.bankroll) / self.bankroll,
        }

    def save(self, filepath: Optional[Path] = None):
        """Save portfolio state to file."""
        filepath = filepath or self.data_dir / "portfolio.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bankroll": self.bankroll,
            "cash": self._cash,
            "positions": [
                {
                    "market_id": p.market_id,
                    "side": p.side,
                    "shares": p.shares,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                }
                for p in self.positions
            ],
            "trade_history": [
                {
                    "trade_id": t.trade_id,
                    "market_id": t.market_id,
                    "side": t.side,
                    "shares": t.shares,
                    "price": t.price,
                    "amount": t.amount,
                    "timestamp": t.timestamp.isoformat(),
                    "pnl": t.pnl,
                }
                for t in self._trade_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[Path] = None):
        """Load portfolio state from file."""
        filepath = filepath or self.data_dir / "portfolio.json"

        if not filepath.exists():
            return

        with open(filepath) as f:
            data = json.load(f)

        self.bankroll = data["bankroll"]
        self._cash = data["cash"]

        self._positions = {}
        for p in data["positions"]:
            self._positions[p["market_id"]] = Position(
                market_id=p["market_id"],
                side=p["side"],
                shares=p["shares"],
                avg_price=p["avg_price"],
                current_price=p["current_price"],
                unrealized_pnl=0
            )

        self._trade_history = []
        for t in data["trade_history"]:
            self._trade_history.append(TradeRecord(
                trade_id=t["trade_id"],
                market_id=t["market_id"],
                side=t["side"],
                shares=t["shares"],
                price=t["price"],
                amount=t["amount"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                pnl=t.get("pnl"),
            ))
