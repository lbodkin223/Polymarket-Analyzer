"""
Polymarket Client - Wrapper around Polymarket APIs.

Aligned with official Polymarket agents SDK patterns:
https://github.com/Polymarket/agents

This module provides:
- Gamma API integration (market data)
- CLOB API integration (order placement)
- Paper trading simulation
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from core import Market
from config.settings import (
    POLYMARKET_API_KEY,
    POLYGON_WALLET_KEY,
    POLYMARKET_GAMMA_API,
    POLYMARKET_CLOB_API,
    MIN_VOLUME
)


class OrderSide(Enum):
    """Order side enum matching SDK."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    GTC = "GTC"  # Good til cancelled
    FOK = "FOK"  # Fill or kill


@dataclass
class TokenInfo:
    """Token information for a market outcome."""
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float
    winner: Optional[bool] = None


@dataclass
class MarketInfo:
    """Extended market information matching SDK structure."""
    condition_id: str
    question_id: str
    tokens: List[TokenInfo]
    minimum_order_size: float = 1.0
    minimum_tick_size: float = 0.01
    active: bool = True


@dataclass
class Order:
    """A trading order aligned with SDK OrderArgs."""
    market_id: str
    token_id: str  # Specific token to trade
    side: OrderSide
    size: float  # Number of shares
    price: float  # Limit price
    order_type: OrderType = OrderType.LIMIT
    expiration: Optional[int] = None  # Unix timestamp


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    status: str  # "LIVE", "FILLED", "PARTIAL", "CANCELLED", "REJECTED"
    filled_size: float
    remaining_size: float
    average_price: float
    timestamp: datetime
    transaction_hash: Optional[str] = None


@dataclass
class MarketBook:
    """Order book for a market."""
    token_id: str
    bids: List[Dict[str, float]]  # [{"price": 0.50, "size": 100}, ...]
    asks: List[Dict[str, float]]
    spread: float
    mid_price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeInfo:
    """Information about an executed trade."""
    trade_id: str
    market_id: str
    token_id: str
    side: str
    size: float
    price: float
    timestamp: datetime
    fee: float = 0.0


class PolymarketClient:
    """
    Client for interacting with Polymarket APIs.

    Patterns aligned with official Polymarket agents SDK:
    https://github.com/Polymarket/agents

    The SDK provides:
    - CLOB API for order placement
    - Multi-source data aggregation
    - RAG support for research
    - LLM tool scaffolding
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key: Optional[str] = None,
        paper_trade: bool = True,
        chain_id: int = 137  # Polygon mainnet
    ):
        """
        Initialize the client.

        For real trading, you need:
        1. API key from Polymarket
        2. Polygon wallet private key
        3. USDC balance on Polygon

        Args:
            api_key: Polymarket CLOB API key
            private_key: Polygon wallet private key
            paper_trade: If True, simulate trades
            chain_id: Blockchain chain ID (137 = Polygon)
        """
        self.api_key = api_key or POLYMARKET_API_KEY
        self.private_key = private_key or POLYGON_WALLET_KEY
        self.paper_trade = paper_trade
        self.chain_id = chain_id

        # Paper trading state
        self._balance = 10000.0
        self._positions: Dict[str, Dict] = {}  # token_id -> position
        self._trade_history: List[TradeInfo] = []

        # Initialize real client if credentials provided
        self._clob_client = None
        if not paper_trade and self.api_key and self.private_key:
            self._init_clob_client()

    def _init_clob_client(self):
        """
        Initialize the real CLOB client.

        STUB: Replace with actual py-clob-client initialization.

        Example:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            )

            self._clob_client = ClobClient(
                host=POLYMARKET_CLOB_API,
                chain_id=self.chain_id,
                creds=creds,
                signature_type=2,  # POLY_GNOSIS_SAFE
            )
        """
        pass  # Stub - implement with real SDK

    def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        min_volume: float = MIN_VOLUME,
        category: Optional[str] = None
    ) -> List[Market]:
        """
        Fetch markets from Polymarket Gamma API.

        Endpoint: https://gamma-api.polymarket.com/markets

        Args:
            active: Only return active markets
            limit: Maximum number of markets
            min_volume: Minimum volume filter
            category: Optional category filter

        Returns:
            List of Market objects

        STUB: Replace with actual API call:
            import requests

            params = {
                "active": active,
                "limit": limit,
                "order": "volume",
            }
            if category:
                params["tag"] = category

            response = requests.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params=params
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for m in data:
                if m.get("volume", 0) >= min_volume:
                    # Parse outcome prices
                    prices = {}
                    for token in m.get("tokens", []):
                        prices[token["outcome"]] = float(token.get("price", 0.5))

                    markets.append(Market(
                        id=m["condition_id"],
                        question=m["question"],
                        price=prices.get("Yes", 0.5),
                        volume=float(m.get("volume", 0)),
                        category=m.get("tags", ["general"])[0],
                        end_date=m.get("end_date_iso", ""),
                        description=m.get("description"),
                        liquidity=float(m.get("liquidity", 0)),
                    ))
            return markets
        """
        # Return mock markets for testing
        return [
            Market(
                id="0x1234567890abcdef",
                question="Will the Fed cut rates in January 2026?",
                price=0.45,
                volume=500000,
                category="economy",
                end_date="2026-01-31",
                description="Resolves YES if the Federal Reserve announces a rate cut at its January 2026 FOMC meeting."
            ),
            Market(
                id="0xabcdef1234567890",
                question="Will Bitcoin reach $100,000 in 2026?",
                price=0.62,
                volume=1200000,
                category="crypto",
                end_date="2026-12-31",
                description="Resolves YES if Bitcoin trades at or above $100,000 on any major exchange in 2026."
            ),
            Market(
                id="0xfedcba0987654321",
                question="Will there be a US recession in 2026?",
                price=0.28,
                volume=350000,
                category="economy",
                end_date="2026-12-31",
                description="Resolves YES if NBER declares a recession starting in 2026."
            ),
            Market(
                id="0x9876543210fedcba",
                question="Will an AI system achieve AGI by end of 2026?",
                price=0.15,
                volume=750000,
                category="tech",
                end_date="2026-12-31",
                description="Resolves based on specified AGI benchmark criteria."
            ),
            Market(
                id="0x5555666677778888",
                question="Will the S&P 500 close above 6000 in 2026?",
                price=0.58,
                volume=420000,
                category="economy",
                end_date="2026-12-31",
                description="Resolves YES if S&P 500 closes above 6000 on any trading day in 2026."
            ),
        ]

    def get_market(self, market_id: str) -> Optional[Market]:
        """
        Fetch a specific market by ID.

        Args:
            market_id: Condition ID or slug

        Returns:
            Market object or None
        """
        # In real implementation, call /markets/{id} endpoint
        markets = self.get_markets()
        for market in markets:
            if market.id == market_id or market_id in market.id:
                return market
        return None

    def get_market_by_url(self, url: str) -> Optional[Market]:
        """
        Parse a Polymarket URL and fetch the market.

        Handles URLs like:
        - https://polymarket.com/event/slug-name
        - https://polymarket.com/event/slug-name?tid=123

        Args:
            url: Polymarket market URL

        Returns:
            Market object or None
        """
        import re

        # Extract slug from URL
        patterns = [
            r'polymarket\.com/event/([^/\?]+)',
            r'polymarket\.com/market/([^/\?]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                slug = match.group(1)
                return self.get_market(slug)

        return None

    def get_market_info(self, market_id: str) -> Optional[MarketInfo]:
        """
        Get detailed market information including token IDs.

        This is needed for placing orders - you need the specific
        token_id for the outcome you want to trade.

        STUB: Real implementation would call CLOB API
        """
        market = self.get_market(market_id)
        if not market:
            return None

        # Mock token info
        return MarketInfo(
            condition_id=market_id,
            question_id=f"qid-{market_id[:8]}",
            tokens=[
                TokenInfo(
                    token_id=f"{market_id}-yes",
                    outcome="Yes",
                    price=market.price
                ),
                TokenInfo(
                    token_id=f"{market_id}-no",
                    outcome="No",
                    price=1 - market.price
                )
            ],
            minimum_order_size=5.0,
            minimum_tick_size=0.01,
            active=True
        )

    def get_order_book(self, token_id: str, depth: int = 10) -> MarketBook:
        """
        Fetch order book for a specific token.

        STUB: Real implementation:
            response = self._clob_client.get_order_book(token_id)
            return MarketBook(
                token_id=token_id,
                bids=response.bids,
                asks=response.asks,
                spread=response.asks[0].price - response.bids[0].price,
                mid_price=(response.asks[0].price + response.bids[0].price) / 2
            )

        Args:
            token_id: Token ID to get book for
            depth: Number of levels to fetch

        Returns:
            MarketBook
        """
        # Mock order book
        base_price = 0.45
        return MarketBook(
            token_id=token_id,
            bids=[
                {"price": base_price - 0.01 * i, "size": 100 * (i + 1)}
                for i in range(depth)
            ],
            asks=[
                {"price": base_price + 0.01 * (i + 1), "size": 100 * (i + 1)}
                for i in range(depth)
            ],
            spread=0.02,
            mid_price=base_price
        )

    def create_order(
        self,
        market_id: str,
        outcome: str,  # "Yes" or "No"
        side: OrderSide,
        size: float,
        price: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """
        Create an order object.

        This prepares the order but does not submit it.

        Args:
            market_id: Market condition ID
            outcome: "Yes" or "No"
            side: BUY or SELL
            size: Number of shares
            price: Limit price

        Returns:
            Order object ready for submission
        """
        market_info = self.get_market_info(market_id)
        if not market_info:
            raise ValueError(f"Market not found: {market_id}")

        # Find the token for this outcome
        token_id = None
        for token in market_info.tokens:
            if token.outcome.lower() == outcome.lower():
                token_id = token.token_id
                break

        if not token_id:
            raise ValueError(f"Outcome not found: {outcome}")

        return Order(
            market_id=market_id,
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            order_type=order_type
        )

    def place_order(self, order: Order) -> OrderResult:
        """
        Place an order on Polymarket.

        STUB: Real implementation with py-clob-client:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                token_id=order.token_id,
                price=order.price,
                size=order.size,
                side=order.side.value,
                fee_rate_bps=0,  # Maker fee
                nonce=int(time.time() * 1000),
                expiration=order.expiration or 0,
            )

            if order.order_type == OrderType.LIMIT:
                result = self._clob_client.create_and_post_order(order_args)
            else:
                result = self._clob_client.create_and_post_market_order(order_args)

            return OrderResult(
                order_id=result.id,
                status=result.status,
                filled_size=result.size_matched,
                remaining_size=result.size - result.size_matched,
                average_price=result.price,
                timestamp=datetime.now(),
                transaction_hash=result.transaction_hashes[0] if result.transaction_hashes else None
            )

        Args:
            order: Order to place

        Returns:
            OrderResult with execution details
        """
        if self.paper_trade:
            return self._paper_trade_order(order)

        if not self._clob_client:
            raise RuntimeError("CLOB client not initialized. Provide API credentials.")

        raise NotImplementedError("Real trading not yet implemented")

    def _paper_trade_order(self, order: Order) -> OrderResult:
        """Simulate order execution for paper trading."""
        # Check balance
        cost = order.size * order.price
        if order.side == OrderSide.BUY and cost > self._balance:
            return OrderResult(
                order_id="rejected-insufficient-funds",
                status="REJECTED",
                filled_size=0,
                remaining_size=order.size,
                average_price=0,
                timestamp=datetime.now()
            )

        # Simulate fill at order price (optimistic)
        fill_price = order.price

        if order.side == OrderSide.BUY:
            self._balance -= cost
            # Add to positions
            if order.token_id not in self._positions:
                self._positions[order.token_id] = {"size": 0, "avg_price": 0}

            pos = self._positions[order.token_id]
            total_size = pos["size"] + order.size
            pos["avg_price"] = (pos["size"] * pos["avg_price"] + order.size * fill_price) / total_size
            pos["size"] = total_size
        else:
            # Selling
            if order.token_id in self._positions:
                pos = self._positions[order.token_id]
                pos["size"] -= order.size
                self._balance += order.size * fill_price
                if pos["size"] <= 0:
                    del self._positions[order.token_id]

        # Record trade
        trade = TradeInfo(
            trade_id=f"paper-{len(self._trade_history)}",
            market_id=order.market_id,
            token_id=order.token_id,
            side=order.side.value,
            size=order.size,
            price=fill_price,
            timestamp=datetime.now()
        )
        self._trade_history.append(trade)

        return OrderResult(
            order_id=f"paper-{datetime.now().timestamp()}",
            status="FILLED",
            filled_size=order.size,
            remaining_size=0,
            average_price=fill_price,
            timestamp=datetime.now()
        )

    def get_balance(self) -> float:
        """Get current USDC balance."""
        if self.paper_trade:
            return self._balance

        # Real implementation would query wallet
        # Example: self._clob_client.get_balance()
        return 0.0

    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions.

        Returns:
            Dict mapping token_id to position info
        """
        if self.paper_trade:
            return self._positions.copy()

        # Real implementation would query CLOB API
        return {}

    def get_trade_history(self) -> List[TradeInfo]:
        """Get trade history."""
        return self._trade_history.copy()

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        STUB: Real implementation:
            result = self._clob_client.cancel(order_id)
            return result.status == "CANCELLED"

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled
        """
        return True  # Stub always succeeds


class PolymarketAgentClient(PolymarketClient):
    """
    Extended client with agent-specific features.

    Matches patterns from official Polymarket agents SDK:
    - Multi-source data aggregation
    - RAG support
    - LLM tool integration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_sources: List[str] = []

    def add_data_source(self, source: str):
        """Add a data source for research."""
        self._data_sources.append(source)

    def aggregate_market_data(self, market_id: str) -> Dict[str, Any]:
        """
        Aggregate data from multiple sources for a market.

        From SDK: "Multi-source data aggregation (betting services, news, web search)"

        Args:
            market_id: Market to aggregate data for

        Returns:
            Dict with aggregated data
        """
        market = self.get_market(market_id)
        if not market:
            return {}

        return {
            "market": {
                "id": market.id,
                "question": market.question,
                "price": market.price,
                "volume": market.volume,
            },
            "order_book": self.get_order_book(f"{market_id}-yes").__dict__,
            "sources": self._data_sources,
            "timestamp": datetime.now().isoformat()
        }


def scan_for_opportunities(
    min_volume: float = MIN_VOLUME,
    max_markets: int = 20,
    categories: Optional[List[str]] = None
) -> List[Market]:
    """
    Scan Polymarket for tradeable markets.

    Args:
        min_volume: Minimum volume threshold
        max_markets: Maximum markets to return
        categories: Optional category filter

    Returns:
        List of tradeable markets sorted by volume
    """
    client = PolymarketClient(paper_trade=True)
    markets = client.get_markets(active=True, min_volume=min_volume)

    if categories:
        markets = [m for m in markets if m.category in categories]

    # Sort by volume
    markets.sort(key=lambda m: m.volume, reverse=True)

    return markets[:max_markets]
