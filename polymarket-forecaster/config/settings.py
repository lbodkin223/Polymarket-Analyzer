"""
Configuration settings for Polymarket Forecaster.
All API keys are loaded from environment variables.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# API Keys (from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BRAVE_SEARCH_KEY = os.getenv("BRAVE_SEARCH_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY")
POLYGON_WALLET_KEY = os.getenv("POLYGON_WALLET_PRIVATE_KEY")

# Trading settings
MIN_EDGE_PERCENT = 0.05  # 5% minimum edge to trade
MAX_KELLY_FRACTION = 0.25  # Max 25% of bankroll per trade
MIN_VOLUME = 10000  # Only trade markets with $10k+ volume
MAX_POSITION_SIZE = 1000  # Max $1000 per position
BANKROLL = 10000  # Total bankroll for Kelly calculations

# Model settings
DEFAULT_LLM = "gpt-4o-mini"
EXPENSIVE_LLM = "gpt-4o"
ANTHROPIC_DEFAULT = "claude-3-haiku-20240307"
ANTHROPIC_EXPENSIVE = "claude-3-5-sonnet-20241022"

# Monte Carlo settings
DEFAULT_SIMULATIONS = 10000
MULTIPLIER_VARIANCE = 0.20  # ±20% random variation on multipliers

# Evidence type weights
EVIDENCE_TYPE_WEIGHTS = {
    'A': 1.0,   # Primary source (official announcements, direct quotes)
    'B': 0.8,   # High-quality secondary (major news outlets, expert analysis)
    'C': 0.5,   # Standard (general news, reputable blogs)
    'D': 0.2,   # Weak (social media, unverified sources)
}

# Probability clamping
MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.95

# Cache settings
CACHE_TTL_SECONDS = 3600  # 1 hour cache for research results
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Polymarket API
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
