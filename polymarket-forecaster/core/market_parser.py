"""
Market Parser - Extract structured information from Polymarket questions.
"""
import json
import re
from typing import Optional
from datetime import datetime, timedelta

from . import Market, MarketAnalysis
from config.settings import DEFAULT_LLM


def call_llm(prompt: str, model: str = DEFAULT_LLM) -> str:
    """
    STUB: Replace with OpenAI/Anthropic API call.
    API_KEY: OPENAI_API_KEY or ANTHROPIC_API_KEY

    Example real implementation:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    """
    # Return mock LLM response based on question content
    question_lower = prompt.lower()

    # Detect category from question
    if any(word in question_lower for word in ['fed', 'rate', 'inflation', 'gdp', 'recession', 'economic']):
        category = "economy"
        subcategory = "monetary_policy"
    elif any(word in question_lower for word in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth']):
        category = "crypto"
        subcategory = "price"
    elif any(word in question_lower for word in ['election', 'president', 'congress', 'senate', 'vote']):
        category = "politics"
        subcategory = "election"
    elif any(word in question_lower for word in ['war', 'conflict', 'military', 'invasion']):
        category = "geopolitics"
        subcategory = "conflict"
    elif any(word in question_lower for word in ['nfl', 'nba', 'mlb', 'super bowl', 'championship']):
        category = "sports"
        subcategory = "championship"
    else:
        category = "general"
        subcategory = None

    mock_response = {
        "category": category,
        "subcategory": subcategory,
        "key_entities": ["Entity1", "Entity2"],
        "resolution_criteria": "Market resolves YES if the stated condition is met by the end date.",
        "time_horizon": "medium_term",
        "baseline_probability": 0.50,
        "confidence": "medium",
        "reasoning": "Based on historical patterns and current context."
    }

    return json.dumps(mock_response)


def parse_market_question(question: str, description: Optional[str] = None) -> MarketAnalysis:
    """
    Parse a market question to extract structured information.

    Args:
        question: The market question text
        description: Optional additional context/description

    Returns:
        MarketAnalysis with structured information
    """
    prompt = f"""Analyze this prediction market question and extract structured information.

Question: {question}
{f'Description: {description}' if description else ''}

Respond in JSON format with these fields:
- category: one of [politics, economy, crypto, sports, tech, geopolitics, entertainment, general]
- subcategory: more specific category (e.g., "election", "monetary_policy", "championship")
- key_entities: list of key people, organizations, or things mentioned
- resolution_criteria: how the market resolves YES vs NO
- time_horizon: one of [immediate (< 7 days), short_term (< 30 days), medium_term (< 90 days), long_term (> 90 days)]
- baseline_probability: your initial probability estimate (0.0-1.0) before any research
- confidence: your confidence in this baseline [high, medium, low]
- reasoning: brief explanation of your baseline probability

JSON response:"""

    response = call_llm(prompt)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        data = {
            "category": "general",
            "subcategory": None,
            "key_entities": [],
            "resolution_criteria": "Unknown",
            "time_horizon": "medium_term",
            "baseline_probability": 0.50,
            "confidence": "low",
            "reasoning": "Could not parse market question"
        }

    return MarketAnalysis(
        original_question=question,
        category=data.get("category", "general"),
        subcategory=data.get("subcategory"),
        key_entities=data.get("key_entities", []),
        resolution_criteria=data.get("resolution_criteria", "Unknown"),
        time_horizon=data.get("time_horizon", "medium_term"),
        baseline_probability=data.get("baseline_probability", 0.50),
        confidence=data.get("confidence", "medium"),
        reasoning=data.get("reasoning", "")
    )


def check_resolution_status(market_price: float, end_date: str) -> str:
    """
    Determine if a market is tradeable based on price and time.

    Args:
        market_price: Current YES price (0-1)
        end_date: Market end date string (ISO format)

    Returns:
        "resolved", "near_resolved", or "tradeable"
    """
    # Check if price indicates resolution
    if market_price >= 0.98 or market_price <= 0.02:
        return "resolved"

    if market_price >= 0.95 or market_price <= 0.05:
        return "near_resolved"

    # Check time remaining
    try:
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        now = datetime.now(end.tzinfo) if end.tzinfo else datetime.now()
        days_remaining = (end - now).days

        if days_remaining <= 0:
            return "resolved"
        elif days_remaining <= 1:
            return "near_resolved"
    except (ValueError, TypeError):
        pass  # If we can't parse the date, assume tradeable

    return "tradeable"


def extract_date_from_question(question: str) -> Optional[str]:
    """
    Try to extract a resolution date from the question text.

    Args:
        question: The market question

    Returns:
        ISO date string if found, None otherwise
    """
    # Common patterns
    patterns = [
        r'by (\w+ \d{1,2},? \d{4})',  # "by January 1, 2026"
        r'before (\w+ \d{1,2},? \d{4})',  # "before March 15, 2026"
        r'in (\w+ \d{4})',  # "in January 2026"
        r'(\d{4}-\d{2}-\d{2})',  # ISO format
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Try to parse various formats
            for fmt in ['%B %d, %Y', '%B %d %Y', '%B %Y', '%Y-%m-%d']:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.isoformat()
                except ValueError:
                    continue

    return None


def get_time_horizon_days(time_horizon: str) -> int:
    """Convert time horizon string to approximate days."""
    horizons = {
        "immediate": 7,
        "short_term": 30,
        "medium_term": 90,
        "long_term": 365
    }
    return horizons.get(time_horizon, 90)


def categorize_market(market: Market) -> Market:
    """
    Enrich a market with parsed category information.

    Args:
        market: Market object to categorize

    Returns:
        Market with updated category field
    """
    analysis = parse_market_question(market.question, market.description)
    market.category = analysis.category
    return market
