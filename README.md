# Polymarket AI Forecaster

A Monte Carlo-based probability forecasting system for Polymarket prediction markets. This system combines bilateral research, evidence evaluation, and Kelly criterion position sizing to identify trading opportunities.

## Features

- **Bilateral Research**: Gathers evidence for BOTH sides of a question to prevent confirmation bias
- **Monte Carlo Simulation**: Converts evidence into probability multipliers and runs simulations
- **Domain Priors**: Uses category-specific baseline probabilities
- **Kelly Criterion Sizing**: Optimal position sizing based on edge and confidence
- **Paper Trading**: Test strategies without real money

## Project Structure

```
polymarket-forecaster/
├── config/
│   ├── priors.json          # Domain-specific baseline priors
│   └── settings.py          # Configuration (API keys as env vars)
├── core/
│   ├── __init__.py          # Data classes (Market, Evidence, etc.)
│   ├── market_parser.py     # Parse Polymarket questions
│   ├── evidence_evaluator.py # Classify evidence as YES/NO
│   ├── monte_carlo.py       # Probability engine
│   ├── edge_calculator.py   # Edge & Kelly sizing
│   └── blending.py          # Blend baseline + projection
├── research/
│   ├── news_search.py       # Web search for evidence
│   └── bilateral_research.py # PRO/CON evidence gathering
├── trading/
│   ├── polymarket_client.py # Polymarket API wrapper
│   └── portfolio.py         # Position tracking, risk limits
├── utils/
│   ├── cache.py             # Caching utilities
│   └── llm_router.py        # Route to cheap/expensive models
├── main.py                  # CLI entry point
└── requirements.txt
```

## Installation

```bash
# Clone or create the project
cd polymarket-forecaster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

Create a `.env` file with:

```bash
# LLM Providers (at least one required for full functionality)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Search API (for real research)
BRAVE_SEARCH_KEY=...
# or
SERP_API_KEY=...

# Polymarket (for real trading)
POLYMARKET_API_KEY=...
POLYGON_WALLET_PRIVATE_KEY=...

# Optional
REDIS_URL=redis://localhost:6379
```

## Usage

### Test the Pipeline

```bash
python main.py test
```

This runs the full forecasting pipeline with mock data to verify everything works.

### Scan for Opportunities

```bash
python main.py scan --limit 20 --min-volume 10000
```

Scans Polymarket for markets with significant edge.

### Forecast a Single Market

```bash
# By URL
python main.py forecast --url https://polymarket.com/event/fed-rate-jan-2026

# By ID
python main.py forecast --id fed-rate-jan-2026
```

### Paper Trading

```bash
python main.py paper-trade --bankroll 10000
```

Runs a paper trading simulation.

## How It Works

### 1. Parse Market Question
The system analyzes the market question to:
- Categorize it (politics, economy, crypto, etc.)
- Extract key entities and resolution criteria
- Determine time horizon
- Set baseline probability

### 2. Bilateral Research
To prevent confirmation bias, the system searches for evidence on BOTH sides:
- Generates search queries for YES case
- Generates search queries for NO case
- Evaluates each piece of evidence

### 3. Evidence Evaluation
Each evidence item is classified:
- **Direction**: Does it support YES, NO, or NEUTRAL?
- **Strength**: How strongly? (0.0 - 1.0)
- **Type**: A (primary), B (high-quality), C (standard), D (weak)

### 4. Monte Carlo Simulation
Evidence is converted to probability multipliers:
- YES evidence with high strength → multiplier > 1.0
- NO evidence with high strength → multiplier < 1.0

The simulation:
1. Starts with baseline probability in logit space
2. Applies each multiplier with ±20% random variation
3. Runs 10,000 simulations
4. Returns median and 90% confidence interval

### 5. Probability Blending
Final probability blends baseline with projection:
- High confidence: 75% baseline, 25% projection
- Medium: 50/50
- Low: 25% baseline, 75% projection

### 6. Edge Calculation
```
edge = ai_probability - market_price
```
Only trade if edge > 5% (configurable).

### 7. Kelly Criterion Sizing
```
f* = (bp - q) / b
```
Where:
- b = odds (payout per $1)
- p = probability of winning
- q = probability of losing

Capped at 25% of bankroll per trade.

## Extending the System

### Adding Real API Integration

Replace the stub functions in each module. Example for Brave Search:

```python
# In research/news_search.py
import requests
from config.settings import BRAVE_SEARCH_KEY

def search_news_brave(query: str, num_results: int = 10) -> List[NewsResult]:
    headers = {"X-Subscription-Token": BRAVE_SEARCH_KEY}
    params = {"q": query, "count": num_results}

    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params=params
    )
    response.raise_for_status()

    results = []
    for item in response.json().get("web", {}).get("results", []):
        results.append(NewsResult(
            title=item["title"],
            snippet=item["description"],
            url=item["url"],
            source=item.get("meta_url", {}).get("hostname", "Unknown")
        ))
    return results
```

### Adding Polymarket SDK

For real trading, integrate the official SDK:

```bash
pip install py-clob-client
```

See `trading/polymarket_client.py` for integration points.

## Configuration

Edit `config/settings.py`:

```python
MIN_EDGE_PERCENT = 0.05  # 5% minimum edge to trade
MAX_KELLY_FRACTION = 0.25  # Max 25% of bankroll per trade
MIN_VOLUME = 10000  # Only trade markets with $10k+ volume
```

Edit `config/priors.json` to add domain-specific baseline probabilities.

## Testing

```bash
# Run test pipeline
python main.py test

# Run with pytest (when tests are added)
pytest tests/
```

## License

MIT
