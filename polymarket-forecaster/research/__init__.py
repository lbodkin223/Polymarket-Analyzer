"""Research module for gathering evidence about prediction markets."""
from .news_search import search_news, NewsResult
from .bilateral_research import BilateralResearcher

__all__ = ['search_news', 'NewsResult', 'BilateralResearcher']
