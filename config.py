"""
Configuration settings for keyword similarity analysis.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the application."""

    # OpenAI settings
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000

    # TF-IDF settings
    MIN_DF: int = 1
    MAX_DF: float = 1.0
    MAX_FEATURES: Optional[int] = None
    NGRAM_RANGE: tuple = (1, 2)
    STOP_WORDS: str = 'english'

    # Visualization defaults
    DEFAULT_HEATMAP_SIZE: int = 20
    DEFAULT_THRESHOLD: float = 0.5
    MIN_HEATMAP_SIZE: int = 5
    MAX_HEATMAP_SIZE: int = 100

    # Limits
    MAX_KEYWORDS: int = 1000
    MIN_KEYWORDS_FOR_ANALYSIS: int = 2

    # Export settings
    DEFAULT_TOP_PAIRS: int = 50

    # UI settings
    PAGE_TITLE: str = "Keyword Similarity Analyzer"
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"


# Global config instance
config = Config()
