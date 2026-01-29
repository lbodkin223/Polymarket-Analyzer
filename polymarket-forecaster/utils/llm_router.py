"""
LLM Router - Route prompts to appropriate models based on task complexity.

This module helps optimize costs by using cheaper models for simple tasks
and expensive models only when needed.
"""
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

from config.settings import (
    DEFAULT_LLM,
    EXPENSIVE_LLM,
    ANTHROPIC_DEFAULT,
    ANTHROPIC_EXPENSIVE,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY
)


class TaskComplexity(Enum):
    """Classification of task complexity."""
    SIMPLE = "simple"      # Classification, extraction, simple Q&A
    MODERATE = "moderate"  # Analysis, summarization
    COMPLEX = "complex"    # Multi-step reasoning, nuanced judgment


class Provider(Enum):
    """LLM provider."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: Provider
    model_id: str
    cost_per_1k_input: float  # Cost per 1000 input tokens
    cost_per_1k_output: float  # Cost per 1000 output tokens
    max_tokens: int
    supports_json: bool = True


# Model configurations
MODELS = {
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        max_tokens=128000,
    ),
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        max_tokens=128000,
    ),
    "claude-3-haiku-20240307": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        max_tokens=200000,
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_tokens=200000,
    ),
}


class LLMRouter:
    """
    Routes LLM calls to appropriate models based on task type.
    """

    def __init__(
        self,
        default_provider: Provider = Provider.OPENAI,
        prefer_cheap: bool = True
    ):
        """
        Initialize the router.

        Args:
            default_provider: Preferred provider when both available
            prefer_cheap: If True, prefer cheaper models when possible
        """
        self.default_provider = default_provider
        self.prefer_cheap = prefer_cheap

        # Detect available providers
        self._openai_available = bool(OPENAI_API_KEY)
        self._anthropic_available = bool(ANTHROPIC_API_KEY)

    def get_model_for_task(self, task_type: str) -> str:
        """
        Get the appropriate model for a task type.

        Task types:
        - "classify": Simple classification (cheap model)
        - "extract": Information extraction (cheap model)
        - "evaluate": Evidence evaluation (moderate model)
        - "analyze": Deep analysis (expensive model)
        - "synthesize": Complex synthesis (expensive model)

        Args:
            task_type: Type of task

        Returns:
            Model ID string
        """
        simple_tasks = ["classify", "extract", "parse", "format"]
        moderate_tasks = ["evaluate", "summarize", "compare"]
        complex_tasks = ["analyze", "synthesize", "reason", "forecast"]

        if task_type in simple_tasks:
            complexity = TaskComplexity.SIMPLE
        elif task_type in moderate_tasks:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.COMPLEX

        return self._select_model(complexity)

    def _select_model(self, complexity: TaskComplexity) -> str:
        """Select model based on complexity and availability."""
        if self.default_provider == Provider.OPENAI and self._openai_available:
            if complexity == TaskComplexity.SIMPLE:
                return "gpt-4o-mini"
            else:
                return "gpt-4o" if not self.prefer_cheap else "gpt-4o-mini"

        elif self.default_provider == Provider.ANTHROPIC and self._anthropic_available:
            if complexity == TaskComplexity.SIMPLE:
                return "claude-3-haiku-20240307"
            else:
                return "claude-3-5-sonnet-20241022" if not self.prefer_cheap else "claude-3-haiku-20240307"

        # Fallback
        if self._openai_available:
            return "gpt-4o-mini"
        elif self._anthropic_available:
            return "claude-3-haiku-20240307"
        else:
            return DEFAULT_LLM  # Will use mock

    def call(
        self,
        prompt: str,
        task_type: str = "general",
        model_override: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        STUB: Make an LLM call with automatic model routing.

        Example implementation with OpenAI:
            from openai import OpenAI

            model = model_override or self.get_model_for_task(task_type)
            config = MODELS.get(model)

            if config.provider == Provider.OPENAI:
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        Example with Anthropic:
            import anthropic

            if config.provider == Provider.ANTHROPIC:
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

        Args:
            prompt: The prompt to send
            task_type: Type of task for model selection
            model_override: Force a specific model
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        model = model_override or self.get_model_for_task(task_type)

        # Mock response for now
        return f"[Mock response from {model} for task '{task_type}']"

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """
        Estimate the cost of an LLM call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model ID

        Returns:
            Estimated cost in dollars
        """
        config = MODELS.get(model)
        if not config:
            return 0.0

        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output

        return input_cost + output_cost


# Convenience function
def route_to_model(task_type: str) -> str:
    """Get appropriate model for a task type."""
    router = LLMRouter()
    return router.get_model_for_task(task_type)


def call_llm_routed(prompt: str, task_type: str = "general") -> str:
    """Make a routed LLM call."""
    router = LLMRouter()
    return router.call(prompt, task_type)
