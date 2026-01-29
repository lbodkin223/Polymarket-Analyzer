"""
Base Agent - Foundation for all specialized agents.

Provides common interface and LLM interaction patterns.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from config.settings import DEFAULT_LLM, EXPENSIVE_LLM


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    model: str = DEFAULT_LLM
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None


@dataclass
class AgentResponse:
    """Standard response from an agent."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    reasoning: str
    confidence: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all forecasting agents.

    Agents are specialized components that handle specific
    parts of the forecasting pipeline.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig(name=self.__class__.__name__)
        self._call_count = 0
        self._total_tokens = 0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Execute the agent's main task.

        Args:
            input_data: Input data for the agent

        Returns:
            AgentResponse with results
        """
        pass

    def call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """
        STUB: Call the LLM with the given prompt.

        Replace with actual OpenAI/Anthropic API call.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            LLM response text
        """
        self._call_count += 1

        # Mock response - replace with actual API call
        # Example with OpenAI:
        #   from openai import OpenAI
        #   client = OpenAI()
        #   messages = []
        #   if system:
        #       messages.append({"role": "system", "content": system})
        #   messages.append({"role": "user", "content": prompt})
        #   response = client.chat.completions.create(
        #       model=self.config.model,
        #       messages=messages,
        #       temperature=self.config.temperature,
        #       max_tokens=self.config.max_tokens
        #   )
        #   return response.choices[0].message.content

        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        return json.dumps({
            "status": "success",
            "agent": self.name,
            "mock": True
        })

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles cases where JSON is wrapped in markdown code blocks.
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try extracting any JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        # Return empty dict if all parsing fails
        return {}

    def log(self, message: str):
        """Log a message from this agent."""
        print(f"[{self.name}] {message}")

    def create_response(
        self,
        success: bool,
        data: Dict[str, Any],
        reasoning: str,
        confidence: float,
        error: Optional[str] = None
    ) -> AgentResponse:
        """Create a standardized response."""
        return AgentResponse(
            agent_name=self.name,
            success=success,
            data=data,
            reasoning=reasoning,
            confidence=confidence,
            tokens_used=self._total_tokens,
            error=error
        )
