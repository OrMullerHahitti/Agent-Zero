"""Shared interface for both local and remote SLM clients."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging


class ModelType(Enum):
    """Types of models supported."""
    LOCAL = "local"
    REMOTE = "remote"


class TaskType(Enum):
    """Types of tasks the SLM can perform."""
    CLASSIFICATION = "classification"
    EXPLANATION = "explanation" 
    REASONING = "reasoning"
    CODING = "coding"
    GENERAL = "general"


@dataclass
class SLMConfig:
    """Configuration for SLM clients."""
    model_type: ModelType
    model_name: str
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Performance settings
    batch_size: int = 1
    timeout: float = 60.0
    retry_attempts: int = 3
    
    # Local model specific
    device: str = "auto"
    quantization: Optional[str] = None  # "8bit", "4bit", None
    
    # Remote API specific
    rate_limit: Optional[int] = None
    cost_tracking: bool = True


@dataclass
class SLMResponse:
    """Response from SLM client."""
    text: str
    success: bool
    
    # Metadata
    model_name: str
    task_type: TaskType
    execution_time: float
    token_count: Optional[int] = None
    cost: Optional[float] = None
    confidence: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Performance metrics
    tokens_per_second: Optional[float] = None
    memory_usage: Optional[float] = None


class SLMInterface(ABC):
    """Abstract interface for all SLM clients (local and remote)."""
    
    def __init__(self, config: SLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"slm.{config.model_name}")
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.average_latency = 0.0
        
        # Model state
        self.is_loaded = False
        self.last_error = None
    
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the model. Returns True if successful."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> bool:
        """Unload the model to free resources."""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> SLMResponse:
        """Generate text based on prompt."""
        pass
    
    @abstractmethod
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> List[SLMResponse]:
        """Generate text for multiple prompts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is healthy and responding."""
        pass
    
    # Common functionality
    
    async def classify_problem(self, problem_description: str) -> SLMResponse:
        """Classify a problem description."""
        prompt = self._format_classification_prompt(problem_description)
        return await self.generate(
            prompt=prompt,
            task_type=TaskType.CLASSIFICATION
        )
    
    async def explain_solution(
        self, 
        solution_data: Dict[str, Any],
        original_problem: str,
        methodology: str
    ) -> SLMResponse:
        """Generate natural language explanation of solution."""
        prompt = self._format_explanation_prompt(solution_data, original_problem, methodology)
        return await self.generate(
            prompt=prompt,
            task_type=TaskType.EXPLANATION
        )
    
    async def reason_about_problem(self, problem_description: str) -> SLMResponse:
        """Perform reasoning about a problem."""
        prompt = self._format_reasoning_prompt(problem_description)
        return await self.generate(
            prompt=prompt,
            task_type=TaskType.REASONING
        )
    
    async def generate_code(self, task_description: str, language: str = "python") -> SLMResponse:
        """Generate code for a given task."""
        prompt = self._format_coding_prompt(task_description, language)
        return await self.generate(
            prompt=prompt,
            task_type=TaskType.CODING
        )
    
    def _format_classification_prompt(self, problem_description: str) -> str:
        """Format prompt for problem classification."""
        return f"""Classify this problem and extract parameters.

Problem: {problem_description}

Respond with JSON:
{{
    "category": "dcop|graph_algorithms|optimization|reasoning|coding|general",
    "confidence": 0.0-1.0,
    "extracted_parameters": {{
        "key_concepts": ["concept1", "concept2"],
        "numeric_values": [numbers],
        "constraints": ["constraint1"],
        "objective": "what needs to be solved"
    }},
    "reasoning": "brief explanation"
}}"""
    
    def _format_explanation_prompt(
        self, 
        solution_data: Dict[str, Any], 
        original_problem: str,
        methodology: str
    ) -> str:
        """Format prompt for solution explanation."""
        return f"""Explain this technical solution in clear, natural language.

Original Problem: {original_problem}
Solution: {solution_data}
Method: {methodology}

Provide:
1. What the problem was asking
2. How the solution approach works
3. What the results mean
4. Why this is a good solution

Use clear language that a non-expert can understand."""
    
    def _format_reasoning_prompt(self, problem_description: str) -> str:
        """Format prompt for reasoning tasks."""
        return f"""Solve this problem step by step with clear reasoning.

Problem: {problem_description}

Provide:
1. Problem understanding
2. Step-by-step reasoning
3. Final answer
4. Explanation of why this answer is correct"""
    
    def _format_coding_prompt(self, task_description: str, language: str) -> str:
        """Format prompt for coding tasks."""
        return f"""Write {language} code for this task.

Task: {task_description}

Provide:
1. Working code with comments
2. Explanation of the approach
3. Example usage
4. Time/space complexity if relevant"""
    
    def _track_request(self, response: SLMResponse) -> None:
        """Track request metrics."""
        self.total_requests += 1
        if response.success:
            self.successful_requests += 1
        
        if response.token_count:
            self.total_tokens += response.token_count
        
        if response.cost:
            self.total_cost += response.cost
        
        # Update average latency
        self.average_latency = (
            (self.average_latency * (self.total_requests - 1) + response.execution_time) 
            / self.total_requests
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this client."""
        success_rate = (
            self.successful_requests / self.total_requests * 100 
            if self.total_requests > 0 else 0
        )
        
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type.value,
            "total_requests": self.total_requests,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_latency": self.average_latency,
            "is_loaded": self.is_loaded,
            "last_error": self.last_error
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.average_latency = 0.0