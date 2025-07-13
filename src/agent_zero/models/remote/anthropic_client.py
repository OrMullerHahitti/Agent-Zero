"""Anthropic API client for remote SLM inference."""

import time
import asyncio
from typing import Dict, List, Optional, Any
import logging

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..slm_interface import SLMInterface, SLMConfig, SLMResponse, TaskType


class AnthropicClient(SLMInterface):
    """Anthropic (Claude) API client for remote inference."""
    
    def __init__(self, config: SLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic client not available. Install with: pip install anthropic"
            )
        
        super().__init__(config)
        
        # API client
        self.client = None
        
        # Rate limiting
        self.requests_per_minute = config.rate_limit or 40
        self.request_times = []
        
        # Cost tracking
        self.cost_per_token = self._get_cost_per_token()
        
    def _get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per token for different Claude models."""
        # Costs in USD per 1K tokens (as of 2024)
        costs = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024},
            "claude-instant": {"input": 0.0008, "output": 0.0024}
        }
        
        model_name = self.config.model_name.lower()
        
        # Find best match
        for model_key in costs:
            if model_key in model_name:
                return costs[model_key]
        
        # Default to Claude-3-Sonnet pricing
        return costs["claude-3-sonnet"]
    
    async def load_model(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            self.client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            
            self.is_loaded = True
            self.logger.info(f"Anthropic client initialized for {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            self.last_error = str(e)
            return False
    
    async def unload_model(self) -> bool:
        """Close the Anthropic client."""
        try:
            if self.client:
                await self.client.close()
                self.client = None
            
            self.is_loaded = False
            self.logger.info("Anthropic client closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close Anthropic client: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> SLMResponse:
        """Generate text using Anthropic API."""
        
        if not self.is_loaded:
            await self.load_model()
        
        if not self.is_loaded:
            return SLMResponse(
                text="",
                success=False,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=0.0,
                error_message="Anthropic client not initialized"
            )
        
        # Rate limiting
        await self._rate_limit()
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # API parameters
            api_params = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **kwargs
            }
            
            # Add system prompt if provided
            if system_prompt:
                api_params["system"] = system_prompt
            
            # Make API call
            response = await self.client.messages.create(**api_params)
            
            execution_time = time.time() - start_time
            
            # Extract response text
            response_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    response_text += content_block.text
            
            # Calculate costs
            usage = response.usage
            input_cost = (usage.input_tokens / 1000) * self.cost_per_token["input"]
            output_cost = (usage.output_tokens / 1000) * self.cost_per_token["output"]
            total_cost = input_cost + output_cost
            
            # Create response
            slm_response = SLMResponse(
                text=response_text,
                success=True,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=execution_time,
                token_count=usage.output_tokens,
                cost=total_cost,
                tokens_per_second=usage.output_tokens / execution_time if execution_time > 0 else 0
            )
            
            # Track request
            self._track_request(slm_response)
            
            return slm_response
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return SLMResponse(
                text="",
                success=False,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> List[SLMResponse]:
        """Generate text for multiple prompts."""
        
        # Anthropic doesn't have native batch support, so we'll make concurrent requests
        tasks = [
            self.generate(prompt, system_prompt, task_type, **kwargs)
            for prompt in prompts
        ]
        
        # Limit concurrency to respect rate limits
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        
        async def limited_generate(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_generate(task) for task in tasks]
        responses = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append(SLMResponse(
                    text="",
                    success=False,
                    model_name=self.config.model_name,
                    task_type=task_type,
                    execution_time=0.0,
                    error_message=str(response)
                ))
            else:
                results.append(response)
        
        return results
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            req_time for req_time in self.request_times 
            if current_time - req_time < 60
        ]
        
        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Anthropic model."""
        return {
            "model_name": self.config.model_name,
            "provider": "anthropic",
            "api_base": self.config.api_base,
            "is_loaded": self.is_loaded,
            "parameters": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            },
            "cost_tracking": {
                "cost_per_1k_input_tokens": self.cost_per_token["input"],
                "cost_per_1k_output_tokens": self.cost_per_token["output"],
                "total_cost": self.total_cost
            },
            "rate_limiting": {
                "requests_per_minute": self.requests_per_minute,
                "recent_requests": len(self.request_times)
            },
            "performance": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "average_latency": self.average_latency,
                "total_tokens": self.total_tokens
            }
        }
    
    async def health_check(self) -> bool:
        """Check if the Anthropic API is accessible."""
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not self.is_loaded:
                return False
            
            # Simple test request
            response = await self.client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return len(response.content) > 0
            
        except Exception as e:
            self.logger.error(f"Anthropic health check failed: {e}")
            return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.cost_per_token["input"]
        output_cost = (output_tokens / 1000) * self.cost_per_token["output"]
        return input_cost + output_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.total_requests - self.successful_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.total_cost / self.total_requests if self.total_requests > 0 else 0,
            "average_tokens_per_request": self.total_tokens / self.total_requests if self.total_requests > 0 else 0,
            "average_latency": self.average_latency
        }