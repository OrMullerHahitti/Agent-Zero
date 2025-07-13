"""Together AI API client for remote SLM inference."""

import time
import asyncio
from typing import Dict, List, Optional, Any
import logging

try:
    import aiohttp
    import json
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from ..slm_interface import SLMInterface, SLMConfig, SLMResponse, TaskType


class TogetherClient(SLMInterface):
    """Together AI API client for remote inference."""
    
    def __init__(self, config: SLMConfig):
        if not TOGETHER_AVAILABLE:
            raise ImportError(
                "Together client dependencies not available. Install with: pip install aiohttp"
            )
        
        super().__init__(config)
        
        # API client
        self.session = None
        
        # Rate limiting
        self.requests_per_minute = config.rate_limit or 100
        self.request_times = []
        
        # Cost tracking (Together AI has competitive pricing)
        self.cost_per_token = self._get_cost_per_token()
        
        # API endpoint
        self.api_url = (config.api_base or "https://api.together.xyz/v1") + "/chat/completions"
        
    def _get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per token for different Together AI models."""
        # Costs in USD per 1K tokens (Together AI competitive pricing)
        costs = {
            "meta-llama/llama-3-8b-instruct": {"input": 0.0002, "output": 0.0002},
            "meta-llama/llama-3-70b-instruct": {"input": 0.0009, "output": 0.0009},
            "meta-llama/llama-3.1-8b-instruct": {"input": 0.0002, "output": 0.0002},
            "meta-llama/llama-3.1-70b-instruct": {"input": 0.0009, "output": 0.0009},
            "codellama/codellama-7b-instruct": {"input": 0.0002, "output": 0.0002},
            "codellama/codellama-13b-instruct": {"input": 0.0002, "output": 0.0002},
            "codellama/codellama-34b-instruct": {"input": 0.0008, "output": 0.0008},
            "mistralai/mistral-7b-instruct": {"input": 0.0002, "output": 0.0002},
            "mistralai/mixtral-8x7b-instruct": {"input": 0.0006, "output": 0.0006},
            "togethercomputer/redpajama-incite-7b-chat": {"input": 0.0002, "output": 0.0002}
        }
        
        model_name = self.config.model_name.lower()
        
        # Find exact or partial match
        if model_name in costs:
            return costs[model_name]
        
        for model_key in costs:
            if any(part in model_name for part in model_key.split("/")):
                return costs[model_key]
        
        # Default to Llama-3-8B pricing
        return costs["meta-llama/llama-3-8b-instruct"]
    
    async def load_model(self) -> bool:
        """Initialize the Together AI client."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            self.is_loaded = True
            self.logger.info(f"Together AI client initialized for {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Together AI client: {e}")
            self.last_error = str(e)
            return False
    
    async def unload_model(self) -> bool:
        """Close the Together AI client."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_loaded = False
            self.logger.info("Together AI client closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close Together AI client: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> SLMResponse:
        """Generate text using Together AI API."""
        
        if not self.is_loaded:
            await self.load_model()
        
        if not self.is_loaded:
            return SLMResponse(
                text="",
                success=False,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=0.0,
                error_message="Together AI client not initialized"
            )
        
        # Rate limiting
        await self._rate_limit()
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # API payload
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": False,
                **kwargs
            }
            
            # Headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            async with self.session.post(
                self.api_url,
                json=payload,
                headers=headers
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise Exception(f"API error {response.status}: {response_data}")
                
                execution_time = time.time() - start_time
                
                # Extract response text
                response_text = response_data["choices"][0]["message"]["content"] or ""
                
                # Calculate costs (Together AI often doesn't return usage, estimate)
                estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
                estimated_output_tokens = len(response_text.split()) * 1.3
                
                # Try to get actual usage if available
                usage = response_data.get("usage", {})
                actual_input_tokens = usage.get("prompt_tokens", estimated_input_tokens)
                actual_output_tokens = usage.get("completion_tokens", estimated_output_tokens)
                
                input_cost = (actual_input_tokens / 1000) * self.cost_per_token["input"]
                output_cost = (actual_output_tokens / 1000) * self.cost_per_token["output"]
                total_cost = input_cost + output_cost
                
                # Create response
                slm_response = SLMResponse(
                    text=response_text,
                    success=True,
                    model_name=self.config.model_name,
                    task_type=task_type,
                    execution_time=execution_time,
                    token_count=int(actual_output_tokens),
                    cost=total_cost,
                    tokens_per_second=actual_output_tokens / execution_time if execution_time > 0 else 0
                )
                
                # Track request
                self._track_request(slm_response)
                
                return slm_response
            
        except Exception as e:
            self.logger.error(f"Together AI API error: {e}")
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
        
        # Make concurrent requests
        tasks = [
            self.generate(prompt, system_prompt, task_type, **kwargs)
            for prompt in prompts
        ]
        
        # Limit concurrency to respect rate limits
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
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
        """Get information about the Together AI model."""
        return {
            "model_name": self.config.model_name,
            "provider": "together",
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
        """Check if the Together AI API is accessible."""
        try:
            if not self.is_loaded:
                await self.load_model()
            
            if not self.is_loaded:
                return False
            
            # Simple test request
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                self.api_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return "choices" in data and len(data["choices"]) > 0
                return False
            
        except Exception as e:
            self.logger.error(f"Together AI health check failed: {e}")
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
    
    def list_available_models(self) -> List[str]:
        """List popular models available on Together AI."""
        return [
            "meta-llama/llama-3-8b-instruct",
            "meta-llama/llama-3-70b-instruct", 
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "codellama/codellama-7b-instruct",
            "codellama/codellama-13b-instruct",
            "codellama/codellama-34b-instruct",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "togethercomputer/redpajama-incite-7b-chat"
        ]