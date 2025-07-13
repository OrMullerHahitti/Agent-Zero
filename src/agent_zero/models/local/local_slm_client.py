"""Local SLM client for inference with optimized performance."""

import time
import asyncio
from typing import Dict, List, Optional, Any
import logging
import torch
from concurrent.futures import ThreadPoolExecutor

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GenerationConfig, StoppingCriteria, StoppingCriteriaList
    )
    from peft import PeftModel
    import torch.nn.functional as F
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

from ..slm_interface import SLMInterface, SLMConfig, SLMResponse, TaskType


class LocalSLMClient(SLMInterface):
    """Local SLM client for inference using transformers."""
    
    def __init__(self, config: SLMConfig):
        if not INFERENCE_AVAILABLE:
            raise ImportError(
                "Inference dependencies not available. Install with: "
                "pip install transformers torch peft accelerate"
            )
        
        super().__init__(config)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # Performance optimization
        self.device = self._determine_device()
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Caching
        self.cache_enabled = True
        self.prompt_cache = {}
        self.max_cache_size = 100
        
        # Threading for async inference
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.inference_times = []
        self.token_counts = []
        
    def _determine_device(self) -> str:
        """Determine the best device for inference."""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_model(self) -> bool:
        """Load the model for inference."""
        try:
            self.logger.info(f"Loading model from: {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                padding_side="left"  # For batch inference
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self.dtype,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            # Add quantization if specified
            if self.config.quantization:
                if self.config.quantization == "8bit":
                    model_kwargs["load_in_8bit"] = True
                elif self.config.quantization == "4bit":
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.dtype
                    )
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Check if this is a LoRA model
            try:
                # Try to load as PEFT model (LoRA)
                self.model = PeftModel.from_pretrained(self.model, self.config.model_path)
                self.logger.info("Loaded as LoRA/PEFT model")
            except:
                # Regular model, no PEFT adapters
                pass
            
            # Move to device if not using device_map
            if self.device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
            self.is_loaded = True
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.last_error = str(e)
            return False
    
    async def unload_model(self) -> bool:
        """Unload the model to free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear cache
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            self.logger.info("Model unloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload model: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        **kwargs
    ) -> SLMResponse:
        """Generate text for a single prompt."""
        
        if not self.is_loaded:
            await self.load_model()
        
        if not self.is_loaded:
            return SLMResponse(
                text="",
                success=False,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=0.0,
                error_message="Model not loaded"
            )
        
        start_time = time.time()
        
        try:
            # Format prompt
            formatted_prompt = self._format_prompt(prompt, system_prompt, task_type)
            
            # Check cache
            cache_key = self._get_cache_key(formatted_prompt, kwargs)
            if self.cache_enabled and cache_key in self.prompt_cache:
                cached_response = self.prompt_cache[cache_key]
                cached_response.execution_time = time.time() - start_time
                return cached_response
            
            # Generate response
            response_text = await self._generate_text(formatted_prompt, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Create response
            response = SLMResponse(
                text=response_text,
                success=True,
                model_name=self.config.model_name,
                task_type=task_type,
                execution_time=execution_time,
                token_count=len(self.tokenizer.encode(response_text)),
                tokens_per_second=len(self.tokenizer.encode(response_text)) / execution_time if execution_time > 0 else 0
            )
            
            # Cache response
            if self.cache_enabled:
                self._cache_response(cache_key, response)
            
            # Track performance
            self._track_request(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
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
        """Generate text for multiple prompts in batch."""
        
        if not self.is_loaded:
            await self.load_model()
        
        if not self.is_loaded:
            return [
                SLMResponse(
                    text="",
                    success=False,
                    model_name=self.config.model_name,
                    task_type=task_type,
                    execution_time=0.0,
                    error_message="Model not loaded"
                ) for _ in prompts
            ]
        
        start_time = time.time()
        
        try:
            # Format all prompts
            formatted_prompts = [
                self._format_prompt(prompt, system_prompt, task_type) 
                for prompt in prompts
            ]
            
            # Batch generation
            response_texts = await self._generate_batch_text(formatted_prompts, **kwargs)
            
            execution_time = time.time() - start_time
            avg_execution_time = execution_time / len(prompts)
            
            # Create responses
            responses = []
            for i, response_text in enumerate(response_texts):
                response = SLMResponse(
                    text=response_text,
                    success=True,
                    model_name=self.config.model_name,
                    task_type=task_type,
                    execution_time=avg_execution_time,
                    token_count=len(self.tokenizer.encode(response_text)),
                    tokens_per_second=len(self.tokenizer.encode(response_text)) / avg_execution_time if avg_execution_time > 0 else 0
                )
                responses.append(response)
                self._track_request(response)
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch generation error: {e}")
            return [
                SLMResponse(
                    text="",
                    success=False,
                    model_name=self.config.model_name,
                    task_type=task_type,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ) for _ in prompts
            ]
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str], task_type: TaskType) -> str:
        """Format prompt for the model."""
        
        # Use chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback to manual formatting
                pass
        
        # Manual formatting
        if system_prompt:
            return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    async def _generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text for a single prompt."""
        
        def _sync_generate():
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.model_max_length - self.config.max_tokens
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    **kwargs
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _sync_generate)
    
    async def _generate_batch_text(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts in batch."""
        
        def _sync_batch_generate():
            # Tokenize all prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.model_max_length - self.config.max_tokens
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    **kwargs
                )
            
            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                generated_tokens = output[inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())
            
            return responses
        
        # Run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _sync_batch_generate)
    
    def _get_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Generate cache key for prompt and parameters."""
        import hashlib
        
        cache_data = {
            "prompt": prompt,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, response: SLMResponse) -> None:
        """Cache a response."""
        if len(self.prompt_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        
        self.prompt_cache[cache_key] = response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantization": self.config.quantization,
            "parameters": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k
            },
            "performance": {
                "cache_size": len(self.prompt_cache),
                "average_latency": self.average_latency,
                "total_requests": self.total_requests
            }
        }
        
        # Add model size info if available
        if self.model is not None:
            try:
                param_count = sum(p.numel() for p in self.model.parameters())
                info["parameter_count"] = param_count
                info["parameter_count_human"] = self._format_param_count(param_count)
            except:
                pass
        
        return info
    
    def _format_param_count(self, count: int) -> str:
        """Format parameter count in human readable form."""
        if count >= 1e9:
            return f"{count/1e9:.1f}B"
        elif count >= 1e6:
            return f"{count/1e6:.1f}M"
        elif count >= 1e3:
            return f"{count/1e3:.1f}K"
        else:
            return str(count)
    
    async def health_check(self) -> bool:
        """Check if the model is healthy and responding."""
        try:
            if not self.is_loaded:
                return False
            
            # Simple test generation
            test_prompt = "Hello"
            response = await self.generate(test_prompt, task_type=TaskType.GENERAL)
            
            return response.success and len(response.text) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self.prompt_cache.clear()
        self.logger.info("Prompt cache cleared")
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def optimize_for_inference(self) -> None:
        """Apply inference optimizations."""
        if not self.is_loaded:
            return
        
        try:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled for faster inference")
            
            # Enable inference mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
        except Exception as e:
            self.logger.warning(f"Could not apply all optimizations: {e}")