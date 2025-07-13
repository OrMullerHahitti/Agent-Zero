"""Factory for creating SLM clients based on configuration."""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .slm_interface import SLMInterface, SLMConfig, ModelType, TaskType
from .config_manager import ModelConfigManager


class ModelFactory:
    """Factory for creating appropriate SLM clients."""
    
    _local_clients: Dict[str, type] = {}
    _remote_clients: Dict[str, type] = {}
    _instances: Dict[str, SLMInterface] = {}
    
    @classmethod
    def register_local_client(cls, provider: str, client_class: type):
        """Register a local SLM client."""
        cls._local_clients[provider] = client_class
    
    @classmethod
    def register_remote_client(cls, provider: str, client_class: type):
        """Register a remote API client."""
        cls._remote_clients[provider] = client_class
    
    @classmethod
    def create(
        cls, 
        model_config: SLMConfig,
        task_type: Optional[TaskType] = None,
        singleton: bool = True
    ) -> SLMInterface:
        """Create an SLM client based on configuration."""
        
        # Create cache key
        cache_key = f"{model_config.model_type.value}:{model_config.model_name}"
        if task_type:
            cache_key += f":{task_type.value}"
        
        # Return existing instance if singleton requested
        if singleton and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        client = cls._create_client(model_config, task_type)
        
        if singleton:
            cls._instances[cache_key] = client
        
        return client
    
    @classmethod
    def _create_client(cls, config: SLMConfig, task_type: Optional[TaskType]) -> SLMInterface:
        """Create the appropriate client instance."""
        
        if config.model_type == ModelType.LOCAL:
            return cls._create_local_client(config, task_type)
        elif config.model_type == ModelType.REMOTE:
            return cls._create_remote_client(config, task_type)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @classmethod
    def _create_local_client(cls, config: SLMConfig, task_type: Optional[TaskType]) -> SLMInterface:
        """Create a local SLM client."""
        
        # Import local clients (lazy import to avoid dependencies if not used)
        try:
            from .local.local_slm_client import LocalSLMClient
            cls._local_clients["transformers"] = LocalSLMClient
        except ImportError as e:
            logging.error(f"Failed to import local SLM client: {e}")
            raise ImportError("Local SLM dependencies not available. Install with: pip install transformers torch")
        
        # Default to transformers-based client
        provider = "transformers"
        
        if provider not in cls._local_clients:
            available = list(cls._local_clients.keys())
            raise ValueError(f"Local provider '{provider}' not available. Available: {available}")
        
        client_class = cls._local_clients[provider]
        return client_class(config)
    
    @classmethod
    def _create_remote_client(cls, config: SLMConfig, task_type: Optional[TaskType]) -> SLMInterface:
        """Create a remote API client."""
        
        # Import remote clients (lazy import)
        try:
            from .remote.openai_client import OpenAIClient
            from .remote.anthropic_client import AnthropicClient
            from .remote.together_client import TogetherClient
            
            cls._remote_clients["openai"] = OpenAIClient
            cls._remote_clients["anthropic"] = AnthropicClient
            cls._remote_clients["together"] = TogetherClient
        except ImportError as e:
            logging.warning(f"Some remote clients not available: {e}")
        
        # Determine provider from model name
        provider = cls._detect_provider(config.model_name)
        
        if provider not in cls._remote_clients:
            available = list(cls._remote_clients.keys())
            raise ValueError(f"Remote provider '{provider}' not available. Available: {available}")
        
        client_class = cls._remote_clients[provider]
        return client_class(config)
    
    @classmethod
    def _detect_provider(cls, model_name: str) -> str:
        """Detect provider from model name."""
        model_name_lower = model_name.lower()
        
        if any(name in model_name_lower for name in ["gpt", "text-davinci", "text-curie"]):
            return "openai"
        elif any(name in model_name_lower for name in ["claude", "anthropic"]):
            return "anthropic"
        elif any(name in model_name_lower for name in ["llama", "mistral", "code", "together"]):
            return "together"
        else:
            # Default fallback
            return "openai"
    
    @classmethod
    def create_from_config(
        cls, 
        config_path: str, 
        task_type: Optional[TaskType] = None
    ) -> SLMInterface:
        """Create client from configuration file."""
        config_manager = ModelConfigManager(config_path)
        model_config = config_manager.get_model_config(task_type)
        return cls.create(model_config, task_type)
    
    @classmethod
    def create_for_task(
        cls, 
        task_type: TaskType,
        prefer_local: bool = True,
        model_size: str = "medium"
    ) -> SLMInterface:
        """Create optimal client for specific task type."""
        
        # Task-specific model recommendations
        task_models = {
            TaskType.CLASSIFICATION: {
                "local": {"small": "llama-3.1-8b", "medium": "llama-3.1-8b", "large": "llama-3.1-70b"},
                "remote": {"small": "gpt-4o-mini", "medium": "gpt-4o", "large": "gpt-4o"}
            },
            TaskType.EXPLANATION: {
                "local": {"small": "codellama-7b", "medium": "codellama-13b", "large": "codellama-34b"},
                "remote": {"small": "gpt-4o-mini", "medium": "gpt-4o", "large": "gpt-4o"}
            },
            TaskType.REASONING: {
                "local": {"small": "llama-3.1-8b", "medium": "llama-3.1-8b", "large": "llama-3.1-70b"},
                "remote": {"small": "claude-3-haiku", "medium": "claude-3-sonnet", "large": "claude-3-opus"}
            },
            TaskType.CODING: {
                "local": {"small": "codellama-7b", "medium": "codellama-13b", "large": "codellama-34b"},
                "remote": {"small": "gpt-4o-mini", "medium": "gpt-4o", "large": "gpt-4o"}
            }
        }
        
        # Get recommended model
        models = task_models.get(task_type, task_models[TaskType.REASONING])
        model_type = "local" if prefer_local else "remote"
        model_name = models[model_type][model_size]
        
        # Create configuration
        config = SLMConfig(
            model_type=ModelType.LOCAL if prefer_local else ModelType.REMOTE,
            model_name=model_name,
            model_path=f"./models/{model_name}" if prefer_local else None
        )
        
        return cls.create(config, task_type)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Any]:
        """List all available models and providers."""
        return {
            "local_providers": list(cls._local_clients.keys()),
            "remote_providers": list(cls._remote_clients.keys()),
            "active_instances": list(cls._instances.keys())
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances."""
        cls._instances.clear()
    
    @classmethod
    async def shutdown_all(cls):
        """Shutdown all active instances."""
        for instance in cls._instances.values():
            try:
                await instance.unload_model()
            except Exception as e:
                logging.warning(f"Error shutting down model: {e}")
        cls._instances.clear()


# Auto-register clients when imported
def _auto_register_clients():
    """Auto-register available clients."""
    try:
        # Try to register local clients
        from .local.local_slm_client import LocalSLMClient
        ModelFactory.register_local_client("transformers", LocalSLMClient)
    except ImportError:
        pass
    
    try:
        # Try to register remote clients
        from .remote.openai_client import OpenAIClient
        from .remote.anthropic_client import AnthropicClient
        from .remote.together_client import TogetherClient
        
        ModelFactory.register_remote_client("openai", OpenAIClient)
        ModelFactory.register_remote_client("anthropic", AnthropicClient)
        ModelFactory.register_remote_client("together", TogetherClient)
    except ImportError:
        pass


# Auto-register on import
_auto_register_clients()