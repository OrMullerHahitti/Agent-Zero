"""Configuration management for SLM models."""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from .slm_interface import SLMConfig, ModelType, TaskType


class ModelConfigManager:
    """Manages model configurations and environment-based selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config_data = self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        # Look for config in several locations
        possible_paths = [
            Path("models_config.yaml"),
            Path("config/models.yaml"),
            Path.home() / ".agent_zero" / "models.yaml",
            Path(__file__).parent / "configs" / "default.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return default path for creation
        return Path("models_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        default_config = {
            "models": {
                "classification": {
                    "local": {
                        "model_name": "llama-3.1-8b-instruct",
                        "model_path": "./models/llama-3.1-8b-instruct",
                        "quantization": "4bit",
                        "max_tokens": 1024,
                        "temperature": 0.3
                    },
                    "remote": {
                        "model_name": "gpt-4o-mini",
                        "api_key": "${OPENAI_API_KEY}",
                        "max_tokens": 1024,
                        "temperature": 0.3
                    }
                },
                "explanation": {
                    "local": {
                        "model_name": "codellama-7b-instruct",
                        "model_path": "./models/codellama-7b-instruct", 
                        "quantization": "4bit",
                        "max_tokens": 2048,
                        "temperature": 0.7
                    },
                    "remote": {
                        "model_name": "gpt-4o",
                        "api_key": "${OPENAI_API_KEY}",
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                "reasoning": {
                    "local": {
                        "model_name": "llama-3.1-8b-instruct",
                        "model_path": "./models/llama-3.1-8b-instruct",
                        "quantization": "4bit",
                        "max_tokens": 2048,
                        "temperature": 0.7
                    },
                    "remote": {
                        "model_name": "claude-3-sonnet",
                        "api_key": "${ANTHROPIC_API_KEY}",
                        "max_tokens": 2048,
                        "temperature": 0.7
                    }
                },
                "coding": {
                    "local": {
                        "model_name": "codellama-7b-instruct",
                        "model_path": "./models/codellama-7b-instruct",
                        "quantization": "4bit",
                        "max_tokens": 2048,
                        "temperature": 0.2
                    },
                    "remote": {
                        "model_name": "gpt-4o",
                        "api_key": "${OPENAI_API_KEY}",
                        "max_tokens": 2048,
                        "temperature": 0.2
                    }
                }
            },
            "preferences": {
                "default_mode": "local",  # "local" or "remote"
                "fallback_to_remote": True,
                "enable_caching": True,
                "log_level": "INFO"
            },
            "hardware": {
                "device": "auto",
                "gpu_memory_fraction": 0.8,
                "cpu_threads": 4
            },
            "remote_apis": {
                "openai": {
                    "api_base": "https://api.openai.com/v1",
                    "rate_limit": 60,
                    "timeout": 30
                },
                "anthropic": {
                    "api_base": "https://api.anthropic.com",
                    "rate_limit": 40,
                    "timeout": 30
                },
                "together": {
                    "api_base": "https://api.together.xyz/v1",
                    "rate_limit": 100,
                    "timeout": 30
                }
            }
        }
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config to {self.config_path}: {e}")
    
    def get_model_config(self, task_type: Optional[TaskType] = None) -> SLMConfig:
        """Get model configuration for specific task type."""
        
        # Determine task type
        if task_type is None:
            task_type = TaskType.GENERAL
        
        task_name = task_type.value
        if task_name not in self.config_data["models"]:
            task_name = "reasoning"  # fallback
        
        # Determine mode (local vs remote)
        mode = self._get_preferred_mode(task_type)
        
        # Get task-specific config
        task_config = self.config_data["models"][task_name][mode]
        
        # Create SLM config
        model_type = ModelType.LOCAL if mode == "local" else ModelType.REMOTE
        
        config = SLMConfig(
            model_type=model_type,
            model_name=task_config["model_name"],
            model_path=task_config.get("model_path"),
            api_key=self._resolve_env_var(task_config.get("api_key")),
            api_base=self._get_api_base(task_config.get("model_name")),
            max_tokens=task_config.get("max_tokens", 2048),
            temperature=task_config.get("temperature", 0.7),
            top_p=task_config.get("top_p", 0.9),
            top_k=task_config.get("top_k", 50),
            device=self.config_data["hardware"].get("device", "auto"),
            quantization=task_config.get("quantization"),
            timeout=self.config_data["remote_apis"].get("openai", {}).get("timeout", 30)
        )
        
        return config
    
    def _get_preferred_mode(self, task_type: TaskType) -> str:
        """Get preferred mode (local/remote) for task type."""
        
        # Check environment variable override
        env_override = os.getenv("AGENT_ZERO_MODEL_MODE")
        if env_override in ["local", "remote"]:
            return env_override
        
        # Check if local model exists
        default_mode = self.config_data["preferences"]["default_mode"]
        fallback_enabled = self.config_data["preferences"]["fallback_to_remote"]
        
        if default_mode == "local":
            # Check if local model is available
            task_name = task_type.value if task_type.value in self.config_data["models"] else "reasoning"
            local_config = self.config_data["models"][task_name]["local"]
            model_path = local_config.get("model_path")
            
            if model_path and Path(model_path).exists():
                return "local"
            elif fallback_enabled:
                return "remote"
            else:
                return "local"  # Will fail, but user preference
        
        return default_mode
    
    def _resolve_env_var(self, value: Optional[str]) -> Optional[str]:
        """Resolve environment variable in configuration value."""
        if not value:
            return None
        
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var)
        
        return value
    
    def _get_api_base(self, model_name: Optional[str]) -> Optional[str]:
        """Get API base URL for model."""
        if not model_name:
            return None
        
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower or "text-" in model_name_lower:
            return self.config_data["remote_apis"]["openai"]["api_base"]
        elif "claude" in model_name_lower:
            return self.config_data["remote_apis"]["anthropic"]["api_base"]
        elif any(name in model_name_lower for name in ["llama", "mistral", "together"]):
            return self.config_data["remote_apis"]["together"]["api_base"]
        
        return None
    
    def update_model_config(self, task_type: TaskType, mode: str, **kwargs) -> None:
        """Update model configuration."""
        task_name = task_type.value
        if task_name not in self.config_data["models"]:
            self.config_data["models"][task_name] = {"local": {}, "remote": {}}
        
        if mode not in self.config_data["models"][task_name]:
            self.config_data["models"][task_name][mode] = {}
        
        self.config_data["models"][task_name][mode].update(kwargs)
        self._save_config(self.config_data)
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a preference value."""
        self.config_data["preferences"][key] = value
        self._save_config(self.config_data)
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        return self.config_data["preferences"].get(key, default)
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all configured models."""
        models = {}
        for task_type in self.config_data["models"]:
            models[task_type] = {
                "local": self.config_data["models"][task_type]["local"]["model_name"],
                "remote": self.config_data["models"][task_type]["remote"]["model_name"]
            }
        return models
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        issues = []
        
        # Check local model paths
        for task_type, configs in self.config_data["models"].items():
            local_config = configs.get("local", {})
            model_path = local_config.get("model_path")
            if model_path and not Path(model_path).exists():
                issues.append(f"Local model path for {task_type} does not exist: {model_path}")
        
        # Check API keys
        for task_type, configs in self.config_data["models"].items():
            remote_config = configs.get("remote", {})
            api_key = self._resolve_env_var(remote_config.get("api_key"))
            if not api_key:
                issues.append(f"API key for {task_type} remote model not configured")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_path": str(self.config_path)
        }