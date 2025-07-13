"""
Agent Zero Models Package

Provides both local and remote SLM clients with a unified interface.
Supports local fine-tuned models and remote API services.
"""

from .slm_interface import SLMInterface, SLMConfig, SLMResponse
from .model_factory import ModelFactory
from .config_manager import ModelConfigManager

__all__ = [
    "SLMInterface",
    "SLMConfig", 
    "SLMResponse",
    "ModelFactory",
    "ModelConfigManager"
]