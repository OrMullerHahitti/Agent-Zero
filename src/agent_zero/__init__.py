"""
Agent Zero: Distributed Agentic AI System

A sophisticated multi-agent AI system that orchestrates specialized Small Language Models (SLMs) 
for algorithmic problem-solving.
"""

__version__ = "0.1.0"
__author__ = "Or Muller"

from .core.agent_base import BaseAgent
from .core.message import Message, MessageType
from .orchestrator.main_orchestrator import MainOrchestrator

__all__ = [
    "BaseAgent",
    "Message", 
    "MessageType",
    "MainOrchestrator",
]