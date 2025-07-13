"""Message types and structures for inter-agent communication."""

from enum import Enum
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents."""
    
    # Basic communication
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    
    # Agent lifecycle
    REGISTER = "register"
    UNREGISTER = "unregister"
    HEARTBEAT = "heartbeat"
    
    # Problem solving
    PROBLEM_SUBMIT = "problem_submit"
    PROBLEM_RESULT = "problem_result"
    PROBLEM_STATUS = "problem_status"
    
    # Agent coordination
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    AGENT_DISCOVERY = "agent_discovery"


class Message(BaseModel):
    """Base message structure for agent communication."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast messages
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For request-response tracking
    
    # Message content
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProblemRequest(BaseModel):
    """Request to solve a problem."""
    
    problem_description: str
    problem_type: Optional[str] = None  # Auto-detected if None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None


class ProblemResult(BaseModel):
    """Result of problem solving."""
    
    success: bool
    solution: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None  # 0.0 to 1.0
    execution_time: Optional[float] = None  # seconds
    error_message: Optional[str] = None
    alternative_solutions: Optional[list] = None


class AgentCapability(BaseModel):
    """Description of an agent's capabilities."""
    
    agent_id: str
    agent_type: str
    problem_types: list[str]
    description: str
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    load_factor: float = 0.0  # Current load 0.0-1.0
    availability: bool = True


class AgentRegistration(BaseModel):
    """Agent registration information."""
    
    agent_id: str
    agent_type: str
    capabilities: AgentCapability
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None