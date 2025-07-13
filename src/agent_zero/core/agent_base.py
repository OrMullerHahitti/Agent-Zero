"""Base agent interface and abstract implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from .message import (
    Message, 
    MessageType, 
    ProblemRequest, 
    ProblemResult, 
    AgentCapability,
    AgentRegistration
)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.is_running = False
        self._message_handlers: Dict[MessageType, callable] = {}
        self._setup_default_handlers()
        
        # Logging
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
        # Performance tracking
        self.start_time = datetime.now()
        self.problems_solved = 0
        self.total_execution_time = 0.0
        
    def _setup_default_handlers(self) -> None:
        """Set up default message handlers."""
        self._message_handlers.update({
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.CAPABILITY_QUERY: self._handle_capability_query,
            MessageType.PROBLEM_SUBMIT: self._handle_problem_submit,
        })
    
    @abstractmethod
    async def get_capabilities(self) -> AgentCapability:
        """Return this agent's capabilities."""
        pass
    
    @abstractmethod
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """
        Determine if this agent can solve the given problem.
        
        Returns:
            Confidence score 0.0-1.0, where 0.0 means cannot solve
        """
        pass
    
    @abstractmethod
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Solve the given problem and return results."""
        pass
    
    async def start(self) -> None:
        """Start the agent."""
        self.is_running = True
        self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the agent."""
        self.is_running = False
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and return response if needed."""
        if not self.is_running:
            return self._create_error_response(
                message, "Agent is not running"
            )
        
        handler = self._message_handlers.get(message.type)
        if not handler:
            return self._create_error_response(
                message, f"No handler for message type: {message.type}"
            )
        
        try:
            return await handler(message)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_heartbeat(self, message: Message) -> Message:
        """Handle heartbeat request."""
        return Message(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={"status": "alive", "uptime": self._get_uptime()}
        )
    
    async def _handle_capability_query(self, message: Message) -> Message:
        """Handle capability query."""
        capabilities = await self.get_capabilities()
        return Message(
            type=MessageType.CAPABILITY_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload=capabilities.dict()
        )
    
    async def _handle_problem_submit(self, message: Message) -> Message:
        """Handle problem submission."""
        try:
            problem_request = ProblemRequest(**message.payload)
            
            # Check if we can solve this problem
            confidence = await self.can_solve_problem(problem_request)
            if confidence == 0.0:
                return self._create_error_response(
                    message, "Cannot solve this type of problem"
                )
            
            # Solve the problem
            start_time = datetime.now()
            result = await self.solve_problem(problem_request)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self.problems_solved += 1
            self.total_execution_time += execution_time
            
            return Message(
                type=MessageType.PROBLEM_RESULT,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.id,
                payload=result.dict()
            )
            
        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
            return self._create_error_response(message, str(e))
    
    def _create_error_response(self, original_message: Message, error: str) -> Message:
        """Create an error response message."""
        return Message(
            type=MessageType.ERROR,
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.id,
            payload={"error": error}
        )
    
    def _get_uptime(self) -> float:
        """Get agent uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        uptime = self._get_uptime()
        avg_execution_time = (
            self.total_execution_time / self.problems_solved 
            if self.problems_solved > 0 else 0.0
        )
        
        return {
            "uptime_seconds": uptime,
            "problems_solved": self.problems_solved,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "problems_per_minute": (self.problems_solved / uptime * 60) if uptime > 0 else 0.0
        }
    
    def get_registration_info(self) -> AgentRegistration:
        """Get agent registration information."""
        return AgentRegistration(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=self.get_capabilities()
        )