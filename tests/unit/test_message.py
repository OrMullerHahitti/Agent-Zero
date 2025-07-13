"""Tests for message types and structures."""

import pytest
from datetime import datetime
from agent_zero.core.message import (
    Message, 
    MessageType, 
    ProblemRequest, 
    ProblemResult,
    AgentCapability
)


class TestMessage:
    """Test Message class."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = Message(
            type=MessageType.REQUEST,
            sender_id="agent1",
            recipient_id="agent2",
            payload={"test": "data"}
        )
        
        assert message.type == MessageType.REQUEST
        assert message.sender_id == "agent1"
        assert message.recipient_id == "agent2"
        assert message.payload == {"test": "data"}
        assert message.id is not None
        assert isinstance(message.timestamp, datetime)
    
    def test_message_without_recipient(self):
        """Test broadcast message (no recipient)."""
        message = Message(
            type=MessageType.HEARTBEAT,
            sender_id="agent1",
            payload={}
        )
        
        assert message.recipient_id is None


class TestProblemRequest:
    """Test ProblemRequest class."""
    
    def test_problem_request_creation(self):
        """Test basic problem request creation."""
        request = ProblemRequest(
            problem_description="Solve this optimization problem"
        )
        
        assert request.problem_description == "Solve this optimization problem"
        assert request.problem_type is None
        assert request.constraints == {}
        assert request.preferences == {}
        assert request.timeout_seconds is None
    
    def test_problem_request_with_details(self):
        """Test problem request with all details."""
        request = ProblemRequest(
            problem_description="DCOP with 5 variables",
            problem_type="dcop",
            constraints={"num_variables": 5, "domain_size": 3},
            preferences={"algorithm": "min_sum"},
            timeout_seconds=60
        )
        
        assert request.problem_type == "dcop"
        assert request.constraints["num_variables"] == 5
        assert request.preferences["algorithm"] == "min_sum"
        assert request.timeout_seconds == 60


class TestProblemResult:
    """Test ProblemResult class."""
    
    def test_successful_result(self):
        """Test successful problem result."""
        result = ProblemResult(
            success=True,
            solution={"var1": 0, "var2": 1},
            explanation="Found optimal solution",
            confidence=0.95,
            execution_time=1.23
        )
        
        assert result.success is True
        assert result.solution == {"var1": 0, "var2": 1}
        assert result.explanation == "Found optimal solution"
        assert result.confidence == 0.95
        assert result.execution_time == 1.23
        assert result.error_message is None
    
    def test_failed_result(self):
        """Test failed problem result."""
        result = ProblemResult(
            success=False,
            error_message="No solution found"
        )
        
        assert result.success is False
        assert result.error_message == "No solution found"
        assert result.solution is None


class TestAgentCapability:
    """Test AgentCapability class."""
    
    def test_agent_capability_creation(self):
        """Test agent capability creation."""
        capability = AgentCapability(
            agent_id="bp_agent",
            agent_type="belief_propagation",
            problem_types=["dcop", "constraint_optimization"],
            description="Solves DCOP problems using belief propagation"
        )
        
        assert capability.agent_id == "bp_agent"
        assert capability.agent_type == "belief_propagation"
        assert "dcop" in capability.problem_types
        assert capability.availability is True
        assert capability.load_factor == 0.0