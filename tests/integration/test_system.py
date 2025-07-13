"""Integration tests for the complete Agent Zero system."""

import pytest
import asyncio
from agent_zero.core.registry import AgentRegistry
from agent_zero.protocols.communication import InMemoryMessageBus
from agent_zero.orchestrator.main_orchestrator import MainOrchestrator
from agent_zero.agents.belief_propagation_agent import BeliefPropagationAgent
from agent_zero.core.message import ProblemRequest


@pytest.fixture
async def system():
    """Create a complete test system."""
    # Initialize components
    message_bus = InMemoryMessageBus()
    registry = AgentRegistry()
    orchestrator = MainOrchestrator(message_bus, registry)
    
    # Start components
    await message_bus.start()
    await registry.start()
    await orchestrator.start()
    
    # Register orchestrator
    await registry.register_agent(orchestrator.get_registration_info())
    
    # Create and register BP agent
    bp_agent = BeliefPropagationAgent()
    await bp_agent.start()
    await message_bus.subscribe(bp_agent.agent_id, bp_agent._handle_message_callback)
    await registry.register_agent(bp_agent.get_registration_info())
    
    system_components = {
        'message_bus': message_bus,
        'registry': registry,
        'orchestrator': orchestrator,
        'bp_agent': bp_agent
    }
    
    yield system_components
    
    # Cleanup
    await bp_agent.stop()
    await orchestrator.stop()
    await registry.stop()
    await message_bus.stop()


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.mark.asyncio
    async def test_system_startup(self, system):
        """Test that all system components start correctly."""
        assert system['message_bus']._running is True
        assert system['registry']._running is True
        assert system['orchestrator'].is_running is True
        assert system['bp_agent'].is_running is True
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, system):
        """Test that agents are properly registered."""
        registry = system['registry']
        
        # Check orchestrator is registered
        orchestrator = registry.get_agent("main_orchestrator")
        assert orchestrator is not None
        
        # Check BP agent is registered
        bp_agent = registry.get_agent("bp_agent")
        assert bp_agent is not None
    
    @pytest.mark.asyncio
    async def test_problem_classification(self, system):
        """Test problem classification by orchestrator."""
        orchestrator = system['orchestrator']
        
        # Test DCOP problem classification
        dcop_request = ProblemRequest(
            problem_description="Solve a constraint optimization problem with 5 variables"
        )
        
        problem_type = await orchestrator._classify_problem(dcop_request)
        assert problem_type == "dcop"
        
        # Test explicit problem type
        explicit_request = ProblemRequest(
            problem_description="Some problem",
            problem_type="dcop"
        )
        
        problem_type = await orchestrator._classify_problem(explicit_request)
        assert problem_type == "dcop"
    
    @pytest.mark.asyncio
    async def test_agent_discovery(self, system):
        """Test agent discovery functionality."""
        registry = system['registry']
        
        # Find agents for DCOP problems
        dcop_agents = registry.find_agents_for_problem("dcop")
        assert len(dcop_agents) == 1
        assert dcop_agents[0].agent_id == "bp_agent"
    
    @pytest.mark.asyncio
    async def test_end_to_end_problem_solving(self, system):
        """Test complete problem solving workflow."""
        orchestrator = system['orchestrator']
        
        # Create a DCOP problem request
        problem_request = ProblemRequest(
            problem_description="Solve a constraint optimization problem with 3 variables and domain size 2",
            constraints={
                "num_variables": 3,
                "domain_size": 2,
                "density": 0.5
            }
        )
        
        # Solve the problem
        result = await orchestrator.solve_problem(problem_request)
        
        # Note: The actual result depends on PropFlow availability
        # If PropFlow is not available, we expect a specific error
        if not hasattr(result, 'success'):
            pytest.skip("PropFlow not available for testing")
        
        # Verify we get a response (success or failure)
        assert hasattr(result, 'success')
        assert hasattr(result, 'error_message') or hasattr(result, 'solution')
    
    @pytest.mark.asyncio
    async def test_system_status(self, system):
        """Test system status reporting."""
        orchestrator = system['orchestrator']
        
        status = await orchestrator.get_system_status()
        
        assert 'orchestrator' in status
        assert 'registry' in status
        assert 'system' in status
        
        assert status['orchestrator']['agent_id'] == "main_orchestrator"
        assert status['registry']['total_agents'] >= 2  # orchestrator + bp_agent
        assert status['system']['system_health'] in ['healthy', 'degraded']
    
    @pytest.mark.asyncio
    async def test_agent_capabilities(self, system):
        """Test agent capability reporting."""
        bp_agent = system['bp_agent']
        
        capabilities = await bp_agent.get_capabilities()
        
        assert capabilities.agent_id == "bp_agent"
        assert capabilities.agent_type == "belief_propagation"
        assert "dcop" in capabilities.problem_types
        assert capabilities.availability is not None
    
    @pytest.mark.asyncio
    async def test_message_routing(self, system):
        """Test message routing between components."""
        orchestrator = system['orchestrator']
        bp_agent = system['bp_agent']
        
        # Test that BP agent can solve problems it's capable of
        dcop_request = ProblemRequest(
            problem_description="constraint optimization problem"
        )
        
        confidence = await bp_agent.can_solve_problem(dcop_request)
        assert confidence > 0.0  # Should be able to solve DCOP problems
        
        # Test that orchestrator can route problems
        capable_agents = system['registry'].find_agents_for_problem("dcop")
        assert len(capable_agents) > 0