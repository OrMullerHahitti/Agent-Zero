"""Tests for agent registry."""

import pytest
import asyncio
from agent_zero.core.registry import AgentRegistry
from agent_zero.core.message import AgentCapability, AgentRegistration


@pytest.fixture
async def registry():
    """Create a test registry."""
    reg = AgentRegistry(heartbeat_interval=1, timeout_threshold=3)
    await reg.start()
    yield reg
    await reg.stop()


@pytest.fixture
def sample_capability():
    """Create a sample agent capability."""
    return AgentCapability(
        agent_id="test_agent",
        agent_type="test",
        problem_types=["test_problem"],
        description="Test agent"
    )


@pytest.fixture
def sample_registration(sample_capability):
    """Create a sample agent registration."""
    return AgentRegistration(
        agent_id="test_agent",
        agent_type="test",
        capabilities=sample_capability
    )


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    @pytest.mark.asyncio
    async def test_register_agent(self, registry, sample_registration):
        """Test agent registration."""
        success = await registry.register_agent(sample_registration)
        assert success is True
        
        agent = registry.get_agent("test_agent")
        assert agent is not None
        assert agent.agent_id == "test_agent"
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, registry, sample_registration):
        """Test agent unregistration."""
        await registry.register_agent(sample_registration)
        
        success = await registry.unregister_agent("test_agent")
        assert success is True
        
        agent = registry.get_agent("test_agent")
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_find_agents_for_problem(self, registry, sample_registration):
        """Test finding agents for specific problem types."""
        await registry.register_agent(sample_registration)
        
        agents = registry.find_agents_for_problem("test_problem")
        assert len(agents) == 1
        assert agents[0].agent_id == "test_agent"
        
        # Test non-existent problem type
        agents = registry.find_agents_for_problem("nonexistent")
        assert len(agents) == 0
    
    @pytest.mark.asyncio
    async def test_heartbeat_update(self, registry, sample_registration):
        """Test heartbeat updates."""
        await registry.register_agent(sample_registration)
        
        success = await registry.update_heartbeat("test_agent")
        assert success is True
        
        # Test non-existent agent
        success = await registry.update_heartbeat("nonexistent")
        assert success is False
    
    def test_get_all_agents(self, registry, sample_registration):
        """Test getting all agents."""
        asyncio.run(registry.register_agent(sample_registration))
        
        agents = registry.get_all_agents()
        assert len(agents) == 1
        assert agents[0].agent_id == "test_agent"
    
    def test_get_agents_by_type(self, registry, sample_registration):
        """Test getting agents by type."""
        asyncio.run(registry.register_agent(sample_registration))
        
        agents = registry.get_agents_by_type("test")
        assert len(agents) == 1
        
        agents = registry.get_agents_by_type("nonexistent")
        assert len(agents) == 0
    
    def test_get_problem_types(self, registry, sample_registration):
        """Test getting supported problem types."""
        asyncio.run(registry.register_agent(sample_registration))
        
        problem_types = registry.get_problem_types()
        assert "test_problem" in problem_types
    
    def test_registry_stats(self, registry, sample_registration):
        """Test registry statistics."""
        asyncio.run(registry.register_agent(sample_registration))
        
        stats = registry.get_registry_stats()
        assert stats["total_agents"] == 1
        assert stats["active_agents"] == 1
        assert "test" in stats["agent_types"]