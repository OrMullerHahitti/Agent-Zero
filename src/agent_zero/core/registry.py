"""Agent registry for discovering and managing agents."""

from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime, timedelta

from .message import AgentCapability, AgentRegistration


class AgentRegistry:
    """Central registry for agent discovery and management."""
    
    def __init__(self, heartbeat_interval: int = 30, timeout_threshold: int = 90):
        self._agents: Dict[str, AgentRegistration] = {}
        self._capabilities: Dict[str, AgentCapability] = {}
        self._last_heartbeat: Dict[str, datetime] = {}
        
        self.heartbeat_interval = heartbeat_interval  # seconds
        self.timeout_threshold = timeout_threshold    # seconds
        
        self.logger = logging.getLogger("agent_registry")
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the registry with heartbeat monitoring."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self.logger.info("Agent registry started")
    
    async def stop(self) -> None:
        """Stop the registry."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Agent registry stopped")
    
    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent."""
        agent_id = registration.agent_id
        
        if agent_id in self._agents:
            self.logger.warning(f"Agent {agent_id} already registered, updating")
        
        self._agents[agent_id] = registration
        self._capabilities[agent_id] = registration.capabilities
        self._last_heartbeat[agent_id] = datetime.now()
        
        self.logger.info(f"Registered agent: {agent_id} ({registration.agent_type})")
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id not in self._agents:
            return False
        
        del self._agents[agent_id]
        del self._capabilities[agent_id]
        del self._last_heartbeat[agent_id]
        
        self.logger.info(f"Unregistered agent: {agent_id}")
        return True
    
    async def update_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp."""
        if agent_id not in self._agents:
            return False
        
        self._last_heartbeat[agent_id] = datetime.now()
        return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID."""
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentRegistration]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentRegistration]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]
    
    def find_agents_for_problem(self, problem_type: str) -> List[AgentRegistration]:
        """Find agents that can handle a specific problem type."""
        matching_agents = []
        
        for agent_id, capability in self._capabilities.items():
            if (problem_type in capability.problem_types and 
                capability.availability and
                agent_id in self._agents):
                matching_agents.append(self._agents[agent_id])
        
        # Sort by load factor (prefer less loaded agents)
        matching_agents.sort(key=lambda a: a.capabilities.load_factor)
        
        return matching_agents
    
    def get_agent_capabilities(self, agent_id: str) -> Optional[AgentCapability]:
        """Get agent capabilities by ID."""
        return self._capabilities.get(agent_id)
    
    def get_problem_types(self) -> Set[str]:
        """Get all supported problem types across all agents."""
        problem_types = set()
        for capability in self._capabilities.values():
            problem_types.update(capability.problem_types)
        return problem_types
    
    def get_registry_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        now = datetime.now()
        timeout_threshold = timedelta(seconds=self.timeout_threshold)
        
        active_agents = sum(
            1 for last_hb in self._last_heartbeat.values()
            if now - last_hb < timeout_threshold
        )
        
        inactive_agents = len(self._agents) - active_agents
        
        agent_types = {}
        for agent in self._agents.values():
            agent_type = agent.agent_type
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        return {
            "total_agents": len(self._agents),
            "active_agents": active_agents,
            "inactive_agents": inactive_agents,
            "agent_types": agent_types,
            "supported_problem_types": list(self.get_problem_types())
        }
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and remove inactive agents."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._check_inactive_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
    
    async def _check_inactive_agents(self) -> None:
        """Check for and remove inactive agents."""
        now = datetime.now()
        timeout_threshold = timedelta(seconds=self.timeout_threshold)
        
        inactive_agents = []
        for agent_id, last_heartbeat in self._last_heartbeat.items():
            if now - last_heartbeat > timeout_threshold:
                inactive_agents.append(agent_id)
        
        for agent_id in inactive_agents:
            self.logger.warning(f"Removing inactive agent: {agent_id}")
            await self.unregister_agent(agent_id)