"""Main orchestrator that coordinates multiple specialized agents."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from ..core.agent_base import BaseAgent
from ..core.message import (
    Message, 
    MessageType, 
    ProblemRequest, 
    ProblemResult, 
    AgentCapability
)
from ..core.registry import AgentRegistry
from ..protocols.communication import MessageBus, MessageRouter


class MainOrchestrator(BaseAgent):
    """Main orchestrator that routes problems to specialized agents."""
    
    def __init__(
        self, 
        message_bus: MessageBus,
        agent_registry: AgentRegistry,
        agent_id: str = "main_orchestrator"
    ):
        super().__init__(agent_id, "orchestrator")
        
        self.message_bus = message_bus
        self.registry = agent_registry
        self.router = MessageRouter(message_bus, agent_registry)
        
        # Problem classification patterns
        self.problem_classifiers = {
            "dcop": [
                r"constraint.*optimization",
                r"dcop",
                r"distributed.*constraint",
                r"factor.*graph",
                r"belief.*propagation",
                r"min.*sum|max.*sum"
            ],
            "graph": [
                r"graph.*algorithm",
                r"shortest.*path",
                r"minimum.*spanning.*tree",
                r"network.*flow",
                r"graph.*coloring"
            ],
            "search": [
                r"search.*algorithm",
                r"pathfinding",
                r"a\*|astar",
                r"beam.*search",
                r"local.*search"
            ],
            "optimization": [
                r"linear.*programming",
                r"genetic.*algorithm", 
                r"simulated.*annealing",
                r"optimization.*problem"
            ],
            "ml": [
                r"machine.*learning",
                r"classification",
                r"regression",
                r"clustering",
                r"neural.*network"
            ]
        }
        
        # Track problem solving statistics
        self.problems_routed = 0
        self.successful_solutions = 0
        self.failed_solutions = 0
    
    async def start(self) -> None:
        """Start the orchestrator."""
        await super().start()
        
        # Subscribe to message bus
        await self.message_bus.subscribe(
            self.agent_id, 
            self._handle_message_callback
        )
        
        self.logger.info("Main orchestrator started and subscribed to message bus")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        await self.message_bus.unsubscribe(self.agent_id)
        await super().stop()
    
    async def get_capabilities(self) -> AgentCapability:
        """Return orchestrator capabilities."""
        performance_metrics = self.get_performance_metrics()
        performance_metrics.update({
            "problems_routed": self.problems_routed,
            "successful_solutions": self.successful_solutions,
            "failed_solutions": self.failed_solutions,
            "success_rate": (
                self.successful_solutions / max(self.problems_routed, 1)
            ) * 100
        })
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=["orchestration", "problem_routing", "agent_coordination"],
            description=(
                "Main orchestrator that analyzes problems, routes them to appropriate "
                "specialized agents, and coordinates multi-agent problem solving."
            ),
            performance_metrics=performance_metrics,
            load_factor=0.0,
            availability=True
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """Orchestrator can handle any problem by routing it."""
        # Check if we have any agents that can solve this problem
        problem_type = await self._classify_problem(problem_request)
        capable_agents = self.registry.find_agents_for_problem(problem_type)
        
        if capable_agents:
            return 1.0  # We can route it
        else:
            return 0.1  # We can try, but no specialized agents available
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Orchestrate problem solving by routing to appropriate agents."""
        try:
            self.problems_routed += 1
            start_time = datetime.now()
            
            # Classify the problem
            problem_type = await self._classify_problem(problem_request)
            self.logger.info(f"Classified problem as: {problem_type}")
            
            # Find capable agents
            capable_agents = self.registry.find_agents_for_problem(problem_type)
            
            if not capable_agents:
                self.failed_solutions += 1
                return ProblemResult(
                    success=False,
                    error_message=f"No agents available to solve {problem_type} problems"
                )
            
            # Try agents in order of preference (lowest load first)
            last_error = None
            
            for agent_reg in capable_agents:
                try:
                    result = await self._request_solution_from_agent(
                        agent_reg.agent_id,
                        problem_request
                    )
                    
                    if result and result.success:
                        self.successful_solutions += 1
                        execution_time = (datetime.now() - start_time).total_seconds()
                        
                        # Add orchestration metadata
                        result.metadata = result.metadata or {}
                        result.metadata.update({
                            "routed_to_agent": agent_reg.agent_id,
                            "agent_type": agent_reg.agent_type,
                            "total_orchestration_time": execution_time,
                            "problem_classification": problem_type
                        })
                        
                        return result
                    else:
                        last_error = result.error_message if result else "No response"
                        
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Agent {agent_reg.agent_id} failed: {e}")
                    continue
            
            # All agents failed
            self.failed_solutions += 1
            return ProblemResult(
                success=False,
                error_message=f"All capable agents failed. Last error: {last_error}"
            )
            
        except Exception as e:
            self.failed_solutions += 1
            self.logger.error(f"Orchestration error: {e}")
            return ProblemResult(
                success=False,
                error_message=f"Orchestration failed: {str(e)}"
            )
    
    async def _classify_problem(self, problem_request: ProblemRequest) -> str:
        """Classify the problem type based on description."""
        
        # Check explicit problem type first
        if problem_request.problem_type:
            return problem_request.problem_type.lower()
        
        # Pattern matching on description
        description = problem_request.problem_description.lower()
        
        scores = {}
        for problem_type, patterns in self.problem_classifiers.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, description):
                    score += 1
            scores[problem_type] = score
        
        # Return the type with highest score, or "general" if no matches
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return "general"
    
    async def _request_solution_from_agent(
        self, 
        agent_id: str, 
        problem_request: ProblemRequest
    ) -> Optional[ProblemResult]:
        """Request solution from a specific agent."""
        
        response = await self.router.send_request(
            sender_id=self.agent_id,
            recipient_id=agent_id,
            message_type=MessageType.PROBLEM_SUBMIT,
            payload=problem_request.dict(),
            timeout=60.0  # 1 minute timeout
        )
        
        if response and response.type == MessageType.PROBLEM_RESULT:
            return ProblemResult(**response.payload)
        elif response and response.type == MessageType.ERROR:
            return ProblemResult(
                success=False,
                error_message=response.payload.get("error", "Unknown error")
            )
        else:
            return None
    
    async def _handle_message_callback(self, message: Message) -> None:
        """Callback for handling messages from message bus."""
        try:
            response = await self.handle_message(message)
            if response:
                await self.message_bus.send_message(response)
        except Exception as e:
            self.logger.error(f"Error in message callback: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        registry_stats = self.registry.get_registry_stats()
        orchestrator_stats = self.get_performance_metrics()
        
        return {
            "orchestrator": {
                "agent_id": self.agent_id,
                "status": "running" if self.is_running else "stopped",
                "uptime_seconds": self._get_uptime(),
                "problems_routed": self.problems_routed,
                "successful_solutions": self.successful_solutions,
                "failed_solutions": self.failed_solutions,
                "success_rate_percent": (
                    self.successful_solutions / max(self.problems_routed, 1)
                ) * 100
            },
            "registry": registry_stats,
            "system": {
                "timestamp": datetime.now().isoformat(),
                "total_problem_types_supported": len(registry_stats.get("supported_problem_types", [])),
                "system_health": "healthy" if registry_stats.get("active_agents", 0) > 0 else "degraded"
            }
        }
    
    async def discover_agents(self) -> List[AgentCapability]:
        """Discover all available agents and their capabilities."""
        agents = self.registry.get_all_agents()
        capabilities = []
        
        for agent_reg in agents:
            capability = self.registry.get_agent_capabilities(agent_reg.agent_id)
            if capability:
                capabilities.append(capability)
        
        return capabilities