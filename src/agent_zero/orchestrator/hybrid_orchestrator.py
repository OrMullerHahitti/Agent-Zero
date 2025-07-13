"""Hybrid orchestrator that handles both SLM and algorithmic agents."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

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
from ..agents.input_classifier_agent import InputClassifierAgent
from ..agents.output_formatter_agent import OutputFormatterAgent, SolutionPipeline


class HybridOrchestrator(BaseAgent):
    """
    Enhanced orchestrator that manages both algorithmic agents and SLM agents,
    with intelligent routing and response formatting.
    """
    
    def __init__(
        self,
        message_bus: MessageBus,
        agent_registry: AgentRegistry,
        input_classifier: Optional[InputClassifierAgent] = None,
        output_formatter: Optional[OutputFormatterAgent] = None,
        agent_id: str = "hybrid_orchestrator"
    ):
        super().__init__(agent_id, "hybrid_orchestrator")
        
        self.message_bus = message_bus
        self.registry = agent_registry
        self.router = MessageRouter(message_bus, agent_registry)
        
        # AI-powered components
        self.input_classifier = input_classifier
        self.output_formatter = output_formatter
        
        # Create solution pipeline if components available
        if input_classifier and output_formatter:
            self.solution_pipeline = SolutionPipeline(
                input_classifier, output_formatter, self
            )
        else:
            self.solution_pipeline = None
        
        # Agent type classification
        self.agent_types = {
            "algorithmic": {
                "description": "Specialized algorithmic solvers",
                "examples": ["belief_propagation", "graph_algorithms", "optimization"],
                "input_format": "structured_parameters",
                "output_format": "numerical_results"
            },
            "slm": {
                "description": "Small Language Model agents", 
                "examples": ["reasoning", "coding", "text_analysis"],
                "input_format": "natural_language",
                "output_format": "natural_language"
            },
            "hybrid": {
                "description": "Agents that combine both approaches",
                "examples": ["multimodal", "complex_reasoning"],
                "input_format": "flexible",
                "output_format": "flexible"
            }
        }
        
        # Enhanced routing strategies
        self.routing_strategies = {
            "direct": self._route_direct,
            "intelligent": self._route_intelligent,
            "pipeline": self._route_pipeline,
            "ensemble": self._route_ensemble
        }
        
        # Performance tracking
        self.routing_decisions = []
        self.agent_performance = {}
        
    async def start(self) -> None:
        """Start the hybrid orchestrator."""
        await super().start()
        
        # Start AI components
        if self.input_classifier:
            await self.input_classifier.start()
        if self.output_formatter:
            await self.output_formatter.start()
        
        # Subscribe to message bus
        await self.message_bus.subscribe(
            self.agent_id,
            self._handle_message_callback
        )
        
        self.logger.info("Hybrid orchestrator started with AI-powered routing")
    
    async def stop(self) -> None:
        """Stop the hybrid orchestrator."""
        if self.input_classifier:
            await self.input_classifier.stop()
        if self.output_formatter:
            await self.output_formatter.stop()
            
        await self.message_bus.unsubscribe(self.agent_id)
        await super().stop()
    
    async def get_capabilities(self) -> AgentCapability:
        """Return hybrid orchestrator capabilities."""
        performance_metrics = self.get_performance_metrics()
        
        # Add hybrid-specific metrics
        performance_metrics.update({
            "ai_routing_available": self.input_classifier is not None,
            "output_formatting_available": self.output_formatter is not None,
            "pipeline_available": self.solution_pipeline is not None,
            "supported_agent_types": list(self.agent_types.keys()),
            "routing_strategies": list(self.routing_strategies.keys())
        })
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=[
                "hybrid_orchestration", 
                "intelligent_routing", 
                "pipeline_processing",
                "natural_language_problems"
            ],
            description=(
                "Hybrid orchestrator that intelligently routes problems between "
                "algorithmic agents and SLM agents, with AI-powered classification "
                "and natural language response formatting."
            ),
            performance_metrics=performance_metrics,
            load_factor=0.0,
            availability=True
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """Hybrid orchestrator can handle any problem through routing."""
        
        # Check if we have the full pipeline available
        if self.solution_pipeline:
            return 1.0  # Can handle any natural language problem
        
        # Check if we have any capable agents
        if self.input_classifier:
            # Use AI classification to find capable agents
            classification = await self._quick_classify(problem_request)
            if classification:
                capable_agents = self._find_agents_by_type(classification["category"])
                return 0.9 if capable_agents else 0.3
        
        # Fallback to traditional routing
        capable_agents = self.registry.find_agents_for_problem(
            problem_request.problem_type or "general"
        )
        return 0.8 if capable_agents else 0.1
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Solve problem using hybrid routing strategy."""
        try:
            start_time = datetime.now()
            
            # Determine routing strategy
            strategy = await self._select_routing_strategy(problem_request)
            self.logger.info(f"Selected routing strategy: {strategy}")
            
            # Execute routing strategy
            router_func = self.routing_strategies[strategy]
            result = await router_func(problem_request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance tracking
            self._track_routing_decision(strategy, problem_request, result, execution_time)
            
            # Add orchestration metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update({
                "routing_strategy": strategy,
                "orchestrator_type": "hybrid",
                "total_orchestration_time": execution_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid orchestration error: {e}")
            return ProblemResult(
                success=False,
                error_message=f"Hybrid orchestration failed: {str(e)}"
            )
    
    async def _select_routing_strategy(self, problem_request: ProblemRequest) -> str:
        """Select the best routing strategy for this problem."""
        
        # Pipeline strategy: use for natural language problems with full pipeline
        if (self.solution_pipeline and 
            not problem_request.problem_type and
            len(problem_request.problem_description) > 20):
            return "pipeline"
        
        # Intelligent strategy: use when AI classifier is available
        if self.input_classifier and not problem_request.problem_type:
            return "intelligent"
        
        # Ensemble strategy: for complex or multi-part problems
        if (problem_request.problem_description and
            any(word in problem_request.problem_description.lower() 
                for word in ["and", "also", "additionally", "furthermore"])):
            return "ensemble"
        
        # Direct strategy: for pre-classified problems
        return "direct"
    
    async def _route_direct(self, problem_request: ProblemRequest) -> ProblemResult:
        """Direct routing to best available agent."""
        problem_type = problem_request.problem_type or "general"
        capable_agents = self.registry.find_agents_for_problem(problem_type)
        
        if not capable_agents:
            return ProblemResult(
                success=False,
                error_message=f"No agents available for {problem_type} problems"
            )
        
        # Try agents in order of preference
        for agent_reg in capable_agents:
            try:
                result = await self._request_solution_from_agent(
                    agent_reg.agent_id, problem_request
                )
                if result and result.success:
                    return result
            except Exception as e:
                self.logger.warning(f"Agent {agent_reg.agent_id} failed: {e}")
                continue
        
        return ProblemResult(
            success=False,
            error_message="All capable agents failed"
        )
    
    async def _route_intelligent(self, problem_request: ProblemRequest) -> ProblemResult:
        """AI-powered intelligent routing."""
        if not self.input_classifier:
            return await self._route_direct(problem_request)
        
        # Use AI classifier to understand the problem
        classification_result = await self.input_classifier.solve_problem(
            ProblemRequest(
                problem_description=problem_request.problem_description,
                problem_type="input_classification"
            )
        )
        
        if not classification_result.success:
            return await self._route_direct(problem_request)
        
        # Route to appropriate agent based on classification
        classification_data = classification_result.solution
        formatted_request = classification_data["formatted_request"]
        
        result = await self._route_direct(formatted_request)
        
        # Add classification metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update({
            "ai_classification": classification_data["classification"],
            "classification_confidence": classification_result.confidence
        })
        
        return result
    
    async def _route_pipeline(self, problem_request: ProblemRequest) -> ProblemResult:
        """Full AI pipeline: classification -> solving -> formatting."""
        if not self.solution_pipeline:
            return await self._route_intelligent(problem_request)
        
        # Use complete solution pipeline
        pipeline_result = await self.solution_pipeline.solve_natural_language_problem(
            problem_request.problem_description
        )
        
        if pipeline_result["success"]:
            return ProblemResult(
                success=True,
                solution=pipeline_result.get("raw_solution", {}),
                explanation=pipeline_result["explanation"],
                confidence=pipeline_result["metadata"]["confidence"],
                execution_time=pipeline_result["metadata"]["total_execution_time"],
                metadata={
                    "pipeline_result": pipeline_result,
                    "methodology": pipeline_result["methodology"],
                    "insights": pipeline_result.get("insights", []),
                    "solution_quality": pipeline_result.get("solution_quality", {})
                }
            )
        else:
            return ProblemResult(
                success=False,
                error_message=pipeline_result["error"]
            )
    
    async def _route_ensemble(self, problem_request: ProblemRequest) -> ProblemResult:
        """Route to multiple agents and combine results."""
        # For complex problems, try multiple approaches
        
        # First, try intelligent routing
        primary_result = await self._route_intelligent(problem_request)
        
        # If successful, also try to get alternative solutions
        if primary_result.success:
            alternatives = []
            
            # Try other capable agents
            problem_type = problem_request.problem_type or "general"
            capable_agents = self.registry.find_agents_for_problem(problem_type)
            
            for agent_reg in capable_agents[:2]:  # Try up to 2 alternatives
                try:
                    alt_result = await self._request_solution_from_agent(
                        agent_reg.agent_id, problem_request
                    )
                    if alt_result and alt_result.success:
                        alternatives.append({
                            "agent_id": agent_reg.agent_id,
                            "solution": alt_result.solution,
                            "confidence": alt_result.confidence
                        })
                except Exception:
                    continue
            
            # Add alternatives to primary result
            if alternatives:
                primary_result.alternative_solutions = alternatives
                if primary_result.metadata is None:
                    primary_result.metadata = {}
                primary_result.metadata["ensemble_alternatives"] = len(alternatives)
        
        return primary_result
    
    async def _quick_classify(self, problem_request: ProblemRequest) -> Optional[Dict[str, Any]]:
        """Quick problem classification without full processing."""
        if not self.input_classifier:
            return None
        
        try:
            # Use lightweight classification
            result = await self.input_classifier.solve_problem(
                ProblemRequest(
                    problem_description=problem_request.problem_description[:500],  # Truncate for speed
                    problem_type="quick_classification"
                )
            )
            return result.solution.get("classification") if result.success else None
        except Exception:
            return None
    
    def _find_agents_by_type(self, agent_category: str) -> List:
        """Find agents that can handle a specific category."""
        all_agents = self.registry.get_all_agents()
        matching_agents = []
        
        for agent_reg in all_agents:
            capability = self.registry.get_agent_capabilities(agent_reg.agent_id)
            if capability and agent_category in capability.problem_types:
                matching_agents.append(agent_reg)
        
        return matching_agents
    
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
            timeout=60.0
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
    
    def _track_routing_decision(
        self,
        strategy: str, 
        problem_request: ProblemRequest,
        result: ProblemResult,
        execution_time: float
    ) -> None:
        """Track routing decisions for performance analysis."""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "problem_type": problem_request.problem_type,
            "success": result.success,
            "execution_time": execution_time,
            "confidence": result.confidence
        }
        
        self.routing_decisions.append(decision_record)
        
        # Keep only last 1000 decisions
        if len(self.routing_decisions) > 1000:
            self.routing_decisions.pop(0)
    
    async def _handle_message_callback(self, message: Message) -> None:
        """Handle incoming messages."""
        try:
            response = await self.handle_message(message)
            if response:
                await self.message_bus.send_message(response)
        except Exception as e:
            self.logger.error(f"Error in message callback: {e}")
    
    async def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions."""
        if not self.routing_decisions:
            return {"message": "No routing decisions recorded yet"}
        
        total_decisions = len(self.routing_decisions)
        successful_decisions = sum(1 for d in self.routing_decisions if d["success"])
        
        strategy_stats = {}
        for decision in self.routing_decisions:
            strategy = decision["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"count": 0, "success_rate": 0, "avg_time": 0}
            
            strategy_stats[strategy]["count"] += 1
            if decision["success"]:
                strategy_stats[strategy]["success_rate"] += 1
        
        # Calculate success rates and average times
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats["success_rate"] = stats["success_rate"] / stats["count"] * 100
            strategy_decisions = [d for d in self.routing_decisions if d["strategy"] == strategy]
            stats["avg_time"] = sum(d["execution_time"] for d in strategy_decisions) / len(strategy_decisions)
        
        return {
            "total_decisions": total_decisions,
            "overall_success_rate": successful_decisions / total_decisions * 100,
            "strategy_performance": strategy_stats,
            "recent_decisions": self.routing_decisions[-10:]  # Last 10 decisions
        }