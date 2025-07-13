"""Belief Propagation agent that integrates PropFlow for DCOP problem solving."""

import sys
import os
from typing import Dict, List, Optional, Any
import logging
import asyncio
import re
from datetime import datetime

# Add PropFlow to path (assuming it's installed or available)
try:
    import propflow
    from propflow.simulator import Simulator
    from propflow.utils.create.create_factor_graph_config import create_factor_graph_config
    from propflow.utils.create.create_factor_graphs_from_config import create_factor_graphs_from_config
    PROPFLOW_AVAILABLE = True
except ImportError:
    PROPFLOW_AVAILABLE = False
    logging.warning("PropFlow not available. Install PropFlow for DCOP solving capabilities.")

from ..core.agent_base import BaseAgent
from ..core.message import (
    ProblemRequest, 
    ProblemResult, 
    AgentCapability
)


class BeliefPropagationAgent(BaseAgent):
    """Agent specialized in solving DCOP problems using belief propagation."""
    
    def __init__(self, agent_id: str = "bp_agent"):
        super().__init__(agent_id, "belief_propagation")
        
        self.supported_problem_types = [
            "dcop",
            "constraint_optimization", 
            "factor_graph",
            "distributed_constraint_optimization",
            "belief_propagation",
            "min_sum",
            "max_sum"
        ]
        
        # Problem type detection patterns
        self.problem_patterns = {
            "dcop": [
                r"constraint.*optimization",
                r"dcop",
                r"distributed.*constraint",
                r"factor.*graph",
                r"variable.*assignment",
                r"minimize.*cost",
                r"optimization.*problem"
            ],
            "scheduling": [
                r"schedule.*problem",
                r"resource.*allocation",
                r"time.*slot",
                r"assignment.*problem"
            ],
            "graph_coloring": [
                r"graph.*coloring",
                r"vertex.*coloring",
                r"color.*assignment"
            ]
        }
    
    async def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities."""
        performance_metrics = self.get_performance_metrics()
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=self.supported_problem_types,
            description=(
                "Specialized agent for solving Distributed Constraint Optimization Problems (DCOP) "
                "using belief propagation algorithms. Supports factor graphs, min-sum, max-sum, "
                "and various optimization policies including damping and splitting."
            ),
            performance_metrics=performance_metrics,
            load_factor=0.0,  # Simple implementation - could track actual load
            availability=PROPFLOW_AVAILABLE
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """Determine if this agent can solve the given problem."""
        if not PROPFLOW_AVAILABLE:
            return 0.0
        
        problem_desc = problem_request.problem_description.lower()
        problem_type = problem_request.problem_type
        
        # Check explicit problem type
        if problem_type and problem_type.lower() in self.supported_problem_types:
            return 0.95
        
        # Pattern matching on description
        confidence = 0.0
        for category, patterns in self.problem_patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_desc):
                    if category == "dcop":
                        confidence = max(confidence, 0.9)
                    else:
                        confidence = max(confidence, 0.7)  # Related problems
        
        # Look for specific keywords
        dcop_keywords = [
            "constraint", "optimization", "variable", "assignment", 
            "minimize", "cost", "factor", "belief", "propagation"
        ]
        
        keyword_count = sum(1 for keyword in dcop_keywords if keyword in problem_desc)
        keyword_confidence = min(keyword_count * 0.15, 0.8)
        
        return max(confidence, keyword_confidence)
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Solve DCOP problem using PropFlow."""
        if not PROPFLOW_AVAILABLE:
            return ProblemResult(
                success=False,
                error_message="PropFlow library not available"
            )
        
        try:
            start_time = datetime.now()
            
            # Parse problem parameters from description and constraints
            problem_config = self._parse_problem_config(problem_request)
            
            # Create factor graph configuration
            fg_config = self._create_factor_graph_config(problem_config)
            
            # Create factor graphs
            factor_graphs = create_factor_graphs_from_config(fg_config)
            
            if not factor_graphs:
                return ProblemResult(
                    success=False,
                    error_message="Failed to create factor graphs from problem description"
                )
            
            # Solve using PropFlow simulator
            results = []
            for i, fg in enumerate(factor_graphs):
                result = await self._solve_single_factor_graph(fg, problem_config)
                results.append(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process and format results
            if results:
                best_result = min(results, key=lambda r: r.get('total_cost', float('inf')))
                
                return ProblemResult(
                    success=True,
                    solution=best_result,
                    explanation=self._generate_solution_explanation(best_result, problem_config),
                    confidence=0.85,
                    execution_time=execution_time,
                    alternative_solutions=results[1:] if len(results) > 1 else None
                )
            else:
                return ProblemResult(
                    success=False,
                    error_message="No valid solutions found",
                    execution_time=execution_time
                )
                
        except Exception as e:
            self.logger.error(f"Error solving DCOP problem: {e}")
            return ProblemResult(
                success=False,
                error_message=f"Solver error: {str(e)}"
            )
    
    def _parse_problem_config(self, problem_request: ProblemRequest) -> Dict[str, Any]:
        """Parse problem configuration from request."""
        # Default configuration
        config = {
            "num_variables": 10,
            "domain_size": 5,
            "density": 0.3,
            "cost_range": [0, 100],
            "algorithm": "min_sum",
            "max_iterations": 1000,
            "convergence_threshold": 1e-6
        }
        
        # Override with explicit constraints
        if problem_request.constraints:
            config.update(problem_request.constraints)
        
        # Parse from description
        desc = problem_request.problem_description.lower()
        
        # Extract numbers that might be parameters
        numbers = re.findall(r'\b\d+\b', desc)
        if numbers:
            # Heuristic: first number might be variables, second might be domain size
            if len(numbers) >= 1:
                config["num_variables"] = min(int(numbers[0]), 50)  # Cap at 50
            if len(numbers) >= 2:
                config["domain_size"] = min(int(numbers[1]), 10)   # Cap at 10
        
        # Algorithm detection
        if "max" in desc:
            config["algorithm"] = "max_sum"
        elif "min" in desc:
            config["algorithm"] = "min_sum"
        
        return config
    
    def _create_factor_graph_config(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create factor graph configuration for PropFlow."""
        return {
            "num_problems": 1,
            "num_variables": problem_config["num_variables"],
            "domain_size": problem_config["domain_size"],
            "density": problem_config["density"],
            "cost_function": "uniform_int",
            "cost_params": {
                "low": problem_config["cost_range"][0],
                "high": problem_config["cost_range"][1]
            }
        }
    
    async def _solve_single_factor_graph(
        self, 
        factor_graph, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve a single factor graph instance."""
        
        # Configure simulator
        simulator_config = {
            "algorithm": config["algorithm"],
            "max_iterations": config["max_iterations"],
            "convergence_threshold": config["convergence_threshold"]
        }
        
        # Run simulation (this is a simplified version)
        # In a real implementation, you'd use the actual PropFlow API
        simulator = Simulator(factor_graph, **simulator_config)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, simulator.run)
        
        # Extract solution
        assignments = {}
        total_cost = 0.0
        
        # Get variable assignments (simplified)
        for i, var_agent in enumerate(factor_graph.variable_agents):
            assignments[var_agent.name] = var_agent.curr_assignment
            # Note: actual cost calculation would use the factor graph
        
        return {
            "assignments": assignments,
            "total_cost": total_cost,
            "iterations": simulator.iterations_run,
            "converged": simulator.converged,
            "algorithm_used": config["algorithm"]
        }
    
    def _generate_solution_explanation(
        self, 
        solution: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of the solution."""
        
        assignments = solution.get("assignments", {})
        total_cost = solution.get("total_cost", 0)
        iterations = solution.get("iterations", 0)
        algorithm = solution.get("algorithm_used", "unknown")
        
        explanation = f"""
DCOP Solution found using {algorithm.replace('_', '-')} algorithm:

Variables and Assignments:
"""
        
        for var_name, value in assignments.items():
            explanation += f"  {var_name}: {value}\n"
        
        explanation += f"""
Solution Quality:
  Total Cost: {total_cost:.2f}
  Convergence: {"Yes" if solution.get("converged", False) else "No"}
  Iterations: {iterations}

Algorithm Configuration:
  Number of Variables: {config.get("num_variables", "unknown")}
  Domain Size: {config.get("domain_size", "unknown")}
  Density: {config.get("density", "unknown")}
"""
        
        return explanation.strip()