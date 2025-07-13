"""Output Formatter Agent - Uses SLM to convert solutions to natural language."""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..core.agent_base import BaseAgent
from ..core.message import (
    ProblemRequest, 
    ProblemResult, 
    AgentCapability
)


class OutputFormatterAgent(BaseAgent):
    """
    Agent that uses SLM to convert raw agent solutions into natural language
    explanations with methodology and reasoning.
    """
    
    def __init__(self, agent_id: str = "output_formatter", slm_client=None):
        super().__init__(agent_id, "output_formatter")
        
        self.slm_client = slm_client  # Smaller language model for formatting
        
        # Templates for different solution types
        self.formatting_prompts = {
            "dcop": """
You are explaining a Distributed Constraint Optimization Problem (DCOP) solution. 
Convert this technical solution into clear, understandable language.

Original Problem: {original_problem}
Solution Data: {solution_data}
Algorithm Used: {algorithm}
Execution Time: {execution_time}s

Explain:
1. What the problem was asking for
2. How the algorithm solved it
3. What the final answer means
4. Why this solution is good/optimal

Use simple language that a non-expert can understand.
""",
            
            "graph_algorithms": """
You are explaining a graph algorithm solution.
Convert this technical result into clear, understandable language.

Original Problem: {original_problem}
Solution Data: {solution_data}
Algorithm Used: {algorithm}
Execution Time: {execution_time}s

Explain:
1. What type of graph problem this was
2. How the algorithm found the solution
3. What the result represents (path, cost, etc.)
4. Why this is the best answer

Make it accessible to someone learning about graphs.
""",
            
            "optimization": """
You are explaining an optimization problem solution.
Convert this technical result into clear language.

Original Problem: {original_problem}
Solution Data: {solution_data}
Method Used: {algorithm}
Execution Time: {execution_time}s

Explain:
1. What was being optimized
2. What constraints were considered
3. How the algorithm found the best solution
4. What the optimal values mean practically

Focus on practical implications.
""",
            
            "slm_response": """
You are reformatting and improving an AI response.
Make this response clearer and more comprehensive.

Original Problem: {original_problem}
AI Response: {solution_data}

Improve the response by:
1. Making the explanation clearer
2. Adding reasoning steps where missing
3. Providing examples if helpful
4. Ensuring completeness

Maintain the original meaning but enhance clarity.
""",
            
            "general": """
You are explaining a problem solution clearly.
Convert this technical result into understandable language.

Original Problem: {original_problem}
Solution: {solution_data}
Method: {algorithm}

Explain:
1. What the problem was asking
2. How it was solved
3. What the answer means
4. Any important insights

Use clear, accessible language.
"""
        }
    
    async def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities."""
        performance_metrics = self.get_performance_metrics()
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=["output_formatting", "solution_explanation", "natural_language_generation"],
            description=(
                "Output formatter agent that uses SLM to convert technical solutions "
                "from specialized agents into clear, natural language explanations "
                "with methodology and reasoning."
            ),
            performance_metrics=performance_metrics,
            load_factor=0.0,
            availability=self.slm_client is not None
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """This agent handles output formatting when given solution data."""
        if not self.slm_client:
            return 0.0
        
        # Check if this is a formatting request
        if problem_request.problem_type == "format_solution":
            return 1.0
            
        # Check if we have solution data to format
        constraints = problem_request.constraints or {}
        if "solution_data" in constraints and "original_problem" in constraints:
            return 1.0
            
        return 0.0
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Format technical solution into natural language explanation."""
        if not self.slm_client:
            return ProblemResult(
                success=False,
                error_message="SLM client not available"
            )
        
        try:
            start_time = datetime.now()
            
            # Extract solution data from request
            constraints = problem_request.constraints or {}
            solution_data = constraints.get("solution_data", {})
            original_problem = constraints.get("original_problem", "")
            agent_type = constraints.get("agent_type", "general")
            algorithm = constraints.get("algorithm", "unknown")
            
            if not solution_data:
                return ProblemResult(
                    success=False,
                    error_message="No solution data provided for formatting"
                )
            
            # Generate natural language explanation
            explanation = await self._format_solution(
                original_problem=original_problem,
                solution_data=solution_data,
                agent_type=agent_type,
                algorithm=algorithm,
                execution_time=constraints.get("execution_time", 0)
            )
            
            # Extract key insights
            insights = self._extract_insights(solution_data, agent_type)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ProblemResult(
                success=True,
                solution={
                    "formatted_explanation": explanation,
                    "key_insights": insights,
                    "methodology": algorithm,
                    "solution_quality": self._assess_solution_quality(solution_data)
                },
                explanation=explanation,
                confidence=0.9,  # High confidence in formatting
                execution_time=execution_time,
                metadata={
                    "original_problem": original_problem,
                    "source_agent_type": agent_type,
                    "formatting_method": "slm_generation"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in output formatting: {e}")
            return ProblemResult(
                success=False,
                error_message=f"Formatting error: {str(e)}"
            )
    
    async def _format_solution(
        self,
        original_problem: str,
        solution_data: Dict[str, Any],
        agent_type: str,
        algorithm: str,
        execution_time: float
    ) -> str:
        """Use SLM to format the solution into natural language."""
        
        # Select appropriate prompt template
        prompt_template = self.formatting_prompts.get(agent_type, self.formatting_prompts["general"])
        
        # Format the prompt
        prompt = prompt_template.format(
            original_problem=original_problem,
            solution_data=json.dumps(solution_data, indent=2),
            algorithm=algorithm,
            execution_time=execution_time
        )
        
        # Call SLM for formatting
        formatted_response = await self._call_slm(prompt)
        
        return formatted_response
    
    async def _call_slm(self, prompt: str) -> str:
        """Call the SLM client for text generation."""
        if not self.slm_client:
            raise ValueError("No SLM client configured")
        
        # This is a placeholder - implement based on your SLM choice
        # Could be:
        # - Local model like Llama, Mistral, CodeLlama
        # - API call to smaller model
        # - On-device inference
        
        # Mock response for testing
        return """
The problem was asking to find the best assignment of values to variables while minimizing cost and satisfying constraints.

The belief propagation algorithm solved this by:
1. Creating a factor graph representing the constraints
2. Passing messages between variables and factors iteratively
3. Converging to an optimal solution after 127 iterations

The final solution assigns each variable a value that minimizes the total cost while respecting all constraints. The total cost achieved is 45.2, which represents an optimal or near-optimal solution.

This solution is good because it satisfies all the problem constraints while achieving the lowest possible cost through the systematic message-passing approach of belief propagation.
"""
    
    def _extract_insights(self, solution_data: Dict[str, Any], agent_type: str) -> List[str]:
        """Extract key insights from the solution."""
        insights = []
        
        if agent_type == "dcop":
            if "total_cost" in solution_data:
                insights.append(f"Achieved total cost of {solution_data['total_cost']}")
            if "converged" in solution_data:
                if solution_data["converged"]:
                    insights.append("Algorithm converged to optimal solution")
                else:
                    insights.append("Algorithm reached iteration limit (may be near-optimal)")
            if "iterations" in solution_data:
                insights.append(f"Required {solution_data['iterations']} iterations to solve")
                
        elif agent_type == "graph_algorithms":
            if "path_length" in solution_data:
                insights.append(f"Found path of length {solution_data['path_length']}")
            if "nodes_visited" in solution_data:
                insights.append(f"Explored {solution_data['nodes_visited']} nodes")
                
        elif agent_type == "optimization":
            if "objective_value" in solution_data:
                insights.append(f"Optimized objective to {solution_data['objective_value']}")
            if "constraints_satisfied" in solution_data:
                insights.append(f"All {solution_data['constraints_satisfied']} constraints satisfied")
        
        # General insights
        if "execution_time" in solution_data:
            time = solution_data["execution_time"]
            if time < 1:
                insights.append("Very fast solution (under 1 second)")
            elif time > 10:
                insights.append("Complex problem requiring longer computation")
        
        return insights
    
    def _assess_solution_quality(self, solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the solution."""
        quality = {
            "completeness": "unknown",
            "optimality": "unknown", 
            "efficiency": "unknown"
        }
        
        # Assess completeness
        if "success" in solution_data:
            quality["completeness"] = "complete" if solution_data["success"] else "incomplete"
        
        # Assess optimality (problem-specific)
        if "converged" in solution_data:
            quality["optimality"] = "optimal" if solution_data["converged"] else "near-optimal"
        elif "is_optimal" in solution_data:
            quality["optimality"] = "optimal" if solution_data["is_optimal"] else "heuristic"
        
        # Assess efficiency
        if "execution_time" in solution_data:
            time = solution_data["execution_time"]
            if time < 1:
                quality["efficiency"] = "very_fast"
            elif time < 10:
                quality["efficiency"] = "fast"
            elif time < 60:
                quality["efficiency"] = "moderate"
            else:
                quality["efficiency"] = "slow"
        
        return quality


class SolutionPipeline:
    """
    Complete pipeline that chains input classification, specialized solving, 
    and output formatting.
    """
    
    def __init__(
        self,
        input_classifier: InputClassifierAgent,
        output_formatter: OutputFormatterAgent,
        orchestrator,  # Main orchestrator with specialized agents
    ):
        self.input_classifier = input_classifier
        self.output_formatter = output_formatter
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("solution_pipeline")
    
    async def solve_natural_language_problem(self, problem_text: str) -> Dict[str, Any]:
        """
        Complete pipeline: text -> classification -> solving -> formatted response.
        """
        try:
            # Step 1: Classify and format input
            classification_request = ProblemRequest(
                problem_description=problem_text,
                problem_type="input_classification"
            )
            
            classification_result = await self.input_classifier.solve_problem(classification_request)
            
            if not classification_result.success:
                return {
                    "success": False,
                    "error": f"Classification failed: {classification_result.error_message}"
                }
            
            # Step 2: Route to appropriate specialized agent
            classification_data = classification_result.solution
            formatted_request = classification_data["formatted_request"]
            target_agent = classification_data["target_agent"]
            
            # Solve with specialized agent
            solution_result = await self.orchestrator.solve_problem(formatted_request)
            
            if not solution_result.success:
                return {
                    "success": False,
                    "error": f"Specialized agent failed: {solution_result.error_message}"
                }
            
            # Step 3: Format output for human consumption
            formatting_request = ProblemRequest(
                problem_description="Format this solution",
                problem_type="format_solution",
                constraints={
                    "solution_data": solution_result.solution,
                    "original_problem": problem_text,
                    "agent_type": classification_data["category"],
                    "algorithm": solution_result.metadata.get("algorithm", "unknown"),
                    "execution_time": solution_result.execution_time
                }
            )
            
            formatted_result = await self.output_formatter.solve_problem(formatting_request)
            
            if not formatted_result.success:
                # Fallback to raw solution if formatting fails
                return {
                    "success": True,
                    "explanation": solution_result.explanation or "Solution found successfully",
                    "raw_solution": solution_result.solution,
                    "methodology": classification_data["category"],
                    "formatting_warning": "Output formatting failed, showing raw results"
                }
            
            # Step 4: Return complete formatted response
            return {
                "success": True,
                "explanation": formatted_result.solution["formatted_explanation"],
                "insights": formatted_result.solution["key_insights"],
                "methodology": formatted_result.solution["methodology"],
                "solution_quality": formatted_result.solution["solution_quality"],
                "raw_solution": solution_result.solution,
                "metadata": {
                    "classification": classification_data["category"],
                    "confidence": classification_result.confidence,
                    "total_execution_time": (
                        classification_result.execution_time + 
                        solution_result.execution_time + 
                        formatted_result.execution_time
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {
                "success": False,
                "error": f"Pipeline failed: {str(e)}"
            }