"""Input Classifier Agent - Uses LLM/SLM to classify and format problems."""

import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import logging

from ..core.agent_base import BaseAgent
from ..core.message import (
    ProblemRequest, 
    ProblemResult, 
    AgentCapability
)


class InputClassifierAgent(BaseAgent):
    """
    Agent that uses LLM/SLM to classify natural language problems and format them
    for appropriate specialized agents (both SLM and algorithmic).
    """
    
    def __init__(self, agent_id: str = "input_classifier", llm_client=None):
        super().__init__(agent_id, "input_classifier")
        
        self.llm_client = llm_client  # Will be injected - could be OpenAI, Anthropic, local model
        
        # Agent type mappings
        self.agent_type_mapping = {
            # Algorithmic agents
            "dcop": {
                "agent_type": "algorithmic",
                "target_agent": "belief_propagation_agent",
                "input_format": "structured_dcop"
            },
            "graph_algorithms": {
                "agent_type": "algorithmic", 
                "target_agent": "graph_agent",
                "input_format": "graph_structure"
            },
            "optimization": {
                "agent_type": "algorithmic",
                "target_agent": "optimization_agent", 
                "input_format": "optimization_problem"
            },
            
            # SLM agents
            "natural_language_reasoning": {
                "agent_type": "slm",
                "target_agent": "reasoning_slm",
                "input_format": "conversation"
            },
            "code_generation": {
                "agent_type": "slm",
                "target_agent": "coding_slm",
                "input_format": "code_request"
            },
            "text_analysis": {
                "agent_type": "slm",
                "target_agent": "analysis_slm",
                "input_format": "text_analysis_request"
            }
        }
        
        # Classification prompt template
        self.classification_prompt = """
You are a problem classifier. Given a natural language problem description, classify it into one of these categories and extract the key parameters.

Categories:
- dcop: Distributed constraint optimization, factor graphs, belief propagation, variable assignment
- graph_algorithms: Shortest path, MST, network flows, graph coloring, graph traversal
- optimization: Linear programming, genetic algorithms, simulated annealing, general optimization
- natural_language_reasoning: Logic puzzles, reasoning tasks, question answering
- code_generation: Programming tasks, code writing, algorithm implementation
- text_analysis: Text classification, sentiment analysis, summarization, NLP tasks

Problem: {problem_description}

Respond with JSON:
{{
    "category": "category_name",
    "confidence": 0.0-1.0,
    "extracted_parameters": {{
        "key_concepts": ["concept1", "concept2"],
        "numeric_values": [numbers found],
        "constraints": ["constraint1", "constraint2"],
        "objective": "what needs to be optimized/solved"
    }},
    "reasoning": "brief explanation of classification"
}}
"""

    async def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities."""
        performance_metrics = self.get_performance_metrics()
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=["input_classification", "problem_formatting", "agent_routing"],
            description=(
                "Input classifier agent that uses LLM/SLM to understand natural language "
                "problem descriptions, classify them into appropriate categories, and format "
                "them for both algorithmic and SLM specialized agents."
            ),
            performance_metrics=performance_metrics,
            load_factor=0.0,
            availability=self.llm_client is not None
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """This agent handles input classification for all problems."""
        if not self.llm_client:
            return 0.0
        
        # Check if this is already a formatted request (avoid recursion)
        if problem_request.problem_type == "pre_formatted":
            return 0.0
            
        return 1.0  # Can classify any natural language input
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Classify problem and format for appropriate agent."""
        if not self.llm_client:
            return ProblemResult(
                success=False,
                error_message="LLM client not available"
            )
        
        try:
            start_time = datetime.now()
            
            # Step 1: Classify the problem using LLM
            classification = await self._classify_problem(problem_request.problem_description)
            
            if not classification:
                return ProblemResult(
                    success=False,
                    error_message="Failed to classify problem"
                )
            
            # Step 2: Format input for target agent
            formatted_request = await self._format_for_target_agent(
                problem_request, 
                classification
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ProblemResult(
                success=True,
                solution={
                    "classification": classification,
                    "formatted_request": formatted_request,
                    "target_agent": classification["target_agent"],
                    "agent_type": classification["agent_type"]
                },
                explanation=f"Problem classified as '{classification['category']}' with {classification['confidence']:.2f} confidence",
                confidence=classification["confidence"],
                execution_time=execution_time,
                metadata={
                    "original_problem": problem_request.problem_description,
                    "classification_reasoning": classification.get("reasoning", ""),
                    "input_format": classification["input_format"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in input classification: {e}")
            return ProblemResult(
                success=False,
                error_message=f"Classification error: {str(e)}"
            )
    
    async def _classify_problem(self, problem_description: str) -> Optional[Dict[str, Any]]:
        """Use LLM to classify the problem."""
        try:
            prompt = self.classification_prompt.format(
                problem_description=problem_description
            )
            
            # Call LLM (this will depend on your LLM client interface)
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            classification_data = json.loads(response)
            
            # Add agent mapping information
            category = classification_data["category"]
            if category in self.agent_type_mapping:
                mapping = self.agent_type_mapping[category]
                classification_data.update(mapping)
            else:
                # Default to general reasoning SLM
                classification_data.update({
                    "agent_type": "slm",
                    "target_agent": "reasoning_slm", 
                    "input_format": "conversation"
                })
            
            return classification_data
            
        except Exception as e:
            self.logger.error(f"LLM classification error: {e}")
            return None
    
    async def _format_for_target_agent(
        self, 
        original_request: ProblemRequest,
        classification: Dict[str, Any]
    ) -> ProblemRequest:
        """Format the problem request for the target agent."""
        
        agent_type = classification["agent_type"]
        input_format = classification["input_format"]
        extracted_params = classification.get("extracted_parameters", {})
        
        if agent_type == "algorithmic":
            return self._format_for_algorithmic_agent(
                original_request, classification, extracted_params
            )
        elif agent_type == "slm":
            return self._format_for_slm_agent(
                original_request, classification, extracted_params
            )
        else:
            # Fallback - minimal formatting
            return ProblemRequest(
                problem_description=original_request.problem_description,
                problem_type=classification["category"],
                constraints=extracted_params
            )
    
    def _format_for_algorithmic_agent(
        self,
        original_request: ProblemRequest,
        classification: Dict[str, Any], 
        extracted_params: Dict[str, Any]
    ) -> ProblemRequest:
        """Format problem for algorithmic agents like DCOP solver."""
        
        category = classification["category"]
        
        if category == "dcop":
            # Extract DCOP-specific parameters
            constraints = {
                "num_variables": self._extract_number(extracted_params, "variables", 10),
                "domain_size": self._extract_number(extracted_params, "domain", 5),
                "density": self._extract_density(extracted_params),
                "cost_range": self._extract_cost_range(extracted_params),
                "algorithm": self._extract_algorithm(extracted_params, original_request.problem_description),
                "max_iterations": 1000,
                "convergence_threshold": 1e-6
            }
            
        elif category == "graph_algorithms":
            constraints = {
                "num_nodes": self._extract_number(extracted_params, "nodes", 20),
                "num_edges": self._extract_number(extracted_params, "edges", None),
                "weighted": self._has_concept(extracted_params, ["weight", "cost", "distance"]),
                "directed": self._has_concept(extracted_params, ["directed", "asymmetric"]),
                "algorithm_type": self._extract_graph_algorithm(original_request.problem_description)
            }
            
        elif category == "optimization":
            constraints = {
                "variables": extracted_params.get("numeric_values", []),
                "objective": extracted_params.get("objective", "minimize"),
                "constraint_type": "general",
                "algorithm": "genetic_algorithm"  # default
            }
        else:
            constraints = extracted_params
        
        return ProblemRequest(
            problem_description=original_request.problem_description,
            problem_type=category,
            constraints=constraints,
            metadata={
                "formatted_by": self.agent_id,
                "original_format": "natural_language",
                "target_format": "algorithmic"
            }
        )
    
    def _format_for_slm_agent(
        self,
        original_request: ProblemRequest,
        classification: Dict[str, Any],
        extracted_params: Dict[str, Any]
    ) -> ProblemRequest:
        """Format problem for SLM agents."""
        
        # For SLMs, we mostly preserve the natural language but add structure
        return ProblemRequest(
            problem_description=original_request.problem_description,
            problem_type=classification["category"],
            constraints={
                "response_format": "detailed_explanation",
                "include_reasoning": True,
                "key_concepts": extracted_params.get("key_concepts", []),
                "context": extracted_params.get("constraints", [])
            },
            metadata={
                "formatted_by": self.agent_id,
                "original_format": "natural_language", 
                "target_format": "slm_conversation",
                "extracted_concepts": extracted_params.get("key_concepts", [])
            }
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client. This method should be implemented based on your LLM choice."""
        if not self.llm_client:
            raise ValueError("No LLM client configured")
        
        # This is a placeholder - implement based on your LLM client
        # Examples:
        # - OpenAI: await self.llm_client.chat.completions.create(...)
        # - Anthropic: await self.llm_client.messages.create(...)
        # - Local model: await self.llm_client.generate(...)
        
        # For now, return a mock response for testing
        return """{
            "category": "dcop",
            "confidence": 0.9,
            "extracted_parameters": {
                "key_concepts": ["constraint", "optimization", "variables"],
                "numeric_values": [10, 5],
                "constraints": ["minimize cost", "satisfy constraints"],
                "objective": "find optimal variable assignment"
            },
            "reasoning": "Problem mentions constraints, optimization, and variable assignment"
        }"""
    
    # Helper methods for parameter extraction
    def _extract_number(self, params: Dict, concept: str, default: int) -> int:
        """Extract number related to a concept."""
        numbers = params.get("numeric_values", [])
        concepts = params.get("key_concepts", [])
        
        # Simple heuristic - could be improved with better NLP
        if numbers:
            return min(int(numbers[0]), 50)  # Cap for safety
        return default
    
    def _extract_density(self, params: Dict) -> float:
        """Extract density parameter."""
        if self._has_concept(params, ["dense", "heavily connected"]):
            return 0.7
        elif self._has_concept(params, ["sparse", "few connections"]):
            return 0.2
        return 0.3  # default
    
    def _extract_cost_range(self, params: Dict) -> List[int]:
        """Extract cost range."""
        numbers = params.get("numeric_values", [])
        if len(numbers) >= 2:
            return [int(min(numbers)), int(max(numbers))]
        return [0, 100]  # default
    
    def _extract_algorithm(self, params: Dict, description: str) -> str:
        """Extract algorithm preference."""
        desc_lower = description.lower()
        if "max" in desc_lower:
            return "max_sum"
        return "min_sum"  # default
    
    def _extract_graph_algorithm(self, description: str) -> str:
        """Extract graph algorithm type."""
        desc_lower = description.lower()
        if "shortest" in desc_lower or "path" in desc_lower:
            return "shortest_path"
        elif "spanning" in desc_lower or "mst" in desc_lower:
            return "minimum_spanning_tree"
        elif "flow" in desc_lower:
            return "max_flow"
        elif "coloring" in desc_lower:
            return "graph_coloring"
        return "general_graph"
    
    def _has_concept(self, params: Dict, concepts: List[str]) -> bool:
        """Check if any of the concepts appear in extracted parameters."""
        key_concepts = [c.lower() for c in params.get("key_concepts", [])]
        constraints = [c.lower() for c in params.get("constraints", [])]
        all_text = " ".join(key_concepts + constraints)
        
        return any(concept.lower() in all_text for concept in concepts)