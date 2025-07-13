"""Base class for Small Language Model (SLM) agents."""

from abc import abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import logging

from ..core.agent_base import BaseAgent
from ..core.message import (
    ProblemRequest, 
    ProblemResult, 
    AgentCapability
)


class SLMBaseAgent(BaseAgent):
    """
    Base class for agents that use Small Language Models for reasoning,
    code generation, text analysis, etc.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        agent_type: str,
        model_name: str = "unknown",
        model_client=None
    ):
        super().__init__(agent_id, agent_type)
        
        self.model_name = model_name
        self.model_client = model_client
        
        # Common SLM configuration
        self.generation_config = {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["<|endoftext|>", "</response>"]
        }
        
        # Response quality tracking
        self.response_quality_scores = []
        
    @abstractmethod
    async def get_system_prompt(self) -> str:
        """Return the system prompt for this SLM agent."""
        pass
    
    @abstractmethod
    def get_supported_problem_types(self) -> List[str]:
        """Return list of problem types this SLM can handle."""
        pass
    
    @abstractmethod
    async def format_prompt(self, problem_request: ProblemRequest) -> str:
        """Format the problem request into a prompt for the SLM."""
        pass
    
    @abstractmethod
    async def parse_response(self, raw_response: str, problem_request: ProblemRequest) -> Dict[str, Any]:
        """Parse the SLM's raw response into structured solution data.""" 
        pass
    
    async def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities."""
        performance_metrics = self.get_performance_metrics()
        
        # Add SLM-specific metrics
        performance_metrics.update({
            "model_name": self.model_name,
            "average_response_quality": (
                sum(self.response_quality_scores) / len(self.response_quality_scores)
                if self.response_quality_scores else 0.0
            ),
            "model_available": self.model_client is not None
        })
        
        return AgentCapability(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            problem_types=self.get_supported_problem_types(),
            description=f"SLM-powered agent using {self.model_name} for {self.agent_type} tasks",
            performance_metrics=performance_metrics,
            load_factor=0.0,  # Could implement actual load tracking
            availability=self.model_client is not None
        )
    
    async def can_solve_problem(self, problem_request: ProblemRequest) -> float:
        """Determine if this SLM agent can solve the given problem."""
        if not self.model_client:
            return 0.0
        
        problem_type = problem_request.problem_type
        supported_types = self.get_supported_problem_types()
        
        # Exact match
        if problem_type in supported_types:
            return 0.9
        
        # Partial match based on keywords
        if problem_type:
            for supported_type in supported_types:
                if supported_type in problem_type or problem_type in supported_type:
                    return 0.7
        
        # Check problem description for relevant keywords
        description = problem_request.problem_description.lower()
        confidence = await self._assess_problem_relevance(description)
        
        return confidence
    
    async def solve_problem(self, problem_request: ProblemRequest) -> ProblemResult:
        """Solve problem using the SLM."""
        if not self.model_client:
            return ProblemResult(
                success=False,
                error_message=f"Model client not available for {self.model_name}"
            )
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Format prompt
            prompt = await self.format_prompt(problem_request)
            system_prompt = await self.get_system_prompt()
            
            # Generate response
            raw_response = await self._call_model(system_prompt, prompt)
            
            # Parse response
            parsed_solution = await self.parse_response(raw_response, problem_request)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Assess response quality
            quality_score = await self._assess_response_quality(
                problem_request, parsed_solution, raw_response
            )
            self.response_quality_scores.append(quality_score)
            
            # Keep only last 100 scores for running average
            if len(self.response_quality_scores) > 100:
                self.response_quality_scores.pop(0)
            
            return ProblemResult(
                success=True,
                solution=parsed_solution,
                explanation=parsed_solution.get("explanation", raw_response),
                confidence=quality_score,
                execution_time=execution_time,
                metadata={
                    "model_name": self.model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(raw_response),
                    "quality_score": quality_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"SLM agent error: {e}")
            return ProblemResult(
                success=False,
                error_message=f"SLM error: {str(e)}"
            )
    
    async def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        """Call the SLM model. Override this method for different model APIs."""
        if not self.model_client:
            raise ValueError("No model client configured")
        
        # This is a placeholder - implement based on your model choice:
        # - For local models: use transformers, vLLM, etc.
        # - For API models: OpenAI, Anthropic, Cohere, etc.
        # - For on-device: ONNX, Core ML, TensorFlow Lite
        
        # Mock implementation
        return f"Mock response for prompt: {user_prompt[:100]}..."
    
    async def _assess_problem_relevance(self, description: str) -> float:
        """Assess how relevant this problem is to this SLM agent."""
        # Override in subclasses for specific relevance scoring
        return 0.5
    
    async def _assess_response_quality(
        self, 
        problem_request: ProblemRequest,
        parsed_solution: Dict[str, Any],
        raw_response: str
    ) -> float:
        """Assess the quality of the generated response."""
        
        quality_score = 0.8  # Base score
        
        # Check if response is complete
        if len(raw_response.strip()) < 50:
            quality_score -= 0.3  # Very short response
        
        # Check if structured data was parsed successfully
        if not parsed_solution or "error" in parsed_solution:
            quality_score -= 0.2
        
        # Check for common failure patterns
        if "I cannot" in raw_response or "I don't know" in raw_response:
            quality_score -= 0.3
        
        # Check for reasoning/explanation
        if "explanation" in parsed_solution and len(parsed_solution["explanation"]) > 100:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))


class ReasoningSLMAgent(SLMBaseAgent):
    """SLM agent specialized in logical reasoning and problem solving."""
    
    def __init__(self, agent_id: str = "reasoning_slm", model_name: str = "llama-3-8b", model_client=None):
        super().__init__(agent_id, "reasoning_slm", model_name, model_client)
    
    async def get_system_prompt(self) -> str:
        return """You are a logical reasoning assistant. You excel at:
- Breaking down complex problems into steps
- Applying logical reasoning principles
- Solving puzzles and brain teasers
- Providing clear explanations of your reasoning process

Always structure your responses with:
1. Problem understanding
2. Step-by-step reasoning
3. Final answer
4. Explanation of why this answer is correct"""
    
    def get_supported_problem_types(self) -> List[str]:
        return [
            "natural_language_reasoning",
            "logical_reasoning", 
            "puzzle_solving",
            "question_answering",
            "problem_decomposition"
        ]
    
    async def format_prompt(self, problem_request: ProblemRequest) -> str:
        context = problem_request.constraints.get("context", []) if problem_request.constraints else []
        context_str = "\n".join(context) if context else ""
        
        prompt = f"""Problem: {problem_request.problem_description}

{f"Additional Context: {context_str}" if context_str else ""}

Please solve this step by step, showing your reasoning clearly."""
        
        return prompt
    
    async def parse_response(self, raw_response: str, problem_request: ProblemRequest) -> Dict[str, Any]:
        """Parse reasoning response into structured format."""
        
        # Try to extract structured parts
        lines = raw_response.split('\n')
        
        solution = {
            "reasoning_steps": [],
            "final_answer": "",
            "explanation": raw_response,
            "confidence": "medium"
        }
        
        # Simple parsing logic - could be improved with better NLP
        current_step = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for step indicators
            if any(indicator in line.lower() for indicator in ["step", "first", "second", "third", "next", "then"]):
                if current_step:
                    solution["reasoning_steps"].append(current_step)
                current_step = line
            elif any(indicator in line.lower() for indicator in ["answer", "conclusion", "result", "solution"]):
                solution["final_answer"] = line
            else:
                current_step += " " + line if current_step else line
        
        if current_step:
            solution["reasoning_steps"].append(current_step)
        
        return solution
    
    async def _assess_problem_relevance(self, description: str) -> float:
        """Assess relevance for reasoning tasks."""
        reasoning_keywords = [
            "logic", "reason", "puzzle", "solve", "think", "analyze", 
            "why", "how", "explain", "prove", "deduce", "infer"
        ]
        
        keyword_count = sum(1 for keyword in reasoning_keywords if keyword in description)
        return min(keyword_count * 0.2, 0.9)


class CodingSLMAgent(SLMBaseAgent):
    """SLM agent specialized in code generation and programming tasks."""
    
    def __init__(self, agent_id: str = "coding_slm", model_name: str = "codellama-7b", model_client=None):
        super().__init__(agent_id, "coding_slm", model_name, model_client)
    
    async def get_system_prompt(self) -> str:
        return """You are a programming assistant. You excel at:
- Writing clean, efficient code
- Explaining programming concepts
- Debugging and optimization
- Algorithm implementation

Always provide:
1. Working code with comments
2. Explanation of the approach
3. Time/space complexity analysis where relevant
4. Example usage or test cases"""
    
    def get_supported_problem_types(self) -> List[str]:
        return [
            "code_generation",
            "algorithm_implementation",
            "programming_help",
            "code_explanation",
            "debugging_assistance"
        ]
    
    async def format_prompt(self, problem_request: ProblemRequest) -> str:
        constraints = problem_request.constraints or {}
        language = constraints.get("programming_language", "Python")
        requirements = constraints.get("requirements", [])
        
        prompt = f"""Programming Task: {problem_request.problem_description}

Language: {language}
"""
        
        if requirements:
            prompt += f"Requirements:\n"
            for req in requirements:
                prompt += f"- {req}\n"
        
        prompt += "\nPlease provide a complete solution with explanation."
        
        return prompt
    
    async def parse_response(self, raw_response: str, problem_request: ProblemRequest) -> Dict[str, Any]:
        """Parse coding response into structured format."""
        
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(?:python|java|cpp|c\+\+|javascript|js)?\n(.*?)\n```', raw_response, re.DOTALL)
        
        solution = {
            "code": code_blocks[0] if code_blocks else "",
            "explanation": raw_response,
            "code_blocks": code_blocks,
            "language": "python",  # default
            "complexity_analysis": ""
        }
        
        # Try to extract complexity analysis
        if "time complexity" in raw_response.lower() or "space complexity" in raw_response.lower():
            lines = raw_response.split('\n')
            complexity_lines = [line for line in lines if "complexity" in line.lower()]
            solution["complexity_analysis"] = "\n".join(complexity_lines)
        
        return solution
    
    async def _assess_problem_relevance(self, description: str) -> float:
        """Assess relevance for coding tasks."""
        coding_keywords = [
            "code", "program", "algorithm", "function", "implement", "write",
            "debug", "optimize", "script", "software", "programming"
        ]
        
        keyword_count = sum(1 for keyword in coding_keywords if keyword in description)
        return min(keyword_count * 0.25, 0.95)