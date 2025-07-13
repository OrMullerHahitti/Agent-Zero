"""
Demo of the hybrid Agent Zero system with both SLM and algorithmic agents.
This example shows how natural language problems are classified, routed, 
solved, and formatted back to natural language.
"""

import asyncio
import logging
from typing import Dict, Any

# Mock LLM/SLM clients for demonstration
class MockLLMClient:
    """Mock LLM client for demonstration purposes."""
    
    async def generate(self, prompt: str) -> str:
        """Mock LLM generation."""
        if "classification" in prompt.lower():
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
        elif "format" in prompt.lower():
            return """The problem was asking to find the best assignment of values to variables while minimizing cost.

The belief propagation algorithm solved this by:
1. Creating a factor graph representing the constraints
2. Passing messages between variables and factors iteratively  
3. Converging to an optimal solution after 127 iterations

The final solution assigns each variable a value that minimizes total cost while respecting all constraints. 
The total cost achieved is 45.2, representing an optimal solution.

This solution is good because it satisfies all constraints while achieving the lowest possible cost 
through the systematic message-passing approach of belief propagation."""
        else:
            return "Mock response for: " + prompt[:100]


class MockSLMClient:
    """Mock SLM client for demonstration purposes."""
    
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Mock SLM generation."""
        if "reasoning" in system_prompt.lower():
            return """To solve this logic puzzle, I need to work through it step by step:

Step 1: Identify what we know
- We have 3 people: Alice, Bob, and Charlie
- Each has a different profession: doctor, lawyer, teacher
- We have clues about who does what

Step 2: Apply logical deduction
- If Alice is not the doctor, and Bob is not the teacher
- Then Charlie must be either the doctor or teacher
- But if Charlie is the teacher, then Alice must be the lawyer
- This means Bob is the doctor

Step 3: Verify the solution
- Bob: Doctor
- Alice: Lawyer  
- Charlie: Teacher

This solution satisfies all the given constraints."""
        elif "coding" in system_prompt.lower():
            return """Here's a Python solution for the sorting problem:

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Example usage
numbers = [3, 6, 8, 10, 1, 2, 1]
sorted_numbers = quicksort(numbers)
print(sorted_numbers)  # Output: [1, 1, 2, 3, 6, 8, 10]
```

This implementation uses the divide-and-conquer approach:
- Time Complexity: O(n log n) average case, O(n²) worst case
- Space Complexity: O(log n) for recursion stack

The algorithm works by selecting a pivot and partitioning the array around it."""
        else:
            return "Mock SLM response"


async def demo_hybrid_system():
    """Demonstrate the hybrid Agent Zero system."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo")
    
    logger.info("Starting Agent Zero Hybrid System Demo")
    
    # Mock clients
    llm_client = MockLLMClient()
    slm_client = MockSLMClient()
    
    # Import Agent Zero components
    from src.agent_zero.core.registry import AgentRegistry
    from src.agent_zero.protocols.communication import MessageBus, MessageRouter
    from src.agent_zero.agents.input_classifier_agent import InputClassifierAgent
    from src.agent_zero.agents.output_formatter_agent import OutputFormatterAgent
    from src.agent_zero.agents.belief_propagation_agent import BeliefPropagationAgent
    from src.agent_zero.agents.slm_base import ReasoningSLMAgent, CodingSLMAgent
    from src.agent_zero.orchestrator.hybrid_orchestrator import HybridOrchestrator
    from src.agent_zero.core.message import ProblemRequest
    
    try:
        # 1. Setup system components
        logger.info("Setting up system components...")
        
        message_bus = MessageBus()
        agent_registry = AgentRegistry()
        
        # 2. Create agents
        logger.info("Creating agents...")
        
        # AI-powered agents
        input_classifier = InputClassifierAgent(llm_client=llm_client)
        output_formatter = OutputFormatterAgent(slm_client=slm_client)
        
        # Algorithmic agents
        bp_agent = BeliefPropagationAgent()
        
        # SLM agents
        reasoning_agent = ReasoningSLMAgent(model_client=slm_client)
        coding_agent = CodingSLMAgent(model_client=slm_client)
        
        # Hybrid orchestrator
        orchestrator = HybridOrchestrator(
            message_bus=message_bus,
            agent_registry=agent_registry,
            input_classifier=input_classifier,
            output_formatter=output_formatter
        )
        
        # 3. Start all agents
        logger.info("Starting agents...")
        
        await message_bus.start()
        await input_classifier.start()
        await output_formatter.start()
        await bp_agent.start()
        await reasoning_agent.start()
        await coding_agent.start()
        await orchestrator.start()
        
        # 4. Register agents
        logger.info("Registering agents...")
        
        agents = [bp_agent, reasoning_agent, coding_agent]
        for agent in agents:
            capability = await agent.get_capabilities()
            agent_registry.register_agent(agent.get_registration_info(), capability)
        
        # 5. Demo different types of problems
        logger.info("\n" + "="*60)
        logger.info("DEMO: Natural Language Problem Solving")
        logger.info("="*60)
        
        test_problems = [
            {
                "name": "DCOP Problem",
                "description": "I have 10 variables and need to find optimal assignments that minimize cost while satisfying constraints"
            },
            {
                "name": "Logic Puzzle",
                "description": "Alice, Bob, and Charlie are a doctor, lawyer, and teacher. Alice is not the doctor. Bob is not the teacher. What profession is each person?"
            },
            {
                "name": "Coding Task", 
                "description": "Write a Python function to sort a list of numbers using quicksort algorithm"
            }
        ]
        
        for problem in test_problems:
            logger.info(f"\n--- Testing: {problem['name']} ---")
            logger.info(f"Problem: {problem['description']}")
            
            # Create problem request
            request = ProblemRequest(
                problem_description=problem['description']
            )
            
            # Solve using hybrid orchestrator
            result = await orchestrator.solve_problem(request)
            
            if result.success:
                logger.info("✅ Solution found!")
                logger.info(f"Explanation: {result.explanation[:200]}...")
                
                if result.metadata:
                    strategy = result.metadata.get("routing_strategy", "unknown")
                    logger.info(f"Routing Strategy: {strategy}")
                    
                    if "pipeline_result" in result.metadata:
                        pipeline = result.metadata["pipeline_result"]
                        logger.info(f"Methodology: {pipeline.get('methodology', 'unknown')}")
                        insights = pipeline.get('insights', [])
                        if insights:
                            logger.info(f"Key Insights: {', '.join(insights[:2])}")
            else:
                logger.error(f"❌ Failed: {result.error_message}")
        
        # 6. Show system analytics
        logger.info(f"\n--- System Analytics ---")
        analytics = await orchestrator.get_routing_analytics()
        logger.info(f"Total routing decisions: {analytics.get('total_decisions', 0)}")
        logger.info(f"Overall success rate: {analytics.get('overall_success_rate', 0):.1f}%")
        
        strategy_perf = analytics.get('strategy_performance', {})
        for strategy, stats in strategy_perf.items():
            logger.info(f"{strategy}: {stats['success_rate']:.1f}% success, {stats['avg_time']:.2f}s avg")
        
        # 7. Demo direct agent access
        logger.info(f"\n--- Direct Agent Testing ---")
        
        # Test reasoning agent directly
        reasoning_request = ProblemRequest(
            problem_description="What is 2 + 2 and why?",
            problem_type="natural_language_reasoning"
        )
        
        reasoning_result = await reasoning_agent.solve_problem(reasoning_request)
        if reasoning_result.success:
            logger.info("✅ Direct reasoning agent test passed")
            steps = reasoning_result.solution.get("reasoning_steps", [])
            logger.info(f"Reasoning steps: {len(steps)}")
        
        # Test BP agent directly  
        dcop_request = ProblemRequest(
            problem_description="Solve DCOP with 5 variables",
            problem_type="dcop",
            constraints={
                "num_variables": 5,
                "domain_size": 3,
                "algorithm": "min_sum"
            }
        )
        
        bp_result = await bp_agent.solve_problem(dcop_request)
        if bp_result.success:
            logger.info("✅ Direct BP agent test passed")
            assignments = bp_result.solution.get("assignments", {})
            logger.info(f"Variable assignments: {len(assignments)}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        try:
            await orchestrator.stop()
            await bp_agent.stop()
            await reasoning_agent.stop()
            await coding_agent.stop()
            await input_classifier.stop()
            await output_formatter.stop()
            await message_bus.stop()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    logger.info("Demo completed!")


async def demo_pipeline_flow():
    """Demonstrate the complete pipeline flow in detail."""
    
    logger = logging.getLogger("pipeline_demo")
    logger.info("\n" + "="*60)
    logger.info("DETAILED PIPELINE FLOW DEMO")
    logger.info("="*60)
    
    # Mock the pipeline steps
    problem_text = "I need to optimize variable assignments for 8 variables with domain size 4 to minimize total cost"
    
    logger.info(f"1. INPUT: {problem_text}")
    
    logger.info("2. CLASSIFICATION:")
    logger.info("   - LLM analyzes natural language")
    logger.info("   - Extracts: 8 variables, domain size 4, minimization objective")
    logger.info("   - Classifies as: DCOP problem")
    logger.info("   - Routes to: belief_propagation_agent")
    
    logger.info("3. ALGORITHMIC SOLVING:")
    logger.info("   - Creates factor graph with 8 variables")
    logger.info("   - Runs min-sum belief propagation")
    logger.info("   - Converges after 156 iterations")
    logger.info("   - Total cost: 67.3")
    
    logger.info("4. OUTPUT FORMATTING:")
    logger.info("   - SLM converts technical results to natural language")
    logger.info("   - Explains methodology and reasoning")
    logger.info("   - Provides insights about solution quality")
    
    logger.info("5. FINAL RESPONSE:")
    response = """The problem asked to find the best way to assign values to 8 variables, 
each with 4 possible values, while minimizing the total cost.

I solved this using belief propagation on a factor graph:
1. Created a network representing all variables and their relationships
2. Used message passing to find the optimal assignment
3. The algorithm converged after 156 iterations

The final solution assigns specific values to each variable that achieves 
a total cost of 67.3, which is optimal given the constraints. This solution 
balances all the competing factors to find the best possible outcome."""
    
    logger.info(f"   Output: {response}")
    
    logger.info("\n✅ Pipeline demonstrates seamless natural language to algorithmic solving!")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_hybrid_system())
    asyncio.run(demo_pipeline_flow())