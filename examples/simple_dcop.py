"""Simple example of solving a DCOP problem with Agent Zero."""

import asyncio
import logging
from agent_zero.core.registry import AgentRegistry
from agent_zero.protocols.communication import InMemoryMessageBus
from agent_zero.orchestrator.main_orchestrator import MainOrchestrator
from agent_zero.agents.belief_propagation_agent import BeliefPropagationAgent
from agent_zero.core.message import ProblemRequest


async def main():
    """Run a simple DCOP solving example."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 Starting Agent Zero System...")
    
    # Initialize system components
    message_bus = InMemoryMessageBus()
    registry = AgentRegistry()
    orchestrator = MainOrchestrator(message_bus, registry)
    
    # Start components
    await message_bus.start()
    await registry.start()
    await orchestrator.start()
    
    # Register orchestrator
    await registry.register_agent(orchestrator.get_registration_info())
    
    # Create and register belief propagation agent
    bp_agent = BeliefPropagationAgent()
    await bp_agent.start()
    await message_bus.subscribe(bp_agent.agent_id, bp_agent._handle_message_callback)
    await registry.register_agent(bp_agent.get_registration_info())
    
    print("✅ System started successfully!")
    
    # Show system status
    print("\n📊 System Status:")
    status = await orchestrator.get_system_status()
    print(f"   Active Agents: {status['registry']['active_agents']}")
    print(f"   Problem Types Supported: {status['system']['total_problem_types_supported']}")
    
    # List registered agents
    print("\n🤖 Registered Agents:")
    agents = registry.get_all_agents()
    for agent in agents:
        capability = registry.get_agent_capabilities(agent.agent_id)
        if capability:
            problem_types = ", ".join(capability.problem_types[:3])
            print(f"   • {agent.agent_id} ({agent.agent_type}): {problem_types}")
    
    # Define example problems
    problems = [
        {
            "description": "Solve a constraint optimization problem with 5 variables and domain size 3",
            "constraints": {
                "num_variables": 5,
                "domain_size": 3,
                "density": 0.4
            }
        },
        {
            "description": "Find optimal assignment for a distributed constraint optimization problem",
            "constraints": {
                "num_variables": 8,
                "domain_size": 4,
                "density": 0.3,
                "cost_range": [0, 50]
            }
        },
        {
            "description": "Minimize cost in a factor graph with belief propagation",
            "problem_type": "dcop",
            "constraints": {
                "num_variables": 6,
                "domain_size": 2,
                "density": 0.5
            }
        }
    ]
    
    # Solve each problem
    for i, problem_def in enumerate(problems, 1):
        print(f"\n🔍 Problem {i}: {problem_def['description']}")
        
        # Create problem request
        problem_request = ProblemRequest(
            problem_description=problem_def['description'],
            problem_type=problem_def.get('problem_type'),
            constraints=problem_def.get('constraints', {})
        )
        
        # Solve the problem
        print("   🧠 Analyzing and solving...")
        result = await orchestrator.solve_problem(problem_request)
        
        # Display results
        if result.success:
            print("   ✅ Solution found!")
            if result.solution:
                assignments = result.solution.get('assignments', {})
                if assignments:
                    print("   📋 Variable Assignments:")
                    for var, value in list(assignments.items())[:5]:  # Show first 5
                        print(f"      {var}: {value}")
                    if len(assignments) > 5:
                        print(f"      ... and {len(assignments) - 5} more variables")
                
                total_cost = result.solution.get('total_cost', 'N/A')
                print(f"   💰 Total Cost: {total_cost}")
                
                if result.execution_time:
                    print(f"   ⏱️  Execution Time: {result.execution_time:.2f} seconds")
            
            if result.explanation:
                print("   📝 Explanation:")
                # Show first few lines of explanation
                lines = result.explanation.strip().split('\n')[:5]
                for line in lines:
                    print(f"      {line}")
        else:
            print(f"   ❌ Failed: {result.error_message}")
        
        # Wait between problems
        if i < len(problems):
            await asyncio.sleep(1)
    
    # Show final system statistics
    print(f"\n📈 Final Statistics:")
    final_status = await orchestrator.get_system_status()
    orch_stats = final_status['orchestrator']
    print(f"   Problems Solved: {orch_stats['successful_solutions']}")
    print(f"   Problems Failed: {orch_stats['failed_solutions']}")
    print(f"   Success Rate: {orch_stats['success_rate_percent']:.1f}%")
    
    # Cleanup
    print("\n🛑 Shutting down system...")
    await bp_agent.stop()
    await orchestrator.stop()
    await registry.stop()
    await message_bus.stop()
    
    print("✅ System shutdown complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()