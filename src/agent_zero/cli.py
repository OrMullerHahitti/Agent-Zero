"""Command line interface for Agent Zero system."""

import asyncio
import logging
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from .core.registry import AgentRegistry
from .protocols.communication import InMemoryMessageBus
from .orchestrator.main_orchestrator import MainOrchestrator
from .agents.belief_propagation_agent import BeliefPropagationAgent
from .core.message import ProblemRequest

app = typer.Typer(
    name="agent-zero",
    help="Agent Zero: Distributed Agentic AI System for Algorithmic Problem Solving"
)
console = Console()


class AgentZeroSystem:
    """Main system coordinator."""
    
    def __init__(self):
        self.message_bus: Optional[InMemoryMessageBus] = None
        self.registry: Optional[AgentRegistry] = None
        self.orchestrator: Optional[MainOrchestrator] = None
        self.agents = []
        self.running = False
    
    async def start(self):
        """Start the Agent Zero system."""
        if self.running:
            return
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.message_bus = InMemoryMessageBus()
        self.registry = AgentRegistry()
        self.orchestrator = MainOrchestrator(self.message_bus, self.registry)
        
        # Start core components
        await self.message_bus.start()
        await self.registry.start()
        await self.orchestrator.start()
        
        # Register orchestrator
        await self.registry.register_agent(self.orchestrator.get_registration_info())
        
        # Create and register belief propagation agent
        bp_agent = BeliefPropagationAgent()
        await bp_agent.start()
        await self.message_bus.subscribe(bp_agent.agent_id, bp_agent._handle_message_callback)
        await self.registry.register_agent(bp_agent.get_registration_info())
        self.agents.append(bp_agent)
        
        self.running = True
        console.print("[green]✓[/green] Agent Zero system started successfully")
    
    async def stop(self):
        """Stop the Agent Zero system."""
        if not self.running:
            return
        
        # Stop agents
        for agent in self.agents:
            await agent.stop()
        
        # Stop core components
        if self.orchestrator:
            await self.orchestrator.stop()
        if self.registry:
            await self.registry.stop()
        if self.message_bus:
            await self.message_bus.stop()
        
        self.running = False
        console.print("[red]✓[/red] Agent Zero system stopped")
    
    async def solve_problem(self, description: str, problem_type: Optional[str] = None) -> None:
        """Solve a problem using the orchestrator."""
        if not self.running:
            console.print("[red]Error:[/red] System not running. Start it first.")
            return
        
        problem_request = ProblemRequest(
            problem_description=description,
            problem_type=problem_type
        )
        
        console.print(f"\n[blue]🤖 Analyzing problem:[/blue] {description}")
        
        with console.status("[bold green]Solving problem..."):
            result = await self.orchestrator.solve_problem(problem_request)
        
        # Display results
        if result.success:
            console.print(Panel(
                f"[green]✓ Solution found![/green]\n\n{result.explanation or 'No explanation provided'}",
                title="Problem Solved",
                border_style="green"
            ))
            
            if result.solution:
                console.print("\n[blue]Solution Details:[/blue]")
                for key, value in result.solution.items():
                    console.print(f"  {key}: {value}")
            
            if result.execution_time:
                console.print(f"\n[dim]Execution time: {result.execution_time:.2f} seconds[/dim]")
        else:
            console.print(Panel(
                f"[red]✗ Failed to solve problem[/red]\n\n{result.error_message}",
                title="Solution Failed",
                border_style="red"
            ))
    
    async def show_status(self):
        """Show system status."""
        if not self.running:
            console.print("[red]System is not running[/red]")
            return
        
        status = await self.orchestrator.get_system_status()
        
        # System overview
        console.print(Panel(
            f"[green]System Status: {status['system']['system_health'].title()}[/green]\n"
            f"Active Agents: {status['registry']['active_agents']}\n"
            f"Total Problem Types: {status['system']['total_problem_types_supported']}\n"
            f"Success Rate: {status['orchestrator']['success_rate_percent']:.1f}%",
            title="Agent Zero System",
            border_style="green" if status['system']['system_health'] == 'healthy' else "yellow"
        ))
        
        # Agent table
        agents_table = Table(title="Registered Agents")
        agents_table.add_column("Agent ID", style="cyan")
        agents_table.add_column("Type", style="magenta")
        agents_table.add_column("Status", style="green")
        agents_table.add_column("Problem Types", style="blue")
        
        agents = self.registry.get_all_agents()
        for agent in agents:
            capability = self.registry.get_agent_capabilities(agent.agent_id)
            if capability:
                problem_types = ", ".join(capability.problem_types[:3])
                if len(capability.problem_types) > 3:
                    problem_types += "..."
                
                agents_table.add_row(
                    agent.agent_id,
                    agent.agent_type,
                    "🟢 Active" if capability.availability else "🔴 Inactive",
                    problem_types
                )
        
        console.print(agents_table)


# Global system instance
system = AgentZeroSystem()


@app.command()
def start():
    """Start the Agent Zero system."""
    asyncio.run(system.start())


@app.command()
def stop():
    """Stop the Agent Zero system."""
    asyncio.run(system.stop())


@app.command()
def solve(
    description: str = typer.Argument(..., help="Problem description"),
    problem_type: Optional[str] = typer.Option(None, "--type", "-t", help="Problem type (optional)")
):
    """Solve a problem using Agent Zero."""
    async def _solve():
        await system.start()
        await system.solve_problem(description, problem_type)
    
    asyncio.run(_solve())


@app.command()
def status():
    """Show system status and registered agents."""
    async def _status():
        await system.start()
        await system.show_status()
    
    asyncio.run(_status())


@app.command()
def interactive():
    """Start interactive mode."""
    async def _interactive():
        await system.start()
        
        console.print(Panel(
            "[bold blue]Agent Zero Interactive Mode[/bold blue]\n\n"
            "Commands:\n"
            "  solve <description>  - Solve a problem\n"
            "  status              - Show system status\n"
            "  agents              - List agents\n"
            "  quit                - Exit\n",
            title="Welcome to Agent Zero",
            border_style="blue"
        ))
        
        try:
            while True:
                command = console.input("\n[bold blue]agent-zero>[/bold blue] ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'status':
                    await system.show_status()
                elif command.lower() == 'agents':
                    await system.show_status()
                elif command.startswith('solve '):
                    problem_desc = command[6:].strip()
                    if problem_desc:
                        await system.solve_problem(problem_desc)
                    else:
                        console.print("[red]Please provide a problem description[/red]")
                elif command:
                    console.print(f"[red]Unknown command: {command}[/red]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            await system.stop()
    
    asyncio.run(_interactive())


@app.command()
def example():
    """Run example DCOP problems."""
    examples = [
        "Solve a constraint optimization problem with 5 variables and domain size 3",
        "Find optimal assignment for a distributed constraint optimization problem",
        "Minimize cost in a factor graph with 10 variables"
    ]
    
    async def _run_examples():
        await system.start()
        
        console.print(Panel(
            "[bold blue]Running Example Problems[/bold blue]",
            title="Agent Zero Examples",
            border_style="blue"
        ))
        
        for i, example in enumerate(examples, 1):
            console.print(f"\n[bold]Example {i}:[/bold]")
            await system.solve_problem(example)
            
            if i < len(examples):
                console.input("\nPress Enter to continue to next example...")
    
    asyncio.run(_run_examples())


def main():
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()