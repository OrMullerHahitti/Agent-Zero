# Agent Zero: Distributed Agentic AI System

A sophisticated multi-agent AI system that orchestrates specialized Small Language Models (SLMs) for algorithmic problem-solving, with distributed intelligence and domain expertise.

## 🎯 Overview

Agent Zero creates a network of specialized AI agents, each focusing on specific problem domains, with intelligent routing and coordination. Rather than using one large model for everything, the system employs multiple specialized SLMs working together.

### Key Features

- **🔄 Distributed Intelligence**: Multiple specialized SLMs working together
- **🎯 Domain Expertise**: Each agent specializes in specific algorithmic problem types  
- **🔌 Extensible Architecture**: Easy addition of new specialized agents
- **🧠 Intelligent Routing**: Automatic problem classification and routing
- **📊 Real-time Monitoring**: Performance tracking and system health monitoring
- **🛠️ PropFlow Integration**: Built-in DCOP solver using belief propagation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Main           │    │   Message Bus    │    │  Agent Registry │
│  Orchestrator   │◄──►│  Communication   │◄──►│  & Discovery    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Problem        │    │  Specialized     │    │  Result         │
│  Classification │    │  Agent Pool      │    │  Aggregation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ BP Agent     │ │ Future:      │ │ Future:      │
        │ (DCOP/       │ │ Graph        │ │ Search       │
        │ PropFlow)    │ │ Algorithms   │ │ Algorithms   │
        └──────────────┘ └──────────────┘ └──────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Agent_Zero

# Install in development mode
pip install -e .

# Or install with PropFlow integration
pip install -e ".[propflow]"
```

### Basic Usage

#### Command Line Interface

```bash
# Start interactive mode
agent-zero interactive

# Solve a problem directly
agent-zero solve "Optimize a constraint satisfaction problem with 5 variables"

# Show system status
agent-zero status

# Run examples
agent-zero example
```

#### Python API

```python
import asyncio
from agent_zero import MainOrchestrator
from agent_zero.core.registry import AgentRegistry
from agent_zero.protocols.communication import InMemoryMessageBus
from agent_zero.agents import BeliefPropagationAgent
from agent_zero.core.message import ProblemRequest

async def solve_problem():
    # Initialize system
    message_bus = InMemoryMessageBus()
    registry = AgentRegistry()
    orchestrator = MainOrchestrator(message_bus, registry)
    
    # Start components
    await message_bus.start()
    await registry.start()
    await orchestrator.start()
    
    # Register agents
    bp_agent = BeliefPropagationAgent()
    await bp_agent.start()
    await registry.register_agent(bp_agent.get_registration_info())
    
    # Solve a problem
    problem = ProblemRequest(
        problem_description="Solve a DCOP with 5 variables and domain size 3",
        constraints={"num_variables": 5, "domain_size": 3}
    )
    
    result = await orchestrator.solve_problem(problem)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Solution: {result.solution}")

asyncio.run(solve_problem())
```

## 🤖 Available Agents

### Belief Propagation Agent
- **Specialty**: Distributed Constraint Optimization Problems (DCOP)
- **Integration**: PropFlow library
- **Algorithms**: Min-sum, Max-sum, Damping, Splitting
- **Problem Types**: `dcop`, `constraint_optimization`, `factor_graph`

### Future Agents (Planned)
- **Graph Algorithms**: Shortest path, MST, network flows
- **Search Algorithms**: A*, beam search, local search  
- **Optimization**: Linear programming, genetic algorithms
- **Machine Learning**: Classification, regression, clustering

## 📋 Examples

### Example 1: Simple DCOP
```python
# Run the complete example
python examples/simple_dcop.py
```

### Example 2: Interactive Problem Solving
```bash
agent-zero interactive
# Then type: solve "Find optimal assignment for 10 variables with domain size 4"
```

### Example 3: Constraint Optimization
```python
problem = ProblemRequest(
    problem_description="Minimize cost in a distributed system",
    constraints={
        "num_variables": 8,
        "domain_size": 3,
        "density": 0.4,
        "cost_range": [0, 100]
    }
)
result = await orchestrator.solve_problem(problem)
```

## 🔧 Development

### Project Structure
```
Agent_Zero/
├── src/agent_zero/          # Main package
│   ├── core/                # Core components
│   ├── agents/              # Specialized agents
│   ├── protocols/           # Communication protocols
│   ├── orchestrator/        # Main orchestrator
│   └── utils/              # Utilities
├── tests/                   # Test suites
├── examples/               # Example scripts
└── docs/                   # Documentation
```

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/agent_zero --cov-report=html
```

### Adding New Agents

1. Create your agent class inheriting from `BaseAgent`:
```python
from agent_zero.core.agent_base import BaseAgent

class MySpecializedAgent(BaseAgent):
    async def can_solve_problem(self, problem_request):
        # Return confidence score 0.0-1.0
        
    async def solve_problem(self, problem_request):
        # Implement your algorithm
        return ProblemResult(success=True, solution=result)
```

2. Register your agent:
```python
my_agent = MySpecializedAgent("my_agent")
await registry.register_agent(my_agent.get_registration_info())
```

## 📊 Monitoring & Analytics

### System Status
```bash
agent-zero status
```
Shows:
- Active agents and their capabilities
- System health and performance metrics
- Problem-solving statistics
- Supported problem types

### Performance Metrics
- Problems solved per minute
- Success/failure rates
- Average execution times
- Agent load distribution

## 🔗 Integration with PropFlow

Agent Zero seamlessly integrates with the [PropFlow](../Belief-Propagation-Simulator) belief propagation simulator:

- **Automatic Factor Graph Generation**: Converts problem descriptions to factor graphs
- **Multiple Algorithm Support**: Min-sum, max-sum with various policies
- **Performance Optimization**: Damping, splitting, and convergence policies
- **Result Translation**: Converts solutions back to natural language

## 🛣️ Roadmap

### Phase 1 (Current)
- ✅ Core agent framework
- ✅ PropFlow integration  
- ✅ Basic orchestration
- ✅ CLI interface

### Phase 2 (Next)
- [ ] Advanced problem classification using NLP
- [ ] Multi-agent collaboration 
- [ ] Performance optimization
- [ ] Web dashboard

### Phase 3 (Future)
- [ ] Additional specialized agents
- [ ] Learning from previous solutions
- [ ] External API integrations
- [ ] Distributed deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed technical documentation
- **Issues**: Report bugs and request features via GitHub Issues
- **Examples**: Check the `examples/` directory for usage patterns

## 🔗 Related Projects

- **[PropFlow](../Belief-Propagation-Simulator)**: Belief propagation simulator for DCOP problems
- **Agent Zero Extensions**: Additional specialized agents (coming soon)

---

*Agent Zero: Where specialized intelligence meets distributed problem-solving* 🤖✨