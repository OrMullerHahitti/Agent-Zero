"""Training data generation for Agent Zero SLMs."""

import json
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ..slm_interface import TaskType
from ..model_factory import ModelFactory, SLMConfig, ModelType


class AgentZeroDataGenerator:
    """Generate training data for Agent Zero specialized SLMs."""
    
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("data_generator")
        
        # Problem templates for different domains
        self.problem_templates = self._load_problem_templates()
        
        # Classification examples
        self.classification_examples = []
        self.explanation_examples = []
        
    def _load_problem_templates(self) -> Dict[str, List[Dict]]:
        """Load problem templates for different domains."""
        return {
            "dcop": [
                {
                    "template": "I have {num_vars} variables, each with {domain_size} possible values. I need to minimize cost while satisfying constraints.",
                    "params": {"num_vars": range(5, 20), "domain_size": range(3, 8)},
                    "category": "dcop",
                    "algorithm": "belief_propagation"
                },
                {
                    "template": "Solve a distributed constraint optimization problem with {num_vars} variables and {num_constraints} constraints to {objective} total cost.",
                    "params": {"num_vars": range(8, 25), "num_constraints": range(10, 50), "objective": ["minimize", "reduce"]},
                    "category": "dcop",
                    "algorithm": "min_sum"
                },
                {
                    "template": "I need to assign values to {num_vars} variables in a factor graph to achieve optimal solution using belief propagation.",
                    "params": {"num_vars": range(6, 15)},
                    "category": "dcop",
                    "algorithm": "belief_propagation"
                }
            ],
            "graph_algorithms": [
                {
                    "template": "Find the shortest path between node {start} and node {end} in a graph with {num_nodes} nodes.",
                    "params": {"start": range(1, 5), "end": range(8, 12), "num_nodes": range(10, 30)},
                    "category": "graph_algorithms",
                    "algorithm": "shortest_path"
                },
                {
                    "template": "Calculate the minimum spanning tree for a {graph_type} graph with {num_nodes} nodes and {num_edges} edges.",
                    "params": {"graph_type": ["weighted", "undirected"], "num_nodes": range(8, 20), "num_edges": range(15, 50)},
                    "category": "graph_algorithms", 
                    "algorithm": "mst"
                },
                {
                    "template": "Solve graph coloring problem for {num_nodes} vertices using minimum number of colors.",
                    "params": {"num_nodes": range(5, 15)},
                    "category": "graph_algorithms",
                    "algorithm": "graph_coloring"
                }
            ],
            "optimization": [
                {
                    "template": "Optimize {num_vars} variables to {objective} the objective function subject to {num_constraints} constraints.",
                    "params": {"num_vars": range(3, 10), "objective": ["minimize", "maximize"], "num_constraints": range(2, 8)},
                    "category": "optimization",
                    "algorithm": "linear_programming"
                },
                {
                    "template": "Use genetic algorithm to find optimal solution for {problem_type} with population size {pop_size}.",
                    "params": {"problem_type": ["traveling salesman", "knapsack", "scheduling"], "pop_size": range(50, 200)},
                    "category": "optimization",
                    "algorithm": "genetic_algorithm"
                }
            ],
            "reasoning": [
                {
                    "template": "If {premise1} and {premise2}, then what can we conclude about {conclusion_topic}?",
                    "params": {
                        "premise1": ["Alice is taller than Bob", "All birds can fly", "The weather is sunny"],
                        "premise2": ["Bob is taller than Charlie", "Penguins are birds", "It's not raining"],
                        "conclusion_topic": ["Charlie's height", "penguins", "outdoor activities"]
                    },
                    "category": "reasoning",
                    "algorithm": "logical_deduction"
                },
                {
                    "template": "Solve this logic puzzle: {puzzle_description}",
                    "params": {
                        "puzzle_description": [
                            "Three people have different professions and live in different cities",
                            "Five friends each like different colors and animals",
                            "Four students take different subjects and have different grades"
                        ]
                    },
                    "category": "reasoning",
                    "algorithm": "constraint_satisfaction"
                }
            ],
            "coding": [
                {
                    "template": "Write a {language} function to {task} using {algorithm} algorithm.",
                    "params": {
                        "language": ["Python", "Java", "C++"],
                        "task": ["sort an array", "search for element", "reverse a string", "find maximum"],
                        "algorithm": ["quicksort", "binary search", "recursion", "dynamic programming"]
                    },
                    "category": "coding",
                    "algorithm": "code_generation"
                },
                {
                    "template": "Implement {data_structure} in {language} with {operations} operations.",
                    "params": {
                        "data_structure": ["binary tree", "hash table", "linked list", "stack", "queue"],
                        "language": ["Python", "Java", "C++"],
                        "operations": ["insert/delete", "search/update", "push/pop", "enqueue/dequeue"]
                    },
                    "category": "coding",
                    "algorithm": "data_structures"
                }
            ]
        }
    
    async def generate_classification_dataset(
        self, 
        num_examples: int = 1000,
        use_external_llm: bool = True,
        external_llm_config: Optional[SLMConfig] = None
    ) -> str:
        """Generate classification training dataset."""
        
        self.logger.info(f"Generating {num_examples} classification examples")
        
        examples = []
        
        # Generate examples for each category
        categories = list(self.problem_templates.keys())
        examples_per_category = num_examples // len(categories)
        
        for category in categories:
            category_examples = await self._generate_category_examples(
                category, examples_per_category, "classification",
                use_external_llm, external_llm_config
            )
            examples.extend(category_examples)
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Save dataset
        output_path = self.output_dir / "classification_training.json"
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        self.logger.info(f"Classification dataset saved to {output_path}")
        return str(output_path)
    
    async def generate_explanation_dataset(
        self,
        num_examples: int = 800,
        use_external_llm: bool = True,
        external_llm_config: Optional[SLMConfig] = None
    ) -> str:
        """Generate explanation training dataset."""
        
        self.logger.info(f"Generating {num_examples} explanation examples")
        
        examples = []
        
        # Generate examples for each category
        categories = list(self.problem_templates.keys())
        examples_per_category = num_examples // len(categories)
        
        for category in categories:
            category_examples = await self._generate_category_examples(
                category, examples_per_category, "explanation",
                use_external_llm, external_llm_config
            )
            examples.extend(category_examples)
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Save dataset
        output_path = self.output_dir / "explanation_training.json"
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        self.logger.info(f"Explanation dataset saved to {output_path}")
        return str(output_path)
    
    async def _generate_category_examples(
        self,
        category: str,
        num_examples: int,
        task_type: str,
        use_external_llm: bool,
        external_llm_config: Optional[SLMConfig]
    ) -> List[Dict[str, Any]]:
        """Generate examples for a specific category."""
        
        templates = self.problem_templates[category]
        examples = []
        
        for i in range(num_examples):
            # Select random template
            template_data = random.choice(templates)
            template = template_data["template"]
            params = template_data["params"]
            
            # Generate parameter values
            param_values = {}
            for param_name, param_options in params.items():
                if isinstance(param_options, range):
                    param_values[param_name] = random.choice(list(param_options))
                elif isinstance(param_options, list):
                    param_values[param_name] = random.choice(param_options)
                else:
                    param_values[param_name] = param_options
            
            # Fill template
            problem_description = template.format(**param_values)
            
            if task_type == "classification":
                example = await self._create_classification_example(
                    problem_description, template_data, param_values,
                    use_external_llm, external_llm_config
                )
            elif task_type == "explanation":
                example = await self._create_explanation_example(
                    problem_description, template_data, param_values,
                    use_external_llm, external_llm_config
                )
            
            if example:
                examples.append(example)
        
        return examples
    
    async def _create_classification_example(
        self,
        problem_description: str,
        template_data: Dict,
        param_values: Dict,
        use_external_llm: bool,
        external_llm_config: Optional[SLMConfig]
    ) -> Optional[Dict[str, Any]]:
        """Create a classification training example."""
        
        # Create ground truth classification
        classification = {
            "category": template_data["category"],
            "confidence": round(random.uniform(0.85, 0.99), 2),
            "extracted_parameters": {
                "key_concepts": self._extract_concepts(problem_description, template_data["category"]),
                "numeric_values": [v for v in param_values.values() if isinstance(v, int)],
                "constraints": self._generate_constraints(template_data["category"]),
                "objective": self._generate_objective(template_data["category"])
            },
            "reasoning": f"Problem mentions {template_data['category']} concepts and {template_data['algorithm']} methodology"
        }
        
        # Optionally enhance with external LLM
        if use_external_llm and external_llm_config:
            try:
                enhanced_classification = await self._enhance_with_llm(
                    problem_description, classification, "classification", external_llm_config
                )
                if enhanced_classification:
                    classification = enhanced_classification
            except Exception as e:
                self.logger.warning(f"Failed to enhance with external LLM: {e}")
        
        return {
            "input": problem_description,
            "output": json.dumps(classification, indent=2),
            "metadata": {
                "category": template_data["category"],
                "algorithm": template_data["algorithm"],
                "generated_at": datetime.now().isoformat()
            }
        }
    
    async def _create_explanation_example(
        self,
        problem_description: str,
        template_data: Dict,
        param_values: Dict,
        use_external_llm: bool,
        external_llm_config: Optional[SLMConfig]
    ) -> Optional[Dict[str, Any]]:
        """Create an explanation training example."""
        
        # Create mock solution data
        solution_data = self._generate_mock_solution(template_data["category"], param_values)
        
        # Create explanation
        explanation = self._generate_base_explanation(
            problem_description, solution_data, template_data
        )
        
        # Optionally enhance with external LLM
        if use_external_llm and external_llm_config:
            try:
                enhanced_explanation = await self._enhance_with_llm(
                    f"Original Problem: {problem_description}\nSolution: {solution_data}",
                    explanation, "explanation", external_llm_config
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.warning(f"Failed to enhance explanation with external LLM: {e}")
        
        # Format as input-output pair
        input_text = f"""Explain this technical solution in clear, natural language.

Original Problem: {problem_description}
Solution Data: {json.dumps(solution_data, indent=2)}
Algorithm Used: {template_data['algorithm']}

Provide:
1. What the problem was asking for
2. How the algorithm solved it
3. What the final answer means
4. Why this solution is good/optimal"""
        
        return {
            "input": input_text,
            "output": explanation,
            "metadata": {
                "category": template_data["category"],
                "algorithm": template_data["algorithm"],
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _extract_concepts(self, problem_description: str, category: str) -> List[str]:
        """Extract key concepts from problem description."""
        concept_maps = {
            "dcop": ["constraint", "optimization", "variables", "assignment", "cost", "factor graph"],
            "graph_algorithms": ["graph", "nodes", "edges", "path", "tree", "connectivity"],
            "optimization": ["minimize", "maximize", "objective", "constraints", "variables"],
            "reasoning": ["logic", "premise", "conclusion", "deduction", "inference"],
            "coding": ["algorithm", "function", "implementation", "data structure"]
        }
        
        concepts = []
        problem_lower = problem_description.lower()
        
        for concept in concept_maps.get(category, []):
            if concept in problem_lower:
                concepts.append(concept)
        
        return concepts
    
    def _generate_constraints(self, category: str) -> List[str]:
        """Generate realistic constraints for category."""
        constraint_templates = {
            "dcop": ["satisfy all constraints", "minimize total cost", "find feasible assignment"],
            "graph_algorithms": ["connected graph", "positive weights", "acyclic path"],
            "optimization": ["linear constraints", "non-negative variables", "bounded domain"],
            "reasoning": ["logical consistency", "sound inference", "complete reasoning"],
            "coding": ["efficient implementation", "correct algorithm", "clean code"]
        }
        
        return random.sample(constraint_templates.get(category, []), k=min(2, len(constraint_templates.get(category, []))))
    
    def _generate_objective(self, category: str) -> str:
        """Generate objective for category."""
        objectives = {
            "dcop": "find optimal variable assignment",
            "graph_algorithms": "solve graph problem efficiently",
            "optimization": "optimize objective function",
            "reasoning": "derive logical conclusions",
            "coding": "implement correct algorithm"
        }
        
        return objectives.get(category, "solve the problem")
    
    def _generate_mock_solution(self, category: str, param_values: Dict) -> Dict[str, Any]:
        """Generate realistic mock solution data."""
        
        if category == "dcop":
            num_vars = param_values.get("num_vars", 10)
            return {
                "assignments": {f"x{i}": random.randint(0, 4) for i in range(num_vars)},
                "total_cost": round(random.uniform(20, 100), 2),
                "iterations": random.randint(50, 200),
                "converged": True,
                "algorithm_used": "min_sum"
            }
        
        elif category == "graph_algorithms":
            return {
                "path": [1, 3, 7, 9],
                "path_length": round(random.uniform(10, 50), 2),
                "nodes_visited": random.randint(8, 20),
                "algorithm_used": "dijkstra"
            }
        
        elif category == "optimization":
            return {
                "optimal_values": [round(random.uniform(0, 10), 2) for _ in range(3)],
                "objective_value": round(random.uniform(50, 200), 2),
                "constraints_satisfied": random.randint(3, 8),
                "algorithm_used": "simplex"
            }
        
        elif category == "reasoning":
            return {
                "conclusion": "Charlie is the shortest person",
                "reasoning_steps": [
                    "Alice is taller than Bob",
                    "Bob is taller than Charlie", 
                    "Therefore, Alice > Bob > Charlie"
                ],
                "logical_validity": True
            }
        
        elif category == "coding":
            return {
                "code": "def quicksort(arr): ...",
                "time_complexity": "O(n log n)",
                "space_complexity": "O(log n)",
                "test_cases_passed": 15
            }
        
        return {}
    
    def _generate_base_explanation(
        self, 
        problem_description: str,
        solution_data: Dict,
        template_data: Dict
    ) -> str:
        """Generate base explanation for solution."""
        
        category = template_data["category"]
        algorithm = template_data["algorithm"]
        
        if category == "dcop":
            return f"""The problem was asking to find the best assignment of values to variables while minimizing cost and satisfying constraints.

The {algorithm} algorithm solved this by:
1. Creating a factor graph representing the relationships between variables
2. Passing messages between variables and factors iteratively
3. Converging to an optimal solution after {solution_data.get('iterations', 'many')} iterations

The final solution assigns each variable a value that minimizes the total cost to {solution_data.get('total_cost', 'X')}. This represents an optimal assignment because the algorithm converged, meaning no better solution exists given the constraints.

This solution is good because it satisfies all the problem constraints while achieving the lowest possible cost through the systematic message-passing approach."""
        
        elif category == "graph_algorithms":
            return f"""This was a graph algorithm problem asking to find the optimal path/structure in a network.

The {algorithm} algorithm solved this by:
1. Systematically exploring the graph structure
2. Maintaining optimal distances/costs during traversal
3. Finding the best path with length {solution_data.get('path_length', 'X')}

The result shows the optimal path: {solution_data.get('path', [])}. This path is guaranteed to be optimal because the algorithm explores all possibilities and keeps track of the best option at each step.

This solution is efficient and correct, visiting {solution_data.get('nodes_visited', 'several')} nodes to find the optimal result."""
        
        elif category == "optimization":
            return f"""This was an optimization problem asking to find the best values for variables to optimize an objective function.

The {algorithm} method solved this by:
1. Setting up the mathematical constraints and objective function
2. Systematically searching the feasible solution space
3. Finding optimal values: {solution_data.get('optimal_values', [])}

The final objective value of {solution_data.get('objective_value', 'X')} represents the best possible outcome while satisfying all {solution_data.get('constraints_satisfied', 'X')} constraints.

This solution is optimal because the algorithm guarantees that no better solution exists within the constraint boundaries."""
        
        elif category == "reasoning":
            return f"""This was a logical reasoning problem requiring step-by-step deduction.

The reasoning process worked by:
1. Analyzing the given premises
2. Applying logical rules step by step: {solution_data.get('reasoning_steps', [])}
3. Reaching the conclusion: {solution_data.get('conclusion', 'X')}

This conclusion is logically valid because each step follows from the previous ones using sound logical principles. The reasoning chain ensures that if the premises are true, the conclusion must also be true."""
        
        elif category == "coding":
            return f"""This was a programming task requiring algorithm implementation.

The solution approach:
1. Implemented the {algorithm} algorithm efficiently
2. Achieved {solution_data.get('time_complexity', 'optimal')} time complexity
3. Used {solution_data.get('space_complexity', 'minimal')} space

The code passes all {solution_data.get('test_cases_passed', 'X')} test cases, demonstrating correctness. The algorithm choice is appropriate because it provides the right balance of efficiency and simplicity for this problem type."""
        
        return f"The {algorithm} algorithm successfully solved this {category} problem with good results."
    
    async def _enhance_with_llm(
        self,
        context: str,
        base_content: Any,
        enhancement_type: str,
        llm_config: SLMConfig
    ) -> Optional[Any]:
        """Enhance content using external LLM."""
        
        try:
            # Create LLM client
            client = ModelFactory.create(llm_config)
            await client.load_model()
            
            if enhancement_type == "classification":
                prompt = f"""Improve this problem classification by making it more accurate and detailed.

Problem: {context}
Current Classification: {json.dumps(base_content, indent=2)}

Return only the improved JSON classification with the same structure but more accurate extracted parameters and better reasoning."""
                
                response = await client.generate(prompt)
                if response.success:
                    try:
                        return json.loads(response.text)
                    except:
                        return None
            
            elif enhancement_type == "explanation":
                prompt = f"""Improve this technical explanation to make it clearer and more educational.

{context}

Current Explanation: {base_content}

Provide an improved explanation that is clearer, more detailed, and more educational while maintaining accuracy."""
                
                response = await client.generate(prompt)
                if response.success:
                    return response.text
            
            await client.unload_model()
            
        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {e}")
        
        return None
    
    def create_evaluation_split(
        self, 
        training_file: str, 
        eval_ratio: float = 0.2
    ) -> Tuple[str, str]:
        """Split training data into train/eval sets."""
        
        with open(training_file, 'r') as f:
            data = json.load(f)
        
        # Shuffle data
        random.shuffle(data)
        
        # Split
        split_idx = int(len(data) * (1 - eval_ratio))
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        # Save splits
        train_path = training_file.replace(".json", "_train.json")
        eval_path = training_file.replace(".json", "_eval.json")
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        self.logger.info(f"Created train split: {len(train_data)} examples -> {train_path}")
        self.logger.info(f"Created eval split: {len(eval_data)} examples -> {eval_path}")
        
        return train_path, eval_path
    
    def get_dataset_stats(self, dataset_path: str) -> Dict[str, Any]:
        """Get statistics about a dataset."""
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        categories = {}
        algorithms = {}
        
        for item in data:
            metadata = item.get("metadata", {})
            category = metadata.get("category", "unknown")
            algorithm = metadata.get("algorithm", "unknown")
            
            categories[category] = categories.get(category, 0) + 1
            algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
        
        return {
            "total_examples": len(data),
            "categories": categories,
            "algorithms": algorithms,
            "average_input_length": sum(len(item["input"]) for item in data) / len(data),
            "average_output_length": sum(len(item["output"]) for item in data) / len(data)
        }