"""Complete training pipeline for Agent Zero SLMs."""

import asyncio
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from .data_generator import AgentZeroDataGenerator
from ..local.local_trainer import LocalSLMTrainer, TrainingConfig
from ..local.model_manager import LocalModelManager
from ..slm_interface import TaskType, SLMConfig, ModelType
from ..model_factory import ModelFactory


class TrainingPipeline:
    """Complete pipeline for training Agent Zero SLMs."""
    
    def __init__(self, output_dir: str = "./agent_zero_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("training_pipeline")
        
        # Components
        self.data_generator = AgentZeroDataGenerator(str(self.output_dir / "data"))
        self.model_manager = LocalModelManager(str(self.output_dir / "models"))
        
        # Training results
        self.training_results = {}
        
    async def train_classification_model(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        num_training_examples: int = 1000,
        use_external_llm: bool = False,
        external_llm_config: Optional[SLMConfig] = None,
        training_config_overrides: Optional[Dict] = None
    ) -> str:
        """Train a classification model for Agent Zero."""
        
        self.logger.info("Starting classification model training pipeline")
        
        # Step 1: Generate training data
        self.logger.info("Generating classification training data...")
        training_data_path = await self.data_generator.generate_classification_dataset(
            num_examples=num_training_examples,
            use_external_llm=use_external_llm,
            external_llm_config=external_llm_config
        )
        
        # Step 2: Create train/eval split
        train_path, eval_path = self.data_generator.create_evaluation_split(training_data_path)
        
        # Step 3: Download base model if needed
        self.logger.info(f"Ensuring base model {base_model} is available...")
        try:
            local_model_path = await self.model_manager.download_model(base_model)
        except Exception as e:
            self.logger.error(f"Failed to download base model: {e}")
            raise
        
        # Step 4: Setup training configuration
        training_config = LocalSLMTrainer.create_training_config(
            task_type="classification",
            base_model=local_model_path,
            train_data_path=train_path,
            eval_data_path=eval_path,
            **(training_config_overrides or {})
        )
        
        # Step 5: Train model
        self.logger.info("Starting model training...")
        trainer = LocalSLMTrainer(training_config)
        await trainer.prepare_model()
        trained_model_path = await trainer.train()
        
        # Step 6: Evaluate model
        self.logger.info("Evaluating trained model...")
        eval_results = trainer.evaluate(trained_model_path, eval_path)
        
        # Step 7: Save results
        results = {
            "model_type": "classification",
            "base_model": base_model,
            "trained_model_path": trained_model_path,
            "training_data_path": training_data_path,
            "training_examples": num_training_examples,
            "training_config": training_config.__dict__,
            "evaluation_results": eval_results,
            "trained_at": datetime.now().isoformat()
        }
        
        results_path = self.output_dir / "classification_model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.training_results["classification"] = results
        
        self.logger.info(f"Classification model training completed: {trained_model_path}")
        return trained_model_path
    
    async def train_explanation_model(
        self,
        base_model: str = "codellama/CodeLlama-7b-Instruct-hf",
        num_training_examples: int = 800,
        use_external_llm: bool = False,
        external_llm_config: Optional[SLMConfig] = None,
        training_config_overrides: Optional[Dict] = None
    ) -> str:
        """Train an explanation model for Agent Zero."""
        
        self.logger.info("Starting explanation model training pipeline")
        
        # Step 1: Generate training data
        self.logger.info("Generating explanation training data...")
        training_data_path = await self.data_generator.generate_explanation_dataset(
            num_examples=num_training_examples,
            use_external_llm=use_external_llm,
            external_llm_config=external_llm_config
        )
        
        # Step 2: Create train/eval split
        train_path, eval_path = self.data_generator.create_evaluation_split(training_data_path)
        
        # Step 3: Download base model if needed
        self.logger.info(f"Ensuring base model {base_model} is available...")
        try:
            local_model_path = await self.model_manager.download_model(base_model)
        except Exception as e:
            self.logger.error(f"Failed to download base model: {e}")
            raise
        
        # Step 4: Setup training configuration
        training_config = LocalSLMTrainer.create_training_config(
            task_type="explanation",
            base_model=local_model_path,
            train_data_path=train_path,
            eval_data_path=eval_path,
            **(training_config_overrides or {})
        )
        
        # Step 5: Train model
        self.logger.info("Starting model training...")
        trainer = LocalSLMTrainer(training_config)
        await trainer.prepare_model()
        trained_model_path = await trainer.train()
        
        # Step 6: Evaluate model
        self.logger.info("Evaluating trained model...")
        eval_results = trainer.evaluate(trained_model_path, eval_path)
        
        # Step 7: Save results
        results = {
            "model_type": "explanation",
            "base_model": base_model,
            "trained_model_path": trained_model_path,
            "training_data_path": training_data_path,
            "training_examples": num_training_examples,
            "training_config": training_config.__dict__,
            "evaluation_results": eval_results,
            "trained_at": datetime.now().isoformat()
        }
        
        results_path = self.output_dir / "explanation_model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.training_results["explanation"] = results
        
        self.logger.info(f"Explanation model training completed: {trained_model_path}")
        return trained_model_path
    
    async def train_both_models(
        self,
        classification_base: str = "meta-llama/Llama-3.1-8B-Instruct",
        explanation_base: str = "codellama/CodeLlama-7b-Instruct-hf",
        num_examples_each: int = 1000,
        use_external_llm: bool = False,
        external_llm_config: Optional[SLMConfig] = None
    ) -> Dict[str, str]:
        """Train both classification and explanation models."""
        
        self.logger.info("Starting complete Agent Zero SLM training pipeline")
        
        results = {}
        
        # Train classification model
        try:
            classification_path = await self.train_classification_model(
                base_model=classification_base,
                num_training_examples=num_examples_each,
                use_external_llm=use_external_llm,
                external_llm_config=external_llm_config
            )
            results["classification"] = classification_path
        except Exception as e:
            self.logger.error(f"Classification model training failed: {e}")
            results["classification"] = None
        
        # Train explanation model
        try:
            explanation_path = await self.train_explanation_model(
                base_model=explanation_base,
                num_training_examples=num_examples_each,
                use_external_llm=use_external_llm,
                external_llm_config=external_llm_config
            )
            results["explanation"] = explanation_path
        except Exception as e:
            self.logger.error(f"Explanation model training failed: {e}")
            results["explanation"] = None
        
        # Save combined results
        combined_results = {
            "pipeline_completed_at": datetime.now().isoformat(),
            "models_trained": results,
            "training_results": self.training_results
        }
        
        pipeline_results_path = self.output_dir / "pipeline_results.json"
        with open(pipeline_results_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        self.logger.info("Complete training pipeline finished")
        return results
    
    async def test_trained_models(
        self,
        classification_model_path: str,
        explanation_model_path: str
    ) -> Dict[str, Any]:
        """Test trained models with sample problems."""
        
        self.logger.info("Testing trained models...")
        
        test_results = {}
        
        # Test classification model
        try:
            classification_config = SLMConfig(
                model_type=ModelType.LOCAL,
                model_name="agent-zero-classifier",
                model_path=classification_model_path
            )
            
            classifier = ModelFactory.create(classification_config)
            await classifier.load_model()
            
            test_problem = "I have 12 variables with 5 possible values each. I need to minimize cost while satisfying all constraints."
            classification_result = await classifier.classify_problem(test_problem)
            
            test_results["classification"] = {
                "success": classification_result.success,
                "response": classification_result.text,
                "execution_time": classification_result.execution_time
            }
            
            await classifier.unload_model()
            
        except Exception as e:
            test_results["classification"] = {"error": str(e)}
        
        # Test explanation model
        try:
            explanation_config = SLMConfig(
                model_type=ModelType.LOCAL,
                model_name="agent-zero-explainer",
                model_path=explanation_model_path
            )
            
            explainer = ModelFactory.create(explanation_config)
            await explainer.load_model()
            
            test_solution = {
                "assignments": {"x1": 2, "x2": 1, "x3": 3},
                "total_cost": 45.2,
                "converged": True
            }
            
            explanation_result = await explainer.explain_solution(
                test_solution,
                "Minimize cost for 3 variables",
                "belief_propagation"
            )
            
            test_results["explanation"] = {
                "success": explanation_result.success,
                "response": explanation_result.text,
                "execution_time": explanation_result.execution_time
            }
            
            await explainer.unload_model()
            
        except Exception as e:
            test_results["explanation"] = {"error": str(e)}
        
        # Save test results
        test_results_path = self.output_dir / "model_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.logger.info("Model testing completed")
        return test_results
    
    def create_deployment_package(
        self,
        classification_model_path: str,
        explanation_model_path: str,
        package_name: str = "agent_zero_models"
    ) -> str:
        """Create a deployment package with trained models."""
        
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Copy models
        import shutil
        
        classification_dest = package_dir / "classification_model"
        explanation_dest = package_dir / "explanation_model"
        
        shutil.copytree(classification_model_path, classification_dest, dirs_exist_ok=True)
        shutil.copytree(explanation_model_path, explanation_dest, dirs_exist_ok=True)
        
        # Create deployment config
        deployment_config = {
            "package_name": package_name,
            "created_at": datetime.now().isoformat(),
            "models": {
                "classification": {
                    "path": "./classification_model",
                    "task_type": "classification",
                    "description": "Agent Zero problem classification model"
                },
                "explanation": {
                    "path": "./explanation_model", 
                    "task_type": "explanation",
                    "description": "Agent Zero solution explanation model"
                }
            },
            "usage": {
                "classification": "Use for classifying natural language problems into Agent Zero categories",
                "explanation": "Use for converting technical solutions into natural language explanations"
            }
        }
        
        config_path = package_dir / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        # Create README
        readme_content = f"""# Agent Zero SLM Models - {package_name}

This package contains trained Small Language Models for Agent Zero.

## Models Included

- **Classification Model**: Classifies natural language problems into appropriate categories
- **Explanation Model**: Converts technical solutions into clear natural language explanations

## Usage

```python
from agent_zero.models import ModelFactory, SLMConfig, ModelType

# Load classification model
classifier_config = SLMConfig(
    model_type=ModelType.LOCAL,
    model_name="agent-zero-classifier",
    model_path="./classification_model"
)
classifier = ModelFactory.create(classifier_config)

# Load explanation model  
explainer_config = SLMConfig(
    model_type=ModelType.LOCAL,
    model_name="agent-zero-explainer", 
    model_path="./explanation_model"
)
explainer = ModelFactory.create(explainer_config)
```

## Requirements

- transformers >= 4.30.0
- torch >= 2.0.0
- peft >= 0.4.0

Generated on: {datetime.now().isoformat()}
"""
        
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Deployment package created: {package_dir}")
        return str(package_dir)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training activities."""
        
        return {
            "output_directory": str(self.output_dir),
            "training_results": self.training_results,
            "available_models": self.model_manager.list_models(),
            "disk_usage": self.model_manager.check_disk_space(),
            "summary_generated_at": datetime.now().isoformat()
        }


# Convenience functions for quick training
async def quick_train_classification_model(
    output_dir: str = "./quick_training",
    num_examples: int = 500
) -> str:
    """Quickly train a classification model with default settings."""
    
    pipeline = TrainingPipeline(output_dir)
    return await pipeline.train_classification_model(
        num_training_examples=num_examples
    )


async def quick_train_explanation_model(
    output_dir: str = "./quick_training",
    num_examples: int = 400
) -> str:
    """Quickly train an explanation model with default settings."""
    
    pipeline = TrainingPipeline(output_dir)
    return await pipeline.train_explanation_model(
        num_training_examples=num_examples
    )


async def full_training_pipeline(
    output_dir: str = "./agent_zero_training",
    num_examples_each: int = 1000
) -> Dict[str, str]:
    """Run complete training pipeline for both models."""
    
    pipeline = TrainingPipeline(output_dir)
    return await pipeline.train_both_models(
        num_examples_each=num_examples_each
    )