"""
Complete demo of Agent Zero SLM training and deployment.
Shows how to train your own models locally and use them in the Agent Zero system.
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("slm_demo")


async def demo_quick_training():
    """Demo: Quick training with minimal setup."""
    
    logger.info("=== Demo: Quick SLM Training ===")
    
    from src.agent_zero.models.training.training_pipeline import quick_train_classification_model
    
    try:
        # Train a small classification model quickly
        logger.info("Training classification model with 100 examples...")
        
        model_path = await quick_train_classification_model(
            output_dir="./demo_training",
            num_examples=100  # Small for demo
        )
        
        logger.info(f"✅ Model trained successfully: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None


async def demo_full_pipeline():
    """Demo: Full training pipeline with both models."""
    
    logger.info("=== Demo: Full Training Pipeline ===")
    
    from src.agent_zero.models.training.training_pipeline import TrainingPipeline
    from src.agent_zero.models.slm_interface import SLMConfig, ModelType
    
    try:
        # Create pipeline
        pipeline = TrainingPipeline("./full_demo_training")
        
        # Optional: Use external LLM to enhance training data
        # external_llm_config = SLMConfig(
        #     model_type=ModelType.REMOTE,
        #     model_name="gpt-4o-mini",
        #     api_key="your-api-key"
        # )
        
        # Train both models
        logger.info("Training both classification and explanation models...")
        
        results = await pipeline.train_both_models(
            num_examples_each=200,  # Small for demo
            use_external_llm=False,  # Set to True if you have API key
            # external_llm_config=external_llm_config
        )
        
        logger.info("✅ Full pipeline completed:")
        for model_type, path in results.items():
            if path:
                logger.info(f"  {model_type}: {path}")
            else:
                logger.info(f"  {model_type}: FAILED")
        
        # Test the models
        if results["classification"] and results["explanation"]:
            logger.info("Testing trained models...")
            test_results = await pipeline.test_trained_models(
                results["classification"],
                results["explanation"]
            )
            
            logger.info("Test results:")
            for model_type, result in test_results.items():
                if "error" in result:
                    logger.error(f"  {model_type}: {result['error']}")
                else:
                    logger.info(f"  {model_type}: Success - {result['execution_time']:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Full pipeline failed: {e}")
        return None


async def demo_model_usage():
    """Demo: Using trained models in Agent Zero."""
    
    logger.info("=== Demo: Using Trained Models ===")
    
    from src.agent_zero.models import ModelFactory, SLMConfig, ModelType
    from src.agent_zero.models.slm_interface import TaskType
    
    try:
        # This assumes you have trained models from previous demos
        model_dir = Path("./demo_training") / "classification_agent-zero-classifier" / "final_model"
        
        if not model_dir.exists():
            logger.warning("No trained model found. Run training demo first.")
            return
        
        # Create model config
        config = SLMConfig(
            model_type=ModelType.LOCAL,
            model_name="demo-classifier",
            model_path=str(model_dir),
            max_tokens=512,
            temperature=0.3
        )
        
        # Create client
        client = ModelFactory.create(config)
        await client.load_model()
        
        # Test classification
        test_problems = [
            "I have 10 variables and need to find optimal assignments to minimize cost",
            "Find the shortest path between nodes in a weighted graph",
            "Write a Python function to sort an array using quicksort"
        ]
        
        logger.info("Testing problem classification:")
        for problem in test_problems:
            result = await client.classify_problem(problem)
            if result.success:
                logger.info(f"✅ '{problem[:40]}...' -> {result.text[:100]}...")
            else:
                logger.error(f"❌ Classification failed: {result.error_message}")
        
        await client.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Model usage demo failed: {e}")


async def demo_hybrid_system_integration():
    """Demo: Integrating trained models with Agent Zero hybrid system."""
    
    logger.info("=== Demo: Hybrid System Integration ===")
    
    from src.agent_zero.models import ModelFactory, SLMConfig, ModelType
    from src.agent_zero.agents.input_classifier_agent import InputClassifierAgent
    from src.agent_zero.agents.output_formatter_agent import OutputFormatterAgent
    
    try:
        # Check if we have trained models
        classification_model = Path("./demo_training") / "classification_agent-zero-classifier" / "final_model"
        
        if not classification_model.exists():
            logger.warning("No trained classification model found. Using mock client.")
            
            # Mock SLM client for demo
            class MockSLMClient:
                async def generate(self, prompt):
                    if "classification" in prompt.lower():
                        return '{"category": "dcop", "confidence": 0.9, "extracted_parameters": {"key_concepts": ["constraint", "optimization"]}}'
                    else:
                        return "The algorithm successfully solved the problem using systematic optimization."
            
            llm_client = MockSLMClient()
            slm_client = MockSLMClient()
        else:
            # Use actual trained models
            classification_config = SLMConfig(
                model_type=ModelType.LOCAL,
                model_name="agent-zero-classifier",
                model_path=str(classification_model)
            )
            
            llm_client = ModelFactory.create(classification_config)
            slm_client = ModelFactory.create(classification_config)  # Same for demo
        
        # Create AI-powered agents
        input_classifier = InputClassifierAgent(llm_client=llm_client)
        output_formatter = OutputFormatterAgent(slm_client=slm_client)
        
        await input_classifier.start()
        await output_formatter.start()
        
        # Test the integration
        from src.agent_zero.core.message import ProblemRequest
        
        test_problem = "I need to optimize 8 variables with constraints to minimize total cost"
        
        # Step 1: Classify problem
        classification_request = ProblemRequest(
            problem_description=test_problem,
            problem_type="input_classification"
        )
        
        classification_result = await input_classifier.solve_problem(classification_request)
        
        if classification_result.success:
            logger.info("✅ Problem classification successful")
            logger.info(f"   Category: {classification_result.solution.get('classification', {}).get('category', 'unknown')}")
        else:
            logger.error(f"❌ Classification failed: {classification_result.error_message}")
        
        # Step 2: Format solution (mock)
        mock_solution = {
            "assignments": {"x1": 2, "x2": 1, "x3": 0},
            "total_cost": 47.5,
            "converged": True
        }
        
        formatting_request = ProblemRequest(
            problem_description="Format this solution",
            problem_type="format_solution",
            constraints={
                "solution_data": mock_solution,
                "original_problem": test_problem,
                "agent_type": "dcop",
                "algorithm": "belief_propagation"
            }
        )
        
        formatting_result = await output_formatter.solve_problem(formatting_request)
        
        if formatting_result.success:
            logger.info("✅ Solution formatting successful")
            logger.info(f"   Explanation: {formatting_result.solution.get('formatted_explanation', '')[:100]}...")
        else:
            logger.error(f"❌ Formatting failed: {formatting_result.error_message}")
        
        await input_classifier.stop()
        await output_formatter.stop()
        
    except Exception as e:
        logger.error(f"❌ Integration demo failed: {e}")


async def demo_remote_vs_local():
    """Demo: Comparing remote API vs local trained models."""
    
    logger.info("=== Demo: Remote vs Local Models ===")
    
    from src.agent_zero.models import ModelFactory, SLMConfig, ModelType
    import time
    
    test_prompt = "Classify this problem: I have 10 variables and need to minimize cost with constraints."
    
    # Test local model (if available)
    local_model_path = Path("./demo_training") / "classification_agent-zero-classifier" / "final_model"
    
    if local_model_path.exists():
        try:
            logger.info("Testing local trained model...")
            
            local_config = SLMConfig(
                model_type=ModelType.LOCAL,
                model_name="local-classifier",
                model_path=str(local_model_path),
                max_tokens=256
            )
            
            local_client = ModelFactory.create(local_config)
            await local_client.load_model()
            
            start_time = time.time()
            local_result = await local_client.generate(test_prompt)
            local_time = time.time() - start_time
            
            if local_result.success:
                logger.info(f"✅ Local model: {local_time:.2f}s, {local_result.tokens_per_second:.1f} tokens/sec")
            else:
                logger.error(f"❌ Local model failed: {local_result.error_message}")
            
            await local_client.unload_model()
            
        except Exception as e:
            logger.error(f"❌ Local model test failed: {e}")
    
    else:
        logger.info("Local model not available (train one first)")
    
    # Test remote model (if API key available)
    try:
        logger.info("Testing remote API model...")
        
        # Note: You would need to set your API key in environment
        remote_config = SLMConfig(
            model_type=ModelType.REMOTE,
            model_name="gpt-4o-mini",
            api_key="your-api-key-here",  # Replace with real key
            max_tokens=256
        )
        
        remote_client = ModelFactory.create(remote_config)
        await remote_client.load_model()
        
        start_time = time.time()
        remote_result = await remote_client.generate(test_prompt)
        remote_time = time.time() - start_time
        
        if remote_result.success:
            logger.info(f"✅ Remote model: {remote_time:.2f}s, cost: ${remote_result.cost:.4f}")
        else:
            logger.error(f"❌ Remote model failed: {remote_result.error_message}")
        
        await remote_client.unload_model()
        
    except Exception as e:
        logger.info(f"Remote model test skipped (no API key): {e}")


async def main():
    """Run all demos."""
    
    logger.info("Starting Agent Zero SLM Training & Usage Demo")
    logger.info("=" * 60)
    
    # Demo 1: Quick training
    # model_path = await demo_quick_training()
    
    # Demo 2: Full pipeline (comment out if you don't want to train)
    # results = await demo_full_pipeline()
    
    # Demo 3: Using trained models
    await demo_model_usage()
    
    # Demo 4: Hybrid system integration
    await demo_hybrid_system_integration()
    
    # Demo 5: Compare local vs remote
    await demo_remote_vs_local()
    
    logger.info("=" * 60)
    logger.info("Demo completed! Check the output directories for trained models.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Train your own models with more data")
    logger.info("2. Set up API keys for remote models")
    logger.info("3. Integrate with the full Agent Zero system")
    logger.info("4. Deploy models for production use")


if __name__ == "__main__":
    asyncio.run(main())