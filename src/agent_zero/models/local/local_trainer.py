"""Local SLM training infrastructure using LoRA/QLoRA."""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

try:
    import torch
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset
    import wandb
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for local SLM training."""
    
    # Model settings
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_max_length: int = 2048
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training settings
    output_dir: str = "./trained_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Optimization
    use_4bit: bool = True
    use_nested_quant: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Logging and evaluation
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Data settings
    train_data_path: str = "./training_data/train.json"
    eval_data_path: str = "./training_data/eval.json"
    
    # Experiment tracking
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default LoRA target modules for Llama
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LocalSLMTrainer:
    """Trainer for fine-tuning SLMs locally using LoRA."""
    
    def __init__(self, config: TrainingConfig):
        if not TRAINING_AVAILABLE:
            raise ImportError(
                "Training dependencies not available. Install with: "
                "pip install transformers datasets peft accelerate bitsandbytes wandb"
            )
        
        self.config = config
        self.logger = logging.getLogger("local_trainer")
        
        # Training components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
    
    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.output_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
    
    async def prepare_model(self) -> None:
        """Load and prepare the base model for training."""
        self.logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        quantization_config = None
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare model for training
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        self.logger.info("Model preparation completed")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training."""
        self.logger.info(f"Loading dataset from: {data_path}")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        if isinstance(data, list) and len(data) > 0:
            if "conversations" in data[0]:
                # Multi-turn conversation format
                formatted_data = self._format_conversation_data(data)
            elif "input" in data[0] and "output" in data[0]:
                # Simple input-output format
                formatted_data = self._format_simple_data(data)
            else:
                raise ValueError("Unknown data format")
        else:
            raise ValueError("Empty or invalid dataset")
        
        # Tokenize
        def tokenize_function(examples):
            # Tokenize with truncation and padding
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.model_max_length,
                return_tensors=None
            )
            
            # Labels are the same as input_ids for causal LM
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        self.logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def _format_conversation_data(self, data: List[Dict]) -> List[Dict]:
        """Format conversation data for training."""
        formatted = []
        
        for item in data:
            conversations = item["conversations"]
            text = ""
            
            for turn in conversations:
                role = turn.get("from", "user")
                content = turn.get("value", "")
                
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"
            
            text += "<|endoftext|>"
            formatted.append({"text": text})
        
        return formatted
    
    def _format_simple_data(self, data: List[Dict]) -> List[Dict]:
        """Format simple input-output data for training."""
        formatted = []
        
        for item in data:
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            system_prompt = item.get("system", "")
            
            if system_prompt:
                text = f"<|system|>\n{system_prompt}\n<|user|>\n{input_text}\n<|assistant|>\n{output_text}\n<|endoftext|>"
            else:
                text = f"<|user|>\n{input_text}\n<|assistant|>\n{output_text}\n<|endoftext|>"
            
            formatted.append({"text": text})
        
        return formatted
    
    async def train(self) -> str:
        """Execute the training process."""
        if not self.model or not self.tokenizer:
            await self.prepare_model()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(self.config.train_data_path)
        eval_dataset = None
        if os.path.exists(self.config.eval_data_path):
            eval_dataset = self.prepare_dataset(self.config.eval_data_path)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            report_to="wandb" if self.config.wandb_project else "none",
            run_name=self.config.run_name or f"agent-zero-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            remove_unused_columns=False,
            dataloader_pin_memory=False
        )
        
        # Setup WandB if configured
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=training_args.run_name,
                config=asdict(self.config)
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Start training
        self.logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        self.trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training metrics
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        self.logger.info(f"Training completed. Model saved to: {final_model_path}")
        
        if self.config.wandb_project:
            wandb.finish()
        
        return str(final_model_path)
    
    def evaluate(self, model_path: str, eval_data_path: str) -> Dict[str, float]:
        """Evaluate a trained model."""
        self.logger.info(f"Evaluating model: {model_path}")
        
        # Load model for evaluation
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load evaluation data
        eval_dataset = self.prepare_dataset(eval_data_path)
        
        # Setup trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            dataloader_pin_memory=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        self.logger.info(f"Evaluation completed: {eval_results}")
        return eval_results
    
    @staticmethod
    def create_training_config(
        task_type: str,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs
    ) -> TrainingConfig:
        """Create a training configuration for specific task type."""
        
        # Task-specific defaults
        task_configs = {
            "classification": {
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "lora_rank": 16,
                "model_max_length": 1024
            },
            "explanation": {
                "num_train_epochs": 5,
                "learning_rate": 1e-4,
                "lora_rank": 32,
                "model_max_length": 2048
            },
            "reasoning": {
                "num_train_epochs": 4,
                "learning_rate": 1.5e-4,
                "lora_rank": 24,
                "model_max_length": 2048
            },
            "coding": {
                "num_train_epochs": 6,
                "learning_rate": 1e-4,
                "lora_rank": 32,
                "model_max_length": 2048
            }
        }
        
        # Get task-specific config
        task_specific = task_configs.get(task_type, task_configs["reasoning"])
        
        # Merge with user overrides
        config_dict = {
            "base_model_name": base_model,
            "output_dir": f"./trained_models/{task_type}_{base_model.split('/')[-1]}",
            **task_specific,
            **kwargs
        }
        
        return TrainingConfig(**config_dict)