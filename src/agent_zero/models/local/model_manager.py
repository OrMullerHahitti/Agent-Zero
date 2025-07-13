"""Local model management - downloading, versioning, and storage."""

import os
import json
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import hashlib

try:
    from huggingface_hub import snapshot_download, login, HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LocalModelManager:
    """Manages local model storage, downloading, and versioning."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("model_manager")
        
        # Model registry file
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
        
        if not HF_AVAILABLE:
            self.logger.warning("HuggingFace Hub not available. Install with: pip install huggingface_hub")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
        
        return {
            "models": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    async def download_model(
        self, 
        model_name: str,
        custom_name: Optional[str] = None,
        revision: str = "main",
        force: bool = False
    ) -> str:
        """Download a model from HuggingFace Hub."""
        
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Hub not available")
        
        # Determine local name
        local_name = custom_name or model_name.replace("/", "--")
        model_path = self.models_dir / local_name
        
        # Check if already exists
        if model_path.exists() and not force:
            self.logger.info(f"Model {local_name} already exists at {model_path}")
            return str(model_path)
        
        self.logger.info(f"Downloading model {model_name} to {model_path}")
        
        try:
            # Download model
            downloaded_path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
            
            # Verify download
            if not self._verify_model(model_path):
                raise ValueError("Downloaded model verification failed")
            
            # Update registry
            self.registry["models"][local_name] = {
                "original_name": model_name,
                "revision": revision,
                "download_date": datetime.now().isoformat(),
                "local_path": str(model_path),
                "size_gb": self._calculate_size(model_path),
                "verified": True
            }
            self._save_registry()
            
            self.logger.info(f"Successfully downloaded {model_name} as {local_name}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            # Cleanup partial download
            if model_path.exists():
                shutil.rmtree(model_path)
            raise
    
    def _verify_model(self, model_path: Path) -> bool:
        """Verify that downloaded model is complete and valid."""
        try:
            # Check for essential files
            required_files = ["config.json"]
            for file in required_files:
                if not (model_path / file).exists():
                    self.logger.error(f"Missing required file: {file}")
                    return False
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer is None:
                return False
            
            # Check model files exist (but don't load to save memory)
            model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if not model_files:
                self.logger.error("No model weight files found")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            return False
    
    def _calculate_size(self, path: Path) -> float:
        """Calculate directory size in GB."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)  # Convert to GB
    
    def list_models(self) -> Dict[str, Any]:
        """List all locally available models."""
        return self.registry["models"]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.registry["models"].get(model_name)
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the local path for a model."""
        model_info = self.get_model_info(model_name)
        if model_info and Path(model_info["local_path"]).exists():
            return model_info["local_path"]
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a local model."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            self.logger.warning(f"Model {model_name} not found in registry")
            return False
        
        model_path = Path(model_info["local_path"])
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
                self.logger.info(f"Deleted model {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to delete model {model_name}: {e}")
                return False
        
        # Remove from registry
        del self.registry["models"][model_name]
        self._save_registry()
        
        return True
    
    def update_model(self, model_name: str) -> str:
        """Update a model to the latest version."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        original_name = model_info["original_name"]
        
        # Delete old version
        self.delete_model(model_name)
        
        # Download new version
        return await self.download_model(original_name, model_name, force=True)
    
    def create_model_link(self, source_model: str, link_name: str) -> bool:
        """Create a symbolic link to an existing model."""
        source_info = self.get_model_info(source_model)
        if not source_info:
            self.logger.error(f"Source model {source_model} not found")
            return False
        
        source_path = Path(source_info["local_path"])
        link_path = self.models_dir / link_name
        
        try:
            if link_path.exists():
                link_path.unlink()
            
            link_path.symlink_to(source_path)
            
            # Add to registry
            self.registry["models"][link_name] = {
                **source_info,
                "is_link": True,
                "link_target": source_model,
                "local_path": str(link_path)
            }
            self._save_registry()
            
            self.logger.info(f"Created link {link_name} -> {source_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create link: {e}")
            return False
    
    def get_recommended_models(self, task_type: str) -> List[Dict[str, Any]]:
        """Get recommended models for a specific task type."""
        
        recommendations = {
            "classification": [
                {
                    "name": "meta-llama/Llama-3.1-8B-Instruct",
                    "size": "8B",
                    "description": "General purpose, good for classification tasks",
                    "memory_gb": 16
                },
                {
                    "name": "microsoft/DialoGPT-medium",
                    "size": "355M", 
                    "description": "Lightweight option for simple classification",
                    "memory_gb": 2
                }
            ],
            "explanation": [
                {
                    "name": "codellama/CodeLlama-7b-Instruct-hf",
                    "size": "7B",
                    "description": "Excellent for technical explanations",
                    "memory_gb": 14
                },
                {
                    "name": "meta-llama/Llama-3.1-8B-Instruct",
                    "size": "8B", 
                    "description": "Good general explanation capabilities",
                    "memory_gb": 16
                }
            ],
            "reasoning": [
                {
                    "name": "meta-llama/Llama-3.1-8B-Instruct",
                    "size": "8B",
                    "description": "Strong reasoning capabilities",
                    "memory_gb": 16
                },
                {
                    "name": "mistralai/Mistral-7B-Instruct-v0.1",
                    "size": "7B",
                    "description": "Efficient reasoning model",
                    "memory_gb": 14
                }
            ],
            "coding": [
                {
                    "name": "codellama/CodeLlama-7b-Instruct-hf", 
                    "size": "7B",
                    "description": "Specialized for code generation",
                    "memory_gb": 14
                },
                {
                    "name": "codellama/CodeLlama-13b-Instruct-hf",
                    "size": "13B",
                    "description": "Larger code model for complex tasks",
                    "memory_gb": 26
                }
            ]
        }
        
        return recommendations.get(task_type, recommendations["reasoning"])
    
    def check_disk_space(self) -> Dict[str, float]:
        """Check available disk space."""
        total, used, free = shutil.disk_usage(self.models_dir)
        
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3), 
            "free_gb": free / (1024**3),
            "models_size_gb": sum(
                model["size_gb"] for model in self.registry["models"].values()
                if not model.get("is_link", False)
            )
        }
    
    def cleanup_incomplete_downloads(self) -> List[str]:
        """Clean up incomplete or corrupted downloads."""
        cleaned = []
        
        for model_name, model_info in list(self.registry["models"].items()):
            model_path = Path(model_info["local_path"])
            
            if not model_path.exists():
                # Model directory doesn't exist
                del self.registry["models"][model_name]
                cleaned.append(f"Removed registry entry for missing model: {model_name}")
                continue
            
            if not self._verify_model(model_path):
                # Model is corrupted
                try:
                    shutil.rmtree(model_path)
                    del self.registry["models"][model_name]
                    cleaned.append(f"Removed corrupted model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to remove corrupted model {model_name}: {e}")
        
        if cleaned:
            self._save_registry()
        
        return cleaned
    
    def export_model_list(self, output_path: str) -> None:
        """Export model list to file."""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "models": self.registry["models"],
            "disk_usage": self.check_disk_space()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Model list exported to {output_path}")


# Convenience functions
async def download_recommended_models(task_type: str, models_dir: str = "./models") -> List[str]:
    """Download recommended models for a task type."""
    manager = LocalModelManager(models_dir)
    recommendations = manager.get_recommended_models(task_type)
    
    downloaded = []
    for model in recommendations[:1]:  # Download first recommendation
        try:
            path = await manager.download_model(model["name"])
            downloaded.append(path)
        except Exception as e:
            logging.error(f"Failed to download {model['name']}: {e}")
    
    return downloaded


def setup_model_environment(models_dir: str = "./models") -> LocalModelManager:
    """Setup model environment with common models."""
    manager = LocalModelManager(models_dir)
    
    # Check disk space
    disk_info = manager.check_disk_space()
    if disk_info["free_gb"] < 50:
        logging.warning(f"Low disk space: {disk_info['free_gb']:.1f}GB free")
    
    # Clean up any incomplete downloads
    cleaned = manager.cleanup_incomplete_downloads()
    if cleaned:
        logging.info(f"Cleaned up {len(cleaned)} incomplete downloads")
    
    return manager