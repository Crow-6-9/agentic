import os
import re
from time import time
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from transformers import (AutoTokenizer, TrainingArguments, Trainer, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from configs.parser import TrainingConfig
from utils.data import DataNormalizer

@dataclass
class TrainingMetrics:
    """Training metrics storage"""
    model_type: str
    training_method: str
    epochs: int
    total_time: float
    max_tokens: int
    avg_tokens: float
    batch_size: int
    learning_rate: float
    version: int
    base_model_path: str = ""
    lora_adapter_path: Optional[str] = None
    merged_model_path: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None

class BaseTrainer(ABC):
    """Abstract base trainer with common functionality"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.output_dir = self._get_versioned_dir()
        self.start_time = 0
        
    def _load_tokenizer(self):
        """Load tokenizer from local path"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            local_files_only=True
        )
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @abstractmethod
    def _load_model(self):
        """Load base model - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _create_dataset(self, texts: List[str]) -> Dataset:
        """Create dataset - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_training_args(self) -> TrainingArguments:
        """Get training arguments - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_data_collator(self):
        """Get data collator - implemented by subclasses"""
        pass
    
    def _get_versioned_dir(self) -> str:
        """Auto-versioning: create v1, v2, v3... directories"""
        base = f"./{self.config.model}_{self.config.training_method}_models"
        os.makedirs(base, exist_ok=True)
        existing = os.listdir(base)
        versions = []
        for d in existing:
            match = re.search(r'v(\d+)', d)
            if match:
                versions.append(int(match.group(1)))
        version = max(versions, default=0) + 1
        
        output_dir = f"{base}/{self.config.model}_v{version}_{self.config.training_method}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _apply_lora(self):
        """Apply LoRA if configured"""
        if self.config.is_lora:
            task_type = (TaskType.SEQ_2_SEQ_LM if 't5' in self.config.model.lower() 
                        else TaskType.CAUSAL_LM)
            
            lora_config = LoraConfig(
                task_type=task_type,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q", "v"] if 't5' in self.config.model.lower() else None
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"\n🔧 LoRA Applied (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
            self.model.print_trainable_parameters()
    
    def _compute_token_stats(self, texts: List[str]) -> Dict:
        """Compute token statistics"""
        token_counts = [len(self.tokenizer(text, truncation=True, 
                       max_length=self.config.max_length)['input_ids']) 
                       for text in texts]
        return {
            'max_tokens': max(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts) if len(token_counts) > 0 else 0
        }
    
    def _save_metrics(self, token_stats: Dict):
        """Save training metrics to JSON"""
        import json
        version_match = re.search(r'v(\d+)', self.output_dir)
        version = int(version_match.group(1)) if version_match else 1
        
        merged_path = f"{self.output_dir}_merged" if self.config.is_lora else None
        adapter_path = self.output_dir if self.config.is_lora else None

        metrics = TrainingMetrics(
            model_type=self.config.model,
            training_method=self.config.training_method,
            epochs=self.config.epochs,
            total_time=time() - self.start_time,
            max_tokens=token_stats['max_tokens'],
            avg_tokens=token_stats['avg_tokens'],
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            version=version,
            base_model_path=self.config.model_path,
            lora_adapter_path=adapter_path,
            merged_model_path=merged_path,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        with open(f"{self.output_dir}/training_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def train(self, texts: List[str]):
        """Main training pipeline taking a pre-loaded dataset list of strings"""
        print(f"\n🚀 Starting {self.config.model.upper()} Training - {self.config.training_method.upper()}")
        print(f"📂 Output: {self.output_dir}")
        
        normalized_path = f"{self.output_dir}/normalized_data.txt"
        DataNormalizer.save_with_eos(texts, normalized_path, self.tokenizer.eos_token)
        print(f"✅ Saved normalized {len(texts)} subset records → {normalized_path}")
        
        # Compute token stats
        token_stats = self._compute_token_stats(texts)
        print(f"📊 Tokens - Max: {token_stats['max_tokens']}, Avg: {token_stats['avg_tokens']:.2f}")
        
        # Apply LoRA if needed
        self._apply_lora()
        
        # Create dataset
        dataset = self._create_dataset(texts)
        
        # Training arguments
        training_args = self._get_training_args()
        
        # Trainer
        trainer_class = Seq2SeqTrainer if 't5' in self.config.model.lower() else Trainer
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self._get_data_collator()
        )
        
        # Train
        self.start_time = time()
        print(f"\n🏋️ Training started (CPU-only, {self.config.epochs} epochs)...")
        trainer.train()
        
        # Save
        if self.config.is_lora:
            print(f"\n💾 Saving LoRA adapters to {self.output_dir}...")
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            print(f"🔄 Merging LoRA weights with base model...")
            merged_model = self.model.merge_and_unload()
            merged_dir = f"{self.output_dir}_merged"
            print(f"💾 Saving merged model to {merged_dir}...")
            merged_model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)
            print(f"✅ Merged model ready for GGUF conversion at: {merged_dir}")
        else:
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        
        # Save metrics
        self._save_metrics(token_stats)
        
        print(f"\n✅ Training Complete! Time: {time() - self.start_time:.2f}s")
        print(f"📁 Model saved: {self.output_dir}")
