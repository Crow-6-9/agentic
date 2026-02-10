#!/usr/bin/env python3
"""Automated Model Training System - Optimized OOP Architecture"""
import argparse, json, xml.etree.ElementTree as ET, os, torch, re
from pathlib import Path
from time import time
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                         TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer,
                         DataCollatorForLanguageModeling, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


@dataclass
class TrainingConfig:
    """Immutable training configuration"""
    model: str
    model_path: str
    data_path: str
    epochs: int
    max_length: int
    batch_size: int
    learning_rate: float
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    
    @property
    def is_lora(self) -> bool: 
        return all([self.lora_r, self.lora_alpha, self.lora_dropout])
    
    @property
    def training_method(self) -> str: 
        return "lora" if self.is_lora else "traditional"


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
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None


class DataNormalizer:
    """Normalize TXT/JSON/XML to unified format with EOS"""
    
    @staticmethod
    def load(path: str) -> List[str]:
        """Load and normalize data from any supported format"""
        suffix = Path(path).suffix.lower()
        loaders = {
            '.txt': DataNormalizer._load_txt,
            '.json': DataNormalizer._load_json,
            '.xml': DataNormalizer._load_xml
        }
        return loaders.get(suffix, DataNormalizer._load_txt)(path)
    
    @staticmethod
    def _load_txt(path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    @staticmethod
    def _load_json(path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            return [DataNormalizer._flatten_dict(item) if isinstance(item, dict) else str(item) 
                   for item in data]
        elif isinstance(data, dict):
            # Handle nested dict (like your recipe data)
            if all(isinstance(v, dict) for v in data.values()):
                return [DataNormalizer._flatten_dict(v) for v in data.values()]
            return [DataNormalizer._flatten_dict(data)]
        return [str(data)]
    
    @staticmethod
    def _load_xml(path: str) -> List[str]:
        tree = ET.parse(path)
        root = tree.getroot()
        return [DataNormalizer._xml_to_text(elem) for elem in root]
    
    @staticmethod
    def _flatten_dict(d: dict) -> str:
        """Flatten dict to readable text"""
        return ' '.join(f"{k}: {v}" for k, v in d.items() if v)
    
    @staticmethod
    def _xml_to_text(elem) -> str:
        """Convert XML element to text"""
        text = elem.text or ''
        for child in elem:
            text += f" {child.tag}: {child.text or ''}"
        return text.strip()
    
    @staticmethod
    def save_with_eos(texts: List[str], output_path: str, eos_token: str = "<|endoftext|>"):
        """Save normalized data with EOS tokens"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(f"{text}{eos_token}\n" for text in texts)


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
        
        existing = [d for d in os.listdir(base) if re.match(rf"{self.config.model}-v\d+", d)]
        version = max([int(re.findall(r'\d+', d)[0]) for d in existing], default=0) + 1
        
        output_dir = f"{base}/{self.config.model}-v{version}"
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
            'avg_tokens': sum(token_counts) / len(token_counts)
        }
    
    def _save_metrics(self, token_stats: Dict):
        """Save training metrics to JSON"""
        version = int(re.findall(r'\d+', self.output_dir.split('/')[-1])[0])
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
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        with open(f"{self.output_dir}/training_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def train(self):
        """Main training pipeline"""
        print(f"\n🚀 Starting {self.config.model.upper()} Training - {self.config.training_method.upper()}")
        print(f"📂 Output: {self.output_dir}")
        
        # Load and normalize data
        texts = DataNormalizer.load(self.config.data_path)
        normalized_path = f"{self.output_dir}/normalized_data.txt"
        DataNormalizer.save_with_eos(texts, normalized_path, self.tokenizer.eos_token)
        print(f"✅ Normalized {len(texts)} records → {normalized_path}")
        
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
            self.model.save_pretrained(self.output_dir)
        else:
            trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save metrics
        self._save_metrics(token_stats)
        
        print(f"\n✅ Training Complete! Time: {time() - self.start_time:.2f}s")
        print(f"📁 Model saved: {self.output_dir}")


class GPT2Trainer(BaseTrainer):
    """GPT-2 specific trainer"""
    
    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            local_files_only=True
        )
    
    def _create_dataset(self, texts: List[str]) -> Dataset:
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
    
    def _get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            use_cpu=True,
            save_strategy="epoch",
            report_to="none",
            logging_steps=10
        )
    
    def _get_data_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )


class FlanT5Trainer(BaseTrainer):
    """FLAN-T5 specific trainer"""
    
    def _load_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path, 
            local_files_only=True
        )
    
    def _create_dataset(self, texts: List[str]) -> Dataset:
        # Create input-target pairs
        inputs = [f"Process: {text[:100]}" for text in texts]
        targets = texts
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_length,
            truncation=True,
            padding='max_length'
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_length,
                truncation=True,
                padding='max_length'
            )
        
        model_inputs['labels'] = labels['input_ids']
        return Dataset.from_dict(model_inputs)
    
    def _get_training_args(self) -> Seq2SeqTrainingArguments:
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            use_cpu=True,
            save_strategy="epoch",
            report_to="none",
            logging_steps=10,
            prediction_loss_only=True
        )
    
    def _get_data_collator(self):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
        )


class TrainerFactory:
    """Factory pattern for trainer selection"""
    
    _trainers = {
        'gpt2': GPT2Trainer,
        'flant5': FlanT5Trainer,
        't5': FlanT5Trainer
    }
    
    @classmethod
    def create(cls, config: TrainingConfig) -> BaseTrainer:
        trainer_cls = cls._trainers.get(config.model.lower())
        if not trainer_cls:
            raise ValueError(f"Unsupported model: {config.model}")
        return trainer_cls(config)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Automated Model Training System')
    
    # Required arguments
    parser.add_argument('--model', required=True, choices=['gpt2', 'flant5', 't5'],
                       help='Model type to train')
    parser.add_argument('--model_path', required=True, help='Path to local model')
    parser.add_argument('--data_path', required=True, help='Path to training data (txt/json/xml)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    
    # Optional LoRA arguments
    parser.add_argument('--lora_r', type=int, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, help='LoRA dropout rate')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create config
    config = TrainingConfig(
        model=args.model,
        model_path=args.model_path,
        data_path=args.data_path,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Create and run trainer
    trainer = TrainerFactory.create(config)
    trainer.train()


if __name__ == "__main__":
    main()
