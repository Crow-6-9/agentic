from typing import List
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

from core.trainer import BaseTrainer

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
            text=inputs,
            text_target=targets,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Replace pad_token_id with -100 in labels so the loss ignores padding
        if "labels" in model_inputs:
            model_inputs["labels"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in model_inputs["labels"]
            ]
            
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
