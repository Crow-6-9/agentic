import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# --- Configuration Variables ---
DATA_FILE = 'java_rules_dataset.txt'
MODEL_NAME = 'gpt2-large'  # Using GPT-2 Large
BASE_OUTPUT_DIR = 'gpt2_java_rules_output'
NUM_EPOCHS = 3
BATCH_SIZE = 1  # Small batch size for CPU/Memory constraints
BLOCK_SIZE = 128 # reduced block size to save memory
# -------------------------------

def get_next_version_dir(base_dir):
    """
    Creates a versioned directory (e.g., base_dir/v1, base_dir/v2).
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    version = 1
    while True:
        version_dir = os.path.join(base_dir, f"v{version}")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
            return version_dir
        version += 1

def train():
    # 1. Setup Data and Model Paths
    output_dir = get_next_version_dir(BASE_OUTPUT_DIR)
    print(f"Output directory set to: {output_dir}")

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run prepare_data.py first.")
        return

    # 2. Tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Model
    print(f"Loading model {MODEL_NAME}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # 4. Dataset
    print("Loading dataset...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=DATA_FILE,
        block_size=BLOCK_SIZE 
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 5. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=1000,
        save_total_limit=2,
        use_cpu=True, # Force CPU usage
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 7. Train
    print("Starting training (on CPU)... This may take a while.")
    trainer.train()

    # 8. Save Model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()
