import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to your JSON dataset
DATA_PATH = r"C:\Users\Gaurav\OneDrive\Desktop\D-Agent\opp_java_rules.json"

# Path to the downloaded Phi-3 model directory (containing .safetensors)
# Replace this with the actual path where you have downloaded the model
MODEL_PATH = r"C:\path\to\Phi-3-mini-4k-instruct"

# Base directory where trained models will be saved
OUTPUT_BASE_DIR = "phi3-java"

# Training parameters
NUM_EPOCHS = 1
BATCH_SIZE = 1  # Adjust based on your VRAM
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 2048

# ==============================================================================
# DATA LOADING AND FORMATTING
# ==============================================================================

def format_instruction(processed_entry):
    """
    Formats the entry into the Phi-3 chat template.
    """
    system_prompt = "You are an expert Java developer helper. You explain Java linting rules and provide examples."
    user_prompt = f"How do I solve the Java rule: {processed_entry['title']}?"
    assistant_response = f"{processed_entry['description']}\n\nExamples:\n{processed_entry['examples']}"
    
    # Phi-3 Instruction format
    # <|system|>\n system_prompt <|end|>\n<|user|>\n user_prompt <|end|>\n<|assistant|>\n assistant_response <|end|>\n
    
    text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n{assistant_response}<|end|>\n"
    return {"text": text}

def load_and_process_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flat_data = []
    
    # The JSON structure is root -> rulesets -> category -> list of rules
    # We ignore the 'language': 'java' key
    rulesets = data.get("rulesets", {})
    
    for category, rules in rulesets.items():
        if isinstance(rules, list):
            for rule in rules:
                # Extract relevant fields
                title = rule.get("title", "")
                description = rule.get("description", "")
                examples = rule.get("examples", "")
                
                if title and description:
                    flat_data.append({
                        "title": title,
                        "description": description,
                        "examples": examples,
                        "category": category
                    })
    
    print(f"Found {len(flat_data)} rules in total.")
    return flat_data

# ==============================================================================
# VERSIONING LOGIC
# ==============================================================================

def get_next_version_dir(base_dir, epochs):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    max_version = 0
    for d in existing_dirs:
        if d.startswith("v"):
            try:
                # Expected format vX_epochs_Y
                parts = d.split("_")
                version_part = parts[0] # v1
                v_num = int(version_part[1:])
                if v_num > max_version:
                    max_version = v_num
            except:
                pass
                
    next_version = max_version + 1
    new_dir_name = f"v{next_version}_epochs_{epochs}"
    return os.path.join(base_dir, new_dir_name)

# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    # 1. Prepare Output Directory
    output_dir = get_next_version_dir(OUTPUT_BASE_DIR, NUM_EPOCHS)
    print(f"Model will be saved to: {output_dir}")
    
    # 2. Load and Tokenize Data
    raw_data = load_and_process_data(DATA_PATH)
    hf_dataset = Dataset.from_list(raw_data)
    
    formatted_dataset = hf_dataset.map(format_instruction)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Ensure padding token is set
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    
    tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True)
    
    # 3. Load Model
    print("Loading model...")
    # For full fine-tuning, we load in bfloat16 to save some memory, but it's still heavy.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto"
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # Use bfloat16 for Ampere+ GPUs (RTX 30xx/40xx), change to fp16=True if older
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
