import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- Configuration ---
BASE_OUTPUT_DIR = 'gpt2_java_rules_output'
MODEL_NAME = 'gpt2-large' # Fallback if no local model found
# ---------------------

def get_latest_model_dir(base_dir):
    """
    Finds the latest version directory (e.g., v2 is later than v1).
    """
    if not os.path.exists(base_dir):
        return None
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    version_dirs = [d for d in subdirs if os.path.basename(d).startswith('v') and os.path.basename(d)[1:].isdigit()]
    
    if not version_dirs:
        return None
    
    # Sort by version number
    version_dirs.sort(key=lambda x: int(os.path.basename(x)[1:]))
    return version_dirs[-1]

def generate_text(model, tokenizer, prompt_text, max_length=200):
    inputs = tokenizer.encode(prompt_text, return_tensors='pt')
    
    # Generate
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones(inputs.shape, dtype=torch.long)
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    # 1. Determine Model Path
    model_path = get_latest_model_dir(BASE_OUTPUT_DIR)
    
    if model_path:
        print(f"Found trained model at: {model_path}")
    else:
        print(f"No trained model found in {BASE_OUTPUT_DIR}. Using base {MODEL_NAME} config (untrained) for testing structure.")
        model_path = MODEL_NAME

    # 2. Load Model & Tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    # 3. Interactive Loop
    print("\n--- GPT-2 Java Rules Generator ---")
    print("Enter a Title to generate a rule description.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Title: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Format the prompt to match training data structure
        # The model expects "Rule ID: ... \nTitle: ..."
        # Since we want to predict description based on title, we can frame it:
        # Note: If we don't have a Rule ID, we can let the model generate it or just start with Title.
        # Let's try starting with Title since that's what the user provides.
        
        prompt = f"Title: {user_input}\nDescription:"
        
        print("\nGenerating...", end="", flush=True)
        generated_text = generate_text(model, tokenizer, prompt)
        print("\r" + " " * 20 + "\r", end="") # Clear "Generating..."
        
        print("--- Output ---")
        print(generated_text)
        print("----------------\n")

if __name__ == "__main__":
    main()
