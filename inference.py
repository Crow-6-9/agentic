import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the base Phi-3 model (REQUIRED even for LoRA)
BASE_MODEL_PATH = r"C:\path\to\Phi-3-mini-4k-instruct"

# Path to your fine-tuned model or adapter
# If IS_LORA is True, this should be the path to the adapter folder (e.g. phi3-java/v1_epochs_1_lora)
# If IS_LORA is False, this should be the path to the fully fine-tuned model folder (e.g. phi3-java/v1_epochs_1)
TRAINED_MODEL_PATH = r"C:\Users\Gaurav\OneDrive\Desktop\D-Agent\phi3-java\v1_epochs_1_lora"

# Set to True if TRAINED_MODEL_PATH points to a LoRA adapter
IS_LORA = True 

# Inference parameters
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7

# ==============================================================================
# INFERENCE SCRIPT
# ==============================================================================

def main():
    print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if IS_LORA:
        print(f"Loading base model from {BASE_MODEL_PATH}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map="auto"
        )
        print(f"Loading LoRA adapter from {TRAINED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
    else:
        print(f"Loading fine-tuned model from {TRAINED_MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(
            TRAINED_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map="auto"
        )

    print("\nModel loaded successfully!")
    print("Enter a Java rule title to get an explanation (or type 'exit' to quit).")
    
    while True:
        rule_title = input("\nRule Title: ")
        if rule_title.lower() in ['exit', 'quit']:
            break
            
        # Format the prompt exactly as used in training
        system_prompt = "You are an expert Java developer helper. You explain Java linting rules and provide examples."
        user_prompt = f"How do I solve the Java rule: {rule_title}?"
        
        # <|system|>\n ... <|end|>\n<|user|>\n ... <|end|>\n<|assistant|>\n
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant part (ignoring the prompt echo if present)
        # The generate function usually returns the full sequence including input.
        # We can split by the assistant tag or just print the whole thing if we clean it up.
        
        # A simple way to get just new tokens is to slice the output tensor, 
        # but here we decode everything. Let's try to parse the output.
        if "<|assistant|>\n" in response:
            answer = response.split("<|assistant|>\n")[-1]
        else:
            answer = response

        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
