import json
import os

# --- Configuration ---
INPUT_FILE = 'java_rules.json'
OUTPUT_FILE = 'java_rules_dataset.txt'
# ---------------------

def prepare_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    print(f"Found {len(rules)} rules.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for rule in rules:
            # Extract fields
            rule_id = rule.get('rule_id', 'N/A')
            title = rule.get('title', 'N/A')
            description = rule.get('description', 'N/A')

            # Format for GPT-2
            # We structure it so the model learns to associate these fields
            entry = f"Rule ID: {rule_id}\nTitle: {title}\nDescription: {description}\n<|endoftext|>\n"
            f.write(entry)

    print(f"Successfully processed {len(rules)} items into {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
