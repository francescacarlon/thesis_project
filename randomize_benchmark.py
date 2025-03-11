import json
import random
from config import BENCHMARK_PATH, RANDOMIZED_BENCHMARK_PATH

# Load benchmark data
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Process data
randomized_data = {}
for key, value in benchmark_data.items():
    instance_code = f"T{key}"  # Assign unique instance code
    original_text = value.get("original_text")
    original_category = value.get("original_category")
    original_text_title = value.get("topic")  # Keep original text title
    tailored_texts = value.get("tailored_texts", {})
    
    if isinstance(tailored_texts, dict):
        selected_texts = {}
        for model, categories in tailored_texts.items():
            if isinstance(categories, dict):
                selected_texts[model] = {}
                for category, prompts in categories.items():
                    if isinstance(prompts, dict) and prompts:
                        selected_prompt_key = random.choice(list(prompts.keys()))  # Select prompt key
                        selected_key = f"{category}_{model}_{selected_prompt_key}"  # Format key
                        selected_texts[model][category] = {
                            selected_key: prompts[selected_prompt_key]  # Store text with formatted key
                        }
                    else:
                        selected_texts[model][category] = None
    else:
        selected_texts = None
    
    randomized_data[instance_code] = {
        "instance_code": instance_code,
        "original_category": original_category,
        "original_text_title": original_text_title,
        "original_text": original_text,
        "selected_texts": selected_texts
    }

# Save randomized data
with open(RANDOMIZED_BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(randomized_data, f, indent=4, ensure_ascii=False)

print(f"Randomized benchmark saved to: {RANDOMIZED_BENCHMARK_PATH}")
