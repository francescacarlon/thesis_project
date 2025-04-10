import json
import random
from config import BENCHMARK_PATH, RANDOMIZED_BENCHMARK_PATH

# Load benchmark data
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Define which topic keys are allowed
cs_target_topics = {"2", "3", "4"}
l_target_topics = {"7", "8", "10"}
target_keys = cs_target_topics.union(l_target_topics)

# Load benchmark data
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Build the randomized dataset
randomized_data = {}

for key, value in benchmark_data.items():
    if key not in target_keys:
        continue  # Skip everything not explicitly requested

    instance_code = f"T{key}"
    original_text = value.get("original_text")
    original_category = value.get("original_category")
    original_text_title = value.get("topic")
    tailored_texts = value.get("tailored_texts", {})

    selected_texts = {}

    def collect_prompts(category_name):
        collected = []
        for model, categories in tailored_texts.items():
            if not isinstance(categories, dict):
                continue
            prompts = categories.get(category_name, {})
            if not isinstance(prompts, dict):
                continue
            for prompt_key, prompt_text in prompts.items():
                collected.append((f"{category_name}_{model}_{prompt_key}", prompt_text))
        return collected

    if key in cs_target_topics:
        cs_prompts = collect_prompts("CS")
        selected_texts["CS"] = dict(random.sample(cs_prompts, min(2, len(cs_prompts)))) if cs_prompts else None

    if key in l_target_topics:
        l_prompts = collect_prompts("L")
        selected_texts["L"] = dict(random.sample(l_prompts, min(2, len(l_prompts)))) if l_prompts else None

    randomized_data[instance_code] = {
        "instance_code": instance_code,
        "original_category": original_category,
        "original_text_title": original_text_title,
        "original_text": original_text,
        "selected_texts": selected_texts
    }


with open(RANDOMIZED_BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(randomized_data, f, indent=4, ensure_ascii=False)

print(f"Randomized benchmark saved to: {RANDOMIZED_BENCHMARK_PATH}")
