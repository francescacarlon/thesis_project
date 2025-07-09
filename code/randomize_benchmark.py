"""
This script generates a randomized benchmark dataset by selecting a subset of tailored texts
from a filtered metadata JSON file for specific topic categories (CS and L). It then enriches
the randomized dataset by attaching the full prompt texts from the original benchmark data,
reorders fields for clarity, and saves the final output to a JSON file.

Main steps:
- Load filtered benchmark metadata and select target topics
- Randomly sample a limited number of tailored prompts per category and topic
- Attach corresponding prompt texts from the full benchmark data
- Reorder fields for consistency and readability
- Save the finalized randomized benchmark dataset for further use
"""

import json
import random
from collections import OrderedDict
from config import BENCHMARK_PATH, RANDOMIZED_BENCHMARK_PATH

# Load benchmark data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Define which topic keys are allowed
cs_target_topics = {"2", "3", "4"}
l_target_topics = {"7", "8", "10"}
target_keys = cs_target_topics.union(l_target_topics)

# Load benchmark data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Build the randomized dataset
randomized_data = {}

for key, value in benchmark_data.items():
    if key not in target_keys:
        continue  # Skip everything not explicitly requested

    instance_code = f"T{key}"
    # original_text = value.get("original_text")
    # original_category = value.get("original_category")
    # original_text_title = value.get("topic")
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
        # "original_category": original_category,
        # "original_text_title": original_text_title,
        # "original_text": original_text,
        "selected_texts": selected_texts
    }


with open(RANDOMIZED_BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(randomized_data, f, indent=4, ensure_ascii=False)

print(f"Randomized benchmark saved to: {RANDOMIZED_BENCHMARK_PATH}")

# Load both JSON files
with open(RANDOMIZED_BENCHMARK_PATH, "r", encoding="utf-8") as f:
    randomized_data = json.load(f)

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    full_text_data = json.load(f)

# Update randomized_data
for topic_key, instance in list(randomized_data.items()):
    topic_number = topic_key.replace("T", "")

    if topic_number not in full_text_data:
        print(f"[WARNING] Topic {topic_number} not found in benchmark.")
        continue

    topic_entry = full_text_data[topic_number]
    topic_title = topic_entry.get("topic", "UNKNOWN_TOPIC")
    tailored_texts = topic_entry.get("tailored_texts", {})
    selected_texts = instance.get("selected_texts", {})

    # Attach prompt texts and keep text field at the top
    for category, prompts in selected_texts.items():
        if not prompts:
            continue
        for prompt_id in list(prompts.keys()):
            try:
                cat, model, prompt_key = prompt_id.split("_", 2)
                prompt_text = tailored_texts.get(model, {}).get(cat, {}).get(prompt_key)

                if prompt_text is None:
                    print(f"[WARNING] Missing text for {prompt_id} in topic {topic_key}")

                # Move "text" field to top
                old_data = prompts[prompt_id]
                reordered = OrderedDict()
                reordered["text"] = prompt_text
                for k, v in old_data.items():
                    if k != "text":
                        reordered[k] = v
                prompts[prompt_id] = reordered
            except Exception as e:
                print(f"[ERROR] Failed to process {prompt_id}: {e}")
                prompts[prompt_id]["text"] = None

    original_text = topic_entry.get("original_text", "UNKNOWN_ORIGINAL_TEXT")

    # Reorder fields as: topic → instance_code → original_text → selected_texts
    reordered_instance = OrderedDict()
    reordered_instance["topic"] = topic_title
    reordered_instance["instance_code"] = instance["instance_code"]
    reordered_instance["original_text"] = original_text
    reordered_instance["selected_texts"] = selected_texts


    randomized_data[topic_key] = reordered_instance

# Save the final output
with open(RANDOMIZED_BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(randomized_data, f, indent=4, ensure_ascii=False)

print("Ordered output: 'topic' → 'instance_code' → 'selected_texts'. All texts populated.")
