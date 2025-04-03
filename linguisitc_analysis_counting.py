from config import LINGUISTIC_ANALYSIS_PATH
import json

# Load the JSON file
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Recursive function to extract all cosine similarity values
def extract_cosine_similarities(d):
    similarities = []
    if isinstance(d, dict):
        for key, value in d.items():
            if key == "cosine_similarity":
                similarities.append(value)
            else:
                similarities.extend(extract_cosine_similarities(value))
    elif isinstance(d, list):
        for item in d:
            similarities.extend(extract_cosine_similarities(item))
    return similarities


# Extract similarities
cosine_similarities = extract_cosine_similarities(data)

# Count based on threshold
count_above_equal_0_8 = sum(1 for sim in cosine_similarities if sim >= 0.8)
count_below_0_8 = sum(1 for sim in cosine_similarities if sim < 0.8)

# Print results
print(f"Cosine similarity >= 0.8: {count_above_equal_0_8}")
print(f"Cosine similarity < 0.8: {count_below_0_8}")


# Initialize counters
halluc_with_high_cos = 0
halluc_with_low_cos = 0
halluc_with_missing_cos = 0

# Traverse function
def traverse_fixed_hallucination_structure(d, halluc_threshold=4.0, sim_threshold=0.8):
    global halluc_with_high_cos, halluc_with_low_cos, halluc_with_missing_cos
    if isinstance(d, dict):
        hall_scores = d.get("hallucination_scores", {})
        hall_avg = hall_scores.get("hallucinations_overall_average")
        if hall_avg is not None:
            try:
                hall_val = float(hall_avg)
                if hall_val >= halluc_threshold:
                    cos_val = d.get("cosine_similarity")
                    if cos_val is not None:
                        cos_val = float(cos_val)
                        if cos_val >= sim_threshold:
                            halluc_with_high_cos += 1
                        else:
                            halluc_with_low_cos += 1
                    else:
                        halluc_with_missing_cos += 1
            except (ValueError, TypeError):
                pass
        for value in d.values():
            traverse_fixed_hallucination_structure(value, halluc_threshold, sim_threshold)
    elif isinstance(d, list):
        for item in d:
            traverse_fixed_hallucination_structure(item, halluc_threshold, sim_threshold)

# Run the function
traverse_fixed_hallucination_structure(data)

# Print the results
print(f"Entries with hallucinations_overall_average ≥ 4.0 and cosine_similarity ≥ 0.8: {halluc_with_high_cos}")
# print(f"Entries with hallucinations_overall_average ≥ 4.0 and cosine_similarity < 0.8: {halluc_with_low_cos}")
# print(f"Entries with hallucinations_overall_average ≥ 4.0 and cosine_similarity missing: {halluc_with_missing_cos}")

import os
import json
from config import LINGUISTIC_ANALYSIS_PATH

# Load the original JSON file
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Result structure
filtered_tailored_texts = {}

# Thresholds
HALL_THRESHOLD = 4.0
COS_THRESHOLD = 0.7

# Traverse each top-level entry (e.g., ID 1, 2, 3...)
for entry_id, entry_data in data.items():
    tailored = entry_data.get("tailored_texts", {})
    for llm, topics in tailored.items():
        for topic, prompts in topics.items():
            for prompt, content in prompts.items():
                hall_avg = content.get("hallucination_scores", {}).get("hallucinations_overall_average")
                cos_sim = content.get("cosine_similarity")
                token_count = content.get("token_count")
                if hall_avg is not None and cos_sim is not None:
                    try:
                        if float(hall_avg) >= HALL_THRESHOLD and float(cos_sim) >= COS_THRESHOLD:
                            # Build nested filtered structure
                            filtered_tailored_texts.setdefault(entry_id, {}).setdefault("tailored_texts", {}) \
                                .setdefault(llm, {}).setdefault(topic, {})[prompt] = {
                                    "cosine_similarity": float(cos_sim),
                                    "hallucination_avg": float(hall_avg),
                                    "token_count": int(token_count)
                                }
                    except (ValueError, TypeError):
                        continue

# Save filtered results
output_path = "./data/filtered_metadata_0.7.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_tailored_texts, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved filtered results to: {output_path}")


import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Load JSON
with open("./data/filtered_metadata_no_CL_0.7.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Build topic mapping
# Structure: combination_counts[background][LLM-prompt] = count
# Structure: topic_ids[background][LLM-prompt] = [list of topic IDs]
combination_counts = defaultdict(lambda: defaultdict(int))
topic_ids = defaultdict(lambda: defaultdict(list))

for topic_id, entry in data.items():
    tailored_texts = entry.get("tailored_texts", {})
    for llm, backgrounds in tailored_texts.items():
        for background, prompts in backgrounds.items():
            for prompt_id in prompts:
                key = f"{llm} - {prompt_id}"
                combination_counts[background][key] += 1
                topic_ids[background][key].append(topic_id)

# Prepare data
all_combinations = sorted(set(
    combo for bg_counts in combination_counts.values() for combo in bg_counts
))
cs_counts = [combination_counts["CS"].get(combo, 0) for combo in all_combinations]
l_counts = [combination_counts["L"].get(combo, 0) for combo in all_combinations]
cs_topics = [", ".join(topic_ids["CS"].get(combo, [])) for combo in all_combinations]
l_topics = [", ".join(topic_ids["L"].get(combo, [])) for combo in all_combinations]

# Plotting
x = np.arange(len(all_combinations))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))

bars_cs = ax.bar(x - width/2, cs_counts, width, label='CS')
bars_l = ax.bar(x + width/2, l_counts, width, label='L')

# Annotate each bar with entry numbers (topics)
for i, bar in enumerate(bars_cs):
    if cs_counts[i] > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, cs_topics[i],
                ha='center', va='bottom', fontsize=8, rotation=90)

for i, bar in enumerate(bars_l):
    if l_counts[i] > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, l_topics[i],
                ha='center', va='bottom', fontsize=8, rotation=90)

# Aesthetics
ax.set_xlabel('LLM - Prompt')
ax.set_ylabel('Count')
ax.set_title('LLM-Prompt Combinations by Background with Topics')
ax.set_xticks(x)
ax.set_xticklabels(all_combinations, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()
