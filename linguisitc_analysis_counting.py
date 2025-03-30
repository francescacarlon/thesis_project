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

import json
from config import LINGUISTIC_ANALYSIS_PATH

# Load the JSON data
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

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
print(f"Entries with hallucinations_overall_average ≥ 4.0 and cosine_similarity < 0.8: {halluc_with_low_cos}")
print(f"Entries with hallucinations_overall_average ≥ 4.0 and cosine_similarity missing: {halluc_with_missing_cos}")

