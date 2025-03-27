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
