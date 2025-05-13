import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

# Load JSON data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r") as f:
    data = json.load(f)

# Extract hallucination scores per topic
hallucination_data = []
for topic, section in data.items():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data in category_data.items():
                halluc = prompt_data.get("hallucination_avg")
                if halluc is not None:
                    hallucination_data.append({
                        "topic": topic,
                        "hallucination_avg": halluc
                    })

# Create DataFrame
df_halluc = pd.DataFrame(hallucination_data)

# Extract topic number for sorting
def extract_number(topic_name):
    match = re.search(r'(\d+)$', topic_name)
    return int(match.group(1)) if match else float('inf')

df_halluc["topic_number"] = df_halluc["topic"].apply(extract_number)

# Compute stats per topic: count, mean, std
halluc_stats = (
    df_halluc
    .groupby(["topic", "topic_number"])["hallucination_avg"]
    .agg(num_tailored_texts="count", mean="mean", std="std")
    .round(2)
    .sort_values("topic_number")
    .reset_index()
    .drop(columns=["topic_number"])
    .set_index("topic")
)

# Print table with counts
print("Hallucination Score Stats per Topic (with Count, Mean, Std):")
print(halluc_stats)

# Assign colors: blue = CS (1–5), orange = L (6–10)
colors_ordered = [
    "blue" if extract_number(topic) <= 5 else "orange"
    for topic in halluc_stats.index
]

# Plot
plt.figure(figsize=(14, 6))
plt.bar(
    halluc_stats.index,
    halluc_stats["mean"],
    yerr=halluc_stats["std"],
    capsize=4,
    color=colors_ordered
)
plt.title("Average Hallucination Score per Topic (1–10) with Std Dev")
plt.ylabel("Average Hallucination Score")
plt.xlabel("Topic")
plt.xticks(rotation=90)

# Add legend
legend_elements = [
    Patch(facecolor='blue', label='CS'),
    Patch(facecolor='orange', label='L')
]
plt.legend(handles=legend_elements, title="Category")

plt.tight_layout()
plt.show()


import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch

# Load JSON data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r") as f:
    data = json.load(f)

# Extract cosine similarity per topic
cosine_data = []
for topic, section in data.items():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data in category_data.items():
                cosine = prompt_data.get("cosine_similarity")
                if cosine is not None:
                    cosine_data.append({
                        "topic": topic,
                        "cosine_similarity": cosine
                    })

# Create DataFrame
df_cosine = pd.DataFrame(cosine_data)

# Extract topic number for sorting
def extract_number(topic_name):
    match = re.search(r'(\d+)$', topic_name)
    return int(match.group(1)) if match else float('inf')

df_cosine["topic_number"] = df_cosine["topic"].apply(extract_number)

# Compute stats per topic
cosine_stats = (
    df_cosine
    .groupby(["topic", "topic_number"])["cosine_similarity"]
    .agg(num_tailored_texts="count", mean="mean", std="std")
    .round(3)
    .sort_values("topic_number")
    .reset_index()
    .drop(columns=["topic_number"])
    .set_index("topic")
)

# Print the table
print("Cosine Similarity Stats per Topic (with Count, Mean, Std):")
print(cosine_stats)

# Assign colors: blue for CS (1–5), orange for L (6–10)
colors_ordered = [
    "blue" if extract_number(topic) <= 5 else "orange"
    for topic in cosine_stats.index
]

# Plot
plt.figure(figsize=(14, 6))
plt.bar(
    cosine_stats.index,
    cosine_stats["mean"],
    yerr=cosine_stats["std"],
    capsize=4,
    color=colors_ordered
)
plt.title("Average Cosine Similarity per Topic (1–10) with Std Dev")
plt.ylabel("Average Cosine Similarity")
plt.xlabel("Topic")
plt.xticks(rotation=90)

# Legend
legend_elements = [
    Patch(facecolor='blue', label='CS '),
    Patch(facecolor='orange', label='L')
]
plt.legend(handles=legend_elements, title="Category")

plt.tight_layout()
plt.show()
