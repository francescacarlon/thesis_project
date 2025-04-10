"""from config import LINGUISTIC_ANALYSIS_PATH
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
"""

"""
import os
import json
from config import LINGUISTIC_ANALYSIS_PATH  # Adjust this if you're not using a config module

# Load the original JSON file
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Result structure
filtered_tailored_texts = {}

# Thresholds
HALL_THRESHOLD = 4.0

# Traverse each top-level entry (e.g., ID 1, 2, 3...)
for entry_id, entry_data in data.items():
    try:
        if not (1 <= int(entry_id) <= 10):
            continue  # Skip entries not in the range 1–10
    except ValueError:
        continue  # Skip non-integer keys, just in case

    tailored = entry_data.get("tailored_texts", {})
    for llm, topics in tailored.items():
        for topic, prompts in topics.items():
            for prompt, content in prompts.items():
                hall_avg = content.get("hallucination_scores", {}).get("hallucinations_overall_average")
                cos_sim = content.get("cosine_similarity")
                token_count = content.get("token_count")
                readability = content.get("readability")
                pos = content.get("pos")
                bleu = content.get("bleu_score")
                rouge = content.get("rouge_scores")

                if hall_avg is not None:
                    # print(f"Entry {entry_id} | LLM: {llm} | Topic: {topic} | Prompt: {prompt} | Hall Avg: {hall_avg}")

                    try:
                        if float(hall_avg) >= HALL_THRESHOLD:
                            # print(f"Entry {entry_id} | LLM: {llm} | Topic: {topic} | Prompt: {prompt} | Hall Avg: {hall_avg}")

                            # Build nested filtered structure
                            filtered_tailored_texts.setdefault(entry_id, {}).setdefault("tailored_texts", {}) \
                                .setdefault(llm, {}).setdefault(topic, {})[prompt] = {
                                    "cosine_similarity": float(cos_sim) if cos_sim is not None else None,
                                    "hallucination_avg": float(hall_avg),
                                    "token_count": int(token_count) if token_count is not None else None,
                                    "readability": readability if readability is not None else None,
                                    "pos": pos if pos is not None else None,
                                    "bleu_score": bleu if bleu is not None else None,
                                    "rouge_scores": rouge
                                }
                    except (ValueError, TypeError):
                        continue

# Save filtered results
# output_path = "./data/filtered_metadata_halls_4.0.json"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# 
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(filtered_tailored_texts, f, ensure_ascii=False, indent=2)
# 
# print(f"\n✅ Saved filtered results to: {output_path}")


import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Load JSON
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r", encoding="utf-8") as f:
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


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Extract cosine similarity values
all_cosine_similarities = []

for col in df.columns:
    entry = df[col]["tailored_texts"]
    for model in entry:
        for group in entry[model]:
            for prompt in entry[model][group]:
                cos_sim = entry[model][group][prompt].get("cosine_similarity")
                if cos_sim is not None:
                    all_cosine_similarities.append({
                        "entry": int(col),
                        "model": model,
                        "group": group,
                        "prompt": prompt,
                        "cosine_similarity": cos_sim
                    })

# Create a DataFrame
all_cos_sim_df = pd.DataFrame(all_cosine_similarities)

# Sort entries to ensure correct line plotting
all_cos_sim_df.sort_values(by="entry", inplace=True)

# Plot with prompt as color and group as line style
sns.set(style="whitegrid")
g = sns.relplot(
    data=all_cos_sim_df,
    x="entry",
    y="cosine_similarity",
    col="model",
    hue="prompt",        # prompt controls color
    style="group",       # group controls line dashes
    kind="line",
    marker="o",
    col_wrap=2,
    height=5,
    aspect=1.5
)

# Add vertical line between topic 5 and 6
for ax, model_name in zip(g.axes.flat, g.col_names):
    ax.axvline(x=5.5, color="gray", linestyle="--", linewidth=1)
    
    # Extract groups used in this subplot
    groups = all_cos_sim_df[all_cos_sim_df["model"] == model_name]["group"].unique()
    label = " & ".join(sorted(groups))
    

# Final touches
g.set_titles("{col_name}")
g.set_axis_labels("Entry", "Cosine Similarity")
g._legend.set_title("Prompt / Group")
g.fig.suptitle("Cosine Similarity by Model\nColor = Prompt, Line Style = Group", fontsize=16)
plt.subplots_adjust(top=0.88)
plt.show()

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Extract hallucination values
hallucination_data = []
for col in df.columns:
    entry = df[col]["tailored_texts"]
    for model in entry:
        for group in entry[model]:
            for prompt in entry[model][group]:
                hallucination = entry[model][group][prompt].get("hallucination_avg")
                if hallucination is not None:
                    hallucination_data.append({
                        "entry": int(col),
                        "model": model,
                        "group": group,
                        "prompt": prompt,
                        "hallucination_avg": hallucination
                    })

# Create DataFrame
hallucination_df = pd.DataFrame(hallucination_data)
hallucination_df.sort_values(by="entry", inplace=True)

# Plot
sns.set(style="whitegrid")
g = sns.relplot(
    data=hallucination_df,
    x="entry",
    y="hallucination_avg",
    col="model",
    hue="prompt",      # color by prompt
    style="group",     # line style by group
    kind="line",
    marker="o",
    col_wrap=2,
    height=5,
    aspect=1.5
)

# Add vertical line at topic boundary (between entry 5 and 6) and group label
for ax, model_name in zip(g.axes.flat, g.col_names):
    ax.axvline(x=5.5, color="gray", linestyle="--", linewidth=1)
    groups = hallucination_df[hallucination_df["model"] == model_name]["group"].unique()
    label = " & ".join(sorted(groups))

# Final polish
g.set_titles("{col_name}")
g.set_axis_labels("Entry", "Avg. Hallucinations")
g._legend.set_title("Prompt / Group")
g.fig.suptitle("Average Hallucinations by Model\nColor = Prompt, Line Style = Group", fontsize=16)
plt.subplots_adjust(top=0.88)
plt.show()


import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r") as f:
    data = json.load(f)

# Extract token counts per model and category
token_data = []

for section in data.values():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data in category_data.items():
                token_data.append({
                    "model": model,
                    "category": category,
                    "token_count": prompt_data.get("token_count", 0)
                })

# Create DataFrame
df = pd.DataFrame(token_data)

# Compute average token counts per model per category
avg_token_counts = df.groupby(["category", "model"])["token_count"].mean().unstack().round(2)

# Display the average table
print("Average Token Count per Model by Category:\n")
print(avg_token_counts)

# Compute mean and standard deviation per model per category
model_stats = df.groupby(["category", "model"])["token_count"].agg(["mean", "std"]).round(2)

# To print as a readable table
print("Token Count Stats per Model by Category:")
print(model_stats)

# Plot the results
plt.figure(figsize=(12, 6))
avg_token_counts.T.plot(kind="bar")
plt.title("Average Token Count per Model by Category")
plt.ylabel("Average Token Count")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r") as f:
    data = json.load(f)

# Extract prompt, category, and token count
prompt_with_cat_data = []

for section in data.values():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data_item in category_data.items():
                prompt_with_cat_data.append({
                    "prompt": prompt,
                    "category": category,
                    "token_count": prompt_data_item.get("token_count", 0)
                })

# Create DataFrame
prompt_df_cat = pd.DataFrame(prompt_with_cat_data)

# Compute average token count per prompt and category
avg_token_per_prompt_cat = prompt_df_cat.groupby(["category", "prompt"])["token_count"].mean().unstack().round(2)

# Print the results
print("Average Token Count per Prompt by Category:")
print(avg_token_per_prompt_cat)

# Compute mean and standard deviation per prompt per category
prompt_stats = prompt_df_cat.groupby(["category", "prompt"])["token_count"].agg(["mean", "std"]).round(2)

# To print the result
print("Token Count Stats per Prompt by Category:")
print(prompt_stats)


# Plot
avg_token_per_prompt_cat.T.plot(kind="bar", figsize=(12, 6))
plt.title("Average Token Count per Prompt Number by Category")
plt.ylabel("Average Token Count")
plt.xlabel("Prompt Number")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pivot for plotting
mean_model_df = model_stats["mean"].unstack()
std_model_df = model_stats["std"].unstack()

# Print summary table
print("Token Count Stats per Model by Category:")
print(model_stats)

# Plot with error bars
mean_model_df.T.plot(kind="bar", yerr=std_model_df.T, figsize=(12, 6), capsize=4)
plt.title("Average Token Count per Model by Category (with Std Dev)")
plt.ylabel("Average Token Count")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Pivoted versions
mean_df = prompt_stats["mean"].unstack()
std_df = prompt_stats["std"].unstack()

# Plot with error bars
mean_df.T.plot(kind="bar", yerr=std_df.T, figsize=(12, 6), capsize=4)
plt.title("Average Token Count per Prompt Number by Category (with Std Dev)")
plt.ylabel("Average Token Count")
plt.xlabel("Prompt Number")
plt.xticks(rotation=45)
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

# Extract token counts per topic
topic_data = []
for topic, section in data.items():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data in category_data.items():
                topic_data.append({
                    "topic": topic,
                    "token_count": prompt_data.get("token_count", 0)
                })

# Create DataFrame
df_topic = pd.DataFrame(topic_data)

# Helper to extract topic number
def extract_number(topic_name):
    match = re.search(r'(\d+)$', topic_name)
    return int(match.group(1)) if match else float('inf')

df_topic["topic_number"] = df_topic["topic"].apply(extract_number)

# Group by topic and compute mean & std
topic_stats = (
    df_topic
    .groupby(["topic", "topic_number"])["token_count"]
    .agg(["mean", "std"])
    .round(2)
    .sort_values("topic_number")
    .reset_index()
)

# Drop topic_number for display
topic_stats_cleaned = topic_stats.drop(columns=["topic_number"]).set_index("topic")

# Add (CS) or (L) to topic labels
def label_category(topic):
    num = extract_number(topic)
    if num <= 5:
        return f"{topic}"
    elif num <= 10:
        return f"{topic}"
    else:
        return topic

topic_stats_cleaned_labeled = topic_stats_cleaned.copy()
topic_stats_cleaned_labeled.index = [label_category(t) for t in topic_stats_cleaned.index]

# Set bar colors
colors = [
    "blue" if extract_number(topic) <= 5 else "orange"
    for topic in topic_stats_cleaned.index
]

# Plot
plt.figure(figsize=(14, 6))
plt.bar(
    topic_stats_cleaned_labeled.index,
    topic_stats_cleaned["mean"],
    yerr=topic_stats_cleaned["std"],
    capsize=4,
    color=colors
)
plt.title("Average Token Count per Topic with Std Dev")
plt.ylabel("Average Token Count")
plt.xlabel("Topic")
plt.xticks(rotation=90)
plt.tight_layout()

# Add legend
legend_elements = [
    Patch(facecolor='blue', label='CS'),
    Patch(facecolor='orange', label='L')
]
plt.legend(handles=legend_elements, title="Category")

plt.show()

import json
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the JSON data
with open("./data/filtered_metadata_halls_4.0_no_CL.json", "r") as f:
    data = json.load(f)

# Extract token counts per topic
topic_data = []
for topic, section in data.items():
    for model, model_data in section.get("tailored_texts", {}).items():
        for category, category_data in model_data.items():
            for prompt, prompt_data in category_data.items():
                topic_data.append({
                    "topic": topic,
                    "token_count": prompt_data.get("token_count", 0)
                })

# Create DataFrame
df_topic = pd.DataFrame(topic_data)

# Extract numeric suffix for sorting
def extract_number(topic_name):
    match = re.search(r'(\d+)$', topic_name)
    return int(match.group(1)) if match else float('inf')

df_topic["topic_number"] = df_topic["topic"].apply(extract_number)

# Compute stats per topic: number of texts, mean, std
topic_stats = (
    df_topic
    .groupby(["topic", "topic_number"])["token_count"]
    .agg(num_tailored_texts="count", mean="mean", std="std")
    .round(2)
    .sort_values("topic_number")
    .reset_index()
)

# Drop helper column for clean display
topic_stats_cleaned = topic_stats.drop(columns=["topic_number"]).set_index("topic")

# Print the result
print("Token Count Stats per Topic (Number of Texts, Mean, Std):")
print(topic_stats_cleaned)
"""

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
