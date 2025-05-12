from config import RANDOMIZED_BENCHMARK_WITH_SCORES_PATH
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Load JSON data
with open(RANDOMIZED_BENCHMARK_WITH_SCORES_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Build the records list
records = []

for task_id, task_content in benchmark_data.items():
    topic = task_content.get("topic")
    
    # LLM-generated outputs
    for texts_from_group, outputs in task_content.get("selected_texts", {}).items():
        for model_id, model_data in outputs.items():
            texts_from = "Linguistics" if model_id.startswith("L_") else "Computer Science"
            records.append({
                "model_id": model_id,
                "token_count": model_data.get("token_count"),
                "LLMs_judge": model_data.get("LLMs_judge"),
                "humans_judge": model_data.get("humans_judge"),
                "topic": topic,
                "texts_from": texts_from,
            })

    # Original texts
    for key in task_content.keys():
        if key.startswith("original_text") and isinstance(task_content[key], dict):
            score_block = task_content[key]
            if "LLMs_judge" in score_block and "humans_judge" in score_block:
                texts_from = "Linguistics" if task_id in ["T7", "T8", "T10"] else "Computer Science"
                records.append({
                    "model_id": key,
                    "token_count": task_content.get("token_count", None),
                    "LLMs_judge": score_block["LLMs_judge"],
                    "humans_judge": score_block["humans_judge"],
                    "topic": topic,
                    "texts_from": texts_from,
                })

df = pd.DataFrame(records)
sns.set(style="whitegrid")

# --------------------- SCATTER PLOTS WITH LABELS ---------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

for i, score_type in enumerate(["LLMs_judge", "humans_judge"]):
    sns.scatterplot(
        data=df,
        x="token_count",
        y=score_type,
        hue="texts_from",
        palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
        s=80,
        edgecolor='w',
        ax=axes[i]
    )
    axes[i].set_title(f"{score_type} vs Token Count", fontsize=14)
    axes[i].set_xlabel("Token Count")
    axes[i].set_ylabel("Score")
    axes[i].legend(title="Texts from")
    axes[i].set_ylim(bottom=10)

    # Add text labels
    for j in range(len(df)):
        row = df.iloc[j]
        axes[i].text(row["token_count"] + 1, row[score_type] + 0.5, row["model_id"], fontsize=8)

plt.tight_layout()
plt.show()

# --------------------- CORRELATION & DESCRIPTIVE STATS ---------------------
for score_type in ["LLMs_judge", "humans_judge"]:
    print(f"Correlation between Token Count and {score_type} by 'texts_from':\n")
    for group in df["texts_from"].unique():
        subset = df[df["texts_from"] == group]
        rho, spearman_p = spearmanr(subset["token_count"], subset[score_type])
        r, pearson_p = pearsonr(subset["token_count"], subset[score_type])
        
        print(f"{group}:")
        print(f"  Spearman: rho = {rho:.2f}, p = {spearman_p:.4f}")
        print(f"  Pearson:  r = {r:.2f}, p = {pearson_p:.4f}")
        print(f"  Mean token count  = {subset['token_count'].mean():.2f}, SD = {subset['token_count'].std():.2f}")
        print(f"  Mean {score_type} = {subset[score_type].mean():.2f}, SD = {subset[score_type].std():.2f}\n")

# --------------------- REGRESSION LINES ---------------------
for score_type in ["LLMs_judge", "humans_judge"]:
    sns.lmplot(
        data=df,
        x="token_count",
        y=score_type,
        hue="texts_from",
        palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
        scatter_kws={'s': 70, 'edgecolor': 'w'},
        line_kws={'linewidth': 2},
        ci=None
    )

    plt.title(f"{score_type} vs Token Count with Linear Trend Lines")
    plt.xlabel("Token Count")
    plt.ylabel(score_type)
    plt.tight_layout()
    plt.show()
