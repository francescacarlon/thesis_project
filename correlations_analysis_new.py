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
            texts_from = "Computer Science" if model_id.startswith("L_") else "Linguistics"
            readability = model_data.get("readability", {})
            bleu_score = model_data.get("bleu_score", {})
            rouge_scores = model_data.get("rouge_scores", {})
            bertscore = model_data.get("bertscore", {})
            pos = model_data.get("pos", {})
            records.append({
                "model_id": model_id,
                "token_count": model_data.get("token_count"),
                "cosine_similarity": model_data.get("cosine_similarity"),
                "flesch_reading_ease": readability.get("flesch_reading_ease"),
                "flesch_kincaid_grade": readability.get("flesch_kincaid_grade"),
                "smog_index": readability.get("smog_index"),
                "bleu_score":bleu_score.get("bleu_score"),
                "rouge_1":rouge_scores.get("rouge_1"),
                "rouge_2":rouge_scores.get("rouge_2"),
                "rouge_L":rouge_scores.get("rouge_L"),
                "bertscore_precision":bertscore.get("bertscore_precision"),
                "bertscore_recall":bertscore.get("bertscore_recall"),
                "bertscore_f1":bertscore.get("bertscore_f1"),
                "JJ": pos.get("JJ"),
                "NN": pos.get("NN"),
                "VB": pos.get("VB"),
                "DT": pos.get("DT"),
                "IN": pos.get("IN"),
                "MD": pos.get("MD"),
                "CC": pos.get("CC"),
                "VBG": pos.get("VBG"),
                "CD": pos.get("CD"),
                "NNP": pos.get("NNP"),
                "RB": pos.get("RB"),
                "VBN": pos.get("VBN"),
                "RP": pos.get("RP"),
                "FW": pos.get("FW"),
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
                texts_from = "Linguistics" if task_id in ["T2", "T3", "T4"] else "Computer Science"
                readability = score_block.get("readability", {})
                pos = score_block.get("pos", {})
                records.append({
                    "model_id": key,
                    "token_count": task_content.get("token_count", None),
                    "flesch_reading_ease": readability.get("flesch_reading_ease"),
                    "flesch_kincaid_grade": readability.get("flesch_kincaid_grade"),
                    "smog_index": readability.get("smog_index"),
                    "JJ": pos.get("JJ"),
                    "NN": pos.get("NN"),
                    "VB": pos.get("VB"),
                    "DT": pos.get("DT"),
                    "IN": pos.get("IN"),
                    "MD": pos.get("MD"),
                    "CC": pos.get("CC"),
                    "VBG": pos.get("VBG"),
                    "CD": pos.get("CD"),
                    "NNP": pos.get("NNP"),
                    "RB": pos.get("RB"),
                    "VBN": pos.get("VBN"),
                    "RP": pos.get("RP"),
                    "FW": pos.get("FW"),
                    "LLMs_judge": score_block["LLMs_judge"],
                    "humans_judge": score_block["humans_judge"],
                    "topic": topic,
                    "texts_from": texts_from,
                })

df = pd.DataFrame(records)
sns.set(style="whitegrid")

### LLMs and Humans evaluation with lingusitic metrics correlation ###

def compute_linguistic_metric_correlations(df, linguistic_metrics, judgment_scores=["LLMs_judge", "humans_judge"]):
    """
    Compute Spearman and Pearson correlations between linguistic metrics and judgment scores
    (LLMs_judge and humans_judge), grouped by 'texts_from'.

    Parameters:
    - df: pandas DataFrame containing the data
    - linguistic_metrics: list of column names representing linguistic metrics
    - judgment_scores: list of judgment score columns to correlate against (default includes both)

    Output: Prints correlation statistics.
    """
    for metric in linguistic_metrics:
        print(f"\n===== Correlation of {metric} with Judgement Scores =====\n")
        for score_type in judgment_scores:
            print(f"→ Correlation between {metric} and {score_type} by 'texts_from':\n")
            for group in df["texts_from"].unique():
                subset = df[df["texts_from"] == group].dropna(subset=[metric, score_type])
                if len(subset) > 1:
                    try:
                        rho, spearman_p = spearmanr(subset[metric], subset[score_type])
                        r, pearson_p = pearsonr(subset[metric], subset[score_type])
                        print(f"{group}:")
                        print(f"  Spearman: rho = {rho:.2f}, p = {spearman_p:.4f}")
                        print(f"  Pearson:  r = {r:.2f}, p = {pearson_p:.4f}")
                        print(f"  Mean {metric:<18} = {subset[metric].mean():.2f}, SD = {subset[metric].std():.2f}")
                        print(f"  Mean {score_type:<18} = {subset[score_type].mean():.2f}, SD = {subset[score_type].std():.2f}\n")
                    except Exception as e:
                        print(f"{group}: Error computing correlation ({e})\n")
                else:
                    print(f"{group}: Not enough data for correlation.\n")

def plot_linguistic_metric_correlations(df, linguistic_metrics, judgment_scores=["LLMs_judge", "humans_judge"]):
    """
    Generate scatter plots and regression lines showing the relationship between
    linguistic metrics and judgment scores.
    """
    sns.set(style="whitegrid")

    for metric in linguistic_metrics:
        for score_type in judgment_scores:
            print(f"📊 Plotting {score_type} vs {metric}...\n")

            # ---------------- SCATTER PLOT ----------------
            plt.figure(figsize=(10, 6))
            ax = sns.scatterplot(
                data=df,
                x=metric,
                y=score_type,
                hue="texts_from",
                palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
                s=80,
                edgecolor='w'
            )
            plt.title(f"{score_type} vs {metric}", fontsize=14)
            plt.xlabel(metric)
            plt.ylabel(score_type)
            plt.legend(title="Texts from")
            plt.tight_layout()

            # Add model labels
            for _, row in df.dropna(subset=[metric, score_type]).iterrows():
                ax.text(row[metric] + 0.01, row[score_type] + 0.5, row["model_id"], fontsize=8)

            plt.show()

            # ---------------- REGRESSION PLOT ----------------
            sns.lmplot(
                data=df,
                x=metric,
                y=score_type,
                hue="texts_from",
                palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
                scatter_kws={'s': 70, 'edgecolor': 'w'},
                line_kws={'linewidth': 2},
                ci=95,
                height=6,
                aspect=1.3
            )
            plt.title(f"{score_type} vs {metric} with Linear Regression")
            plt.xlabel(metric)
            plt.ylabel(score_type)
            plt.tight_layout()
            plt.show()


### LLMs and Humans evaluation correlation ###

def compute_judgment_alignment(df):
    """
    Compute Pearson and Spearman correlation between LLMs_judge and humans_judge.
    Grouped by 'texts_from' and also overall.
    """
    print("\n===== Correlation between LLMs_judge and humans_judge =====\n")

    # Overall correlation
    df_filtered = df.dropna(subset=["LLMs_judge", "humans_judge"])
    if len(df_filtered) > 1:
        spearman_rho, spearman_p = spearmanr(df_filtered["LLMs_judge"], df_filtered["humans_judge"])
        pearson_r, pearson_p = pearsonr(df_filtered["LLMs_judge"], df_filtered["humans_judge"])
        print("Overall:")
        print(f"  Spearman: rho = {spearman_rho:.2f}, p = {spearman_p:.4f}")
        print(f"  Pearson:  r = {pearson_r:.2f}, p = {pearson_p:.4f}")
        print(f"  Mean LLMs_judge = {df_filtered['LLMs_judge'].mean():.2f}, SD = {df_filtered['LLMs_judge'].std():.2f}")
        print(f"  Mean humans_judge = {df_filtered['humans_judge'].mean():.2f}, SD = {df_filtered['humans_judge'].std():.2f}\n")
    
    # Group-wise correlation
    for group in df["texts_from"].unique():
        subset = df[df["texts_from"] == group].dropna(subset=["LLMs_judge", "humans_judge"])
        if len(subset) > 1:
            try:
                rho, spearman_p = spearmanr(subset["LLMs_judge"], subset["humans_judge"])
                r, pearson_p = pearsonr(subset["LLMs_judge"], subset["humans_judge"])
                print(f"{group}:")
                print(f"  Spearman: rho = {rho:.2f}, p = {spearman_p:.4f}")
                print(f"  Pearson:  r = {r:.2f}, p = {pearson_p:.4f}")
                print(f"  Mean LLMs_judge = {subset['LLMs_judge'].mean():.2f}, SD = {subset['LLMs_judge'].std():.2f}")
                print(f"  Mean humans_judge = {subset['humans_judge'].mean():.2f}, SD = {subset['humans_judge'].std():.2f}\n")
            except Exception as e:
                print(f"{group}: Error computing correlation ({e})\n")
        else:
            print(f"{group}: Not enough data for correlation.\n")

def plot_judgment_correlation(df):
    """
    Plot LLMs_judge vs. humans_judge with regression line.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="LLMs_judge",
        y="humans_judge",
        hue="texts_from",
        palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
        s=80,
        edgecolor='w'
    )
    plt.title("LLMs_judge vs Humans_judge")
    plt.xlabel("LLMs_judge")
    plt.ylabel("humans_judge")
    plt.legend(title="Texts from")
    plt.tight_layout()
    plt.show()

    sns.lmplot(
        data=df,
        x="LLMs_judge",
        y="humans_judge",
        hue="texts_from",
        palette={"Linguistics": "#1f77b4", "Computer Science": "#ff7f0e"},
        scatter_kws={'s': 70, 'edgecolor': 'w'},
        line_kws={'linewidth': 2},
        ci=95,
        height=6,
        aspect=1.3
    )
    plt.title("LLMs_judge vs Humans_judge with Linear Regression")
    plt.tight_layout()
    plt.show()



# Define your list of linguistic metrics

# linguistic_metrics = ["token_count", "cosine_similarity", "flesch_reading_ease", "flesch_kincaid_grade", "smog_index", "bleu_score", "rouge_1", "rouge_2", "rouge_L", "bertscore_precision", "bertscore_recall", "bertscore_f1", "JJ", "NN", "VB", "DT", "IN", "MD", "CC", "VBG", "NNP", "RB", "VBN"]  # all metrics
linguistic_metrics = ["token_count", "NN", "VB", "IN","VBN"]
# linguistic_metrics = ["JJ", "NN", "VB", "DT", "IN", "MD", "CC", "VBG", "NNP", "RB", "VBN"]  # these are the most significant POS tags

compute_linguistic_metric_correlations(df, linguistic_metrics)
plot_linguistic_metric_correlations(df, linguistic_metrics)

# compute_judgment_alignment(df)
# plot_judgment_correlation(df)
