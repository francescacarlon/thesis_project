# updates for benchmark file(s)

from utils import load_dataset, save_dataset
from config import LINGUISTIC_ANALYSIS_PATH, BASE_PATH
from linguistic_analysis import compute_readability
import json
import os

def update_only_readability(linguistic_analysis):
    """Updates only the readability field (incl. SMOG) for all tailored texts."""
    updated = 0

    for key, entry in linguistic_analysis.items():
        # Update original text readability
        if "original_text" in entry:
            original_text = entry["original_text"]
            updated_readability = compute_readability(original_text)
            entry["readability"] = updated_readability

        # Update all tailored texts' readability
        for llm, categories in entry.get("tailored_texts", {}).items():
            for category, prompts in categories.items():
                for prompt_key, analysis in prompts.items():
                    if "text" in analysis:
                        text = analysis["text"]
                        updated_readability = compute_readability(text)
                        analysis["readability"] = updated_readability
                        updated += 1

    print(f"\n✅ Readability scores updated (including SMOG) for {updated} tailored texts.")

# if __name__ == "__main__":
    # print("📂 Loading linguistic analysis data...")
    # linguistic_analysis = load_dataset(LINGUISTIC_ANALYSIS_PATH)
# 
    # print("🔧 Updating readability scores...")
    # update_only_readability(linguistic_analysis)
# 
    # print("💾 Saving updated dataset...")
    # save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)
# 
    # print("✅ Done. SMOG scores are now included.")


# update LLM-as-a-judge and human evaluation scores in benchmark_randomized_with_agreement_scores.json

# Step 1: Define the benchmark path
final_benchmark_path = os.path.join(BASE_PATH, 'data/benchmark_randomized_with_agreement_scores.json')

if __name__ == "__main__":
    # 📂 Load benchmark JSON
    print("📂 Loading benchmark file...")
    with open(final_benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 📝 Insert judge scores
    data['T2']['original_text_2_score'] = {
        "LLMs_judge": 21,
        "humans_judge": 0
    }
    data['T2']['selected_texts']['CS']['CS_gpt4o_prompt3']['LLMs_judge'] = 33
    data['T2']['selected_texts']['CS']['CS_gpt4o_prompt3']['humans_judge'] = 0
    data['T2']['selected_texts']['CS']['CS_claude_prompt4']['LLMs_judge'] = 36
    data['T2']['selected_texts']['CS']['CS_claude_prompt4']['humans_judge'] = 0

    data['T3']['original_text_3_score'] = {
        "LLMs_judge": 20,
        "humans_judge": 0
    }
    data['T3']['selected_texts']['CS']['CS_mistral_prompt4']['LLMs_judge'] = 27
    data['T3']['selected_texts']['CS']['CS_mistral_prompt4']['humans_judge'] = 0
    data['T3']['selected_texts']['CS']['CS_gpt4o_prompt1']['LLMs_judge'] = 43
    data['T3']['selected_texts']['CS']['CS_gpt4o_prompt1']['humans_judge'] = 0

    data['T4']['original_text_4_score'] = {
        "LLMs_judge": 20,
        "humans_judge": 0
    }
    data['T4']['selected_texts']['CS']['CS_claude_prompt4']['LLMs_judge'] = 42
    data['T4']['selected_texts']['CS']['CS_claude_prompt4']['humans_judge'] = 0
    data['T4']['selected_texts']['CS']['CS_mistral_prompt5']['LLMs_judge'] = 28
    data['T4']['selected_texts']['CS']['CS_mistral_prompt5']['humans_judge'] = 0


    data['T7']['original_text_7_score'] = {
        "LLMs_judge": 16,
        "humans_judge": 0
    }
    data['T7']['selected_texts']['L']['L_mistral_prompt1']['LLMs_judge'] = 30
    data['T7']['selected_texts']['L']['L_mistral_prompt1']['humans_judge'] = 0
    data['T7']['selected_texts']['L']['L_deepseek_prompt3']['LLMs_judge'] = 44
    data['T7']['selected_texts']['L']['L_deepseek_prompt3']['humans_judge'] = 0


    data['T8']['original_text_8_score'] = {
    "LLMs_judge": 23,
    "humans_judge": 0
    }
    data['T8']['selected_texts']['L']['L_mistral_prompt5']['LLMs_judge'] = 26
    data['T8']['selected_texts']['L']['L_mistral_prompt5']['humans_judge'] = 0
    data['T8']['selected_texts']['L']['L_gpt4o_prompt3']['LLMs_judge'] = 41
    data['T8']['selected_texts']['L']['L_gpt4o_prompt3']['humans_judge'] = 0


    data['T10']['original_text_10_score'] = {
    "LLMs_judge": 16,
    "humans_judge": 0
    }
    data['T10']['selected_texts']['L']['L_gpt4o_prompt2']['LLMs_judge'] = 34
    data['T10']['selected_texts']['L']['L_gpt4o_prompt2']['humans_judge'] = 0
    data['T10']['selected_texts']['L']['L_claude_prompt5']['LLMs_judge'] = 40
    data['T10']['selected_texts']['L']['L_claude_prompt5']['humans_judge'] = 0

    # 💾 Save updated benchmark
    print("💾 Saving updated benchmark...")
    with open(final_benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ Benchmark file updated successfully.")


