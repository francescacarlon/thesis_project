# update_smog_only.py

from utils import load_dataset, save_dataset
from config import LINGUISTIC_ANALYSIS_PATH
from linguistic_analysis import compute_readability

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

if __name__ == "__main__":
    print("📂 Loading linguistic analysis data...")
    linguistic_analysis = load_dataset(LINGUISTIC_ANALYSIS_PATH)

    print("🔧 Updating readability scores...")
    update_only_readability(linguistic_analysis)

    print("💾 Saving updated dataset...")
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)

    print("✅ Done. SMOG scores are now included.")

