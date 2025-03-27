import json
import re
from statistics import mean
from llm_caller import call_llm
from tqdm import tqdm
import time
import sys
from config import LINGUISTIC_ANALYSIS_PATH


# Prompt to detect hallucinations 

detect_hallucination_prompt = """
You are an expert judge tasked with evaluating the faithfulness of AI-generated tailored texts for difference target audiences from the original text, originally aimed to a specific target audience. 
Analyze the provided original texts and tailored texts to determine if the tailored texts contain any hallucinations or unfaithful information.

Relevance: The rating measures how well the tailored text captures the key points of the original text. Consider
whether all and only the important aspects are contained in the tailored text.
Consistency: The rating measures whether the facts in the tailored text are consistent with the facts in the
original text. Consider whether the tailored text does reproduce all facts accurately and does not make up
untrue information.

Guidelines:
1. The tailored texts must not introduce new information beyond what's provided in the original text.
2. The tailored texts must not contradict any information given in the original text.
2. The tailored texts should not contradict well-established facts or general knowledge.
3. Ignore the original text when evaluating faithfulness; it's provided for context only.
4. Consider partial hallucinations where some information is correct but other parts are not.
5. Pay close attention to the subject of statements. Ensure that attributes, actions, or dates are correctly associated with the right entities.
6. Be vigilant for subtle misattributions or conflations of information, even if the date or other details are correct.
7. Check that the tailored texts do not oversimplify or generalize information in a way that changes its meaning or accuracy.
8. Keep in mind the definitions of relevance and consistency above.
9. Analyze the tailored texts thoroughly and assign a relevance and consistency scores on a 1 to 5 likert-scale, where:
    - 1: The tailored text is entirely unfaithful to the original text;
    - 5: The original text is entirely faithful to the tailored text.

Original Text:
\"\"\"
{original}
\"\"\"

Tailored Text:
\"\"\"
{tailored}
\"\"\"

Please return the result clearly in the format:
Relevance: <score>
Consistency: <score>
"""

# Toggle for test mode
TEST_MODE = False

# Load the JSON data
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if TEST_MODE:
    # ✅ Test mode: example: entry 1, prompt1, mistral only
    # ENTRY
    entry_id = list(data.keys())[0]
    entry = data[entry_id]
    original_text = entry["original_text"]
    # MODEL
    model = "deepseek"
    for user_category, prompts in entry.get("tailored_texts", {}).get(model, {}).items():
        # PROMPT
        if "prompt1" in prompts:
            prompt_data = prompts["prompt1"]
            tailored_text = prompt_data["text"]
            prompt = detect_hallucination_prompt.format(
                original=original_text.strip(),
                tailored=tailored_text.strip()
            )
            response = call_llm(model, prompt)
            relevance = consistency = None
            if response:
                relevance_match = re.search(r"Relevance\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                consistency_match = re.search(r"Consistency\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                if relevance_match and consistency_match:
                    relevance = float(relevance_match.group(1))
                    consistency = float(consistency_match.group(1))
                    average = round((relevance + consistency) / 2, 2)
                    prompt_data.setdefault("hallucination_scores", {})[model] = {
                        "relevance": relevance,
                        "consistency": consistency,
                        "average": average
                    }

else:
    # Models to evaluate
    llms = ["mistral", "llama", "gpt4o", "claude", "deepseek"]

    print(f"🚀 Starting hallucination scoring on {len(data)} entries...")

    total_scored = 0

    for entry_id, entry in tqdm(data.items(), desc="Scoring entries"):
        original_text = entry.get("original_text", "").strip()
        hallucination_scores = {}
        all_avg_scores = []

        for model in llms:
            model_data = entry.get("tailored_texts", {}).get(model)
            if not model_data:
                tqdm.write(f"⚠️ No tailored texts for model {model} in entry {entry_id}")
                continue

            model_scores = []

            for user_category, prompts in model_data.items():
                if not isinstance(prompts, dict):
                    continue

                for prompt_id, prompt_data in prompts.items():
                    if model in prompt_data.get("hallucination_scores", {}):
                        continue  # Already scored

                    tailored_text = prompt_data.get("text", "").strip()
                    if not tailored_text:
                        tqdm.write(f"⚠️ Missing tailored text for {entry_id} → {model} → {user_category} → {prompt_id}")
                        continue

                    tqdm.write(f"📍 Scoring {entry_id} → {model} → {user_category} → {prompt_id}")
                    prompt = detect_hallucination_prompt.format(
                        original=original_text,
                        tailored=tailored_text
                    )

                    try:
                        response = call_llm(model, prompt)
                    except Exception as e:
                        tqdm.write(f"🔥 Error calling LLM '{model}' for {entry_id} → {prompt_id}: {e}")
                        continue

                    if not response:
                        tqdm.write(f"⚠️ No response from {model} for {entry_id} → {prompt_id}")
                        continue

                    relevance_match = re.search(r"Relevance\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                    consistency_match = re.search(r"Consistency\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)

                    if relevance_match and consistency_match:
                        relevance = float(relevance_match.group(1))
                        consistency = float(consistency_match.group(1))
                        average = round((relevance + consistency) / 2, 2)

                        prompt_data.setdefault("hallucination_scores", {})[model] = {
                            "relevance": relevance,
                            "consistency": consistency,
                            "average": average
                        }

                        model_scores.append(average)
                        total_scored += 1
                        tqdm.write(f"✅ Scored {entry_id} → {model} → {user_category} → {prompt_id} → avg: {average}")
                    else:
                        tqdm.write(f"⚠️ Failed to parse scores from {model} for {entry_id} → {prompt_id}:\n{response}")

            # Aggregate per-model average if available
            if model_scores:
                relevance_vals = []
                consistency_vals = []

                for prompts in entry["tailored_texts"][model].values():
                    for p in prompts.values():
                        scores = p.get("hallucination_scores", {}).get(model)
                        if scores:
                            relevance_vals.append(scores["relevance"])
                            consistency_vals.append(scores["consistency"])

                hallucination_scores[model] = {
                    "relevance": round(mean(relevance_vals), 2),
                    "consistency": round(mean(consistency_vals), 2),
                    "average": round(mean(model_scores), 2)
                }

                all_avg_scores.append(hallucination_scores[model]["average"])

        # Update entry-level scores
        entry["hallucination_scores"] = hallucination_scores
        entry["overall_hallucination_average"] = round(mean(all_avg_scores), 2) if all_avg_scores else None

        tqdm.write(f"📦 Finished entry {entry_id}")

    # Save updated file
    with open(LINGUISTIC_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Hallucination scoring complete. Total prompts scored: {total_scored}")


"""
### DOESN'T WORK ###
else:
    llms = ["mistral", "llama", "gpt4o", "claude", "deepseek"]
    print(f"🚀 Starting hallucination scoring on {len(data)} entries...")


    for entry_id, entry in tqdm(data.items(), desc="Scoring entries"):
        print(f"➡️ Processing entry: {entry_id}")
        sys.stdout.flush()

        original_text = entry.get("original_text", "").strip()
        hallucination_scores = {}
        all_avg_scores = []

        for model in llms:
            model_data = entry.get("tailored_texts", {}).get(model)
            if not model_data:
                continue

            model_scores = []

            for user_category, prompts in model_data.items():
                for prompt_id, prompt_data in prompts.items():
                    # Skip if already scored
                    if model in prompt_data.get("hallucination_scores", {}):
                        continue

                    tailored_text = prompt_data.get("text", "").strip()
                    if not tailored_text:
                        tqdm.write(f"⚠️ Missing tailored text for {entry_id} → {model} → {user_category} → {prompt_id}")
                        continue

                    prompt = detect_hallucination_prompt.format(
                        original=original_text,
                        tailored=tailored_text
                    )

                    try:
                        response = call_llm(model, prompt)
                    except Exception as e:
                        tqdm.write(f"🔥 Error calling LLM '{model}' for entry '{entry_id}', prompt '{prompt_id}': {e}")
                        continue

                    if not response:
                        tqdm.write(f"⚠️ No response from {model} for {entry_id} → {prompt_id}")
                        continue

                    # Extract hallucination scores
                    relevance_match = re.search(r"Relevance\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                    consistency_match = re.search(r"Consistency\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)

                    if relevance_match and consistency_match:
                        relevance = float(relevance_match.group(1))
                        consistency = float(consistency_match.group(1))
                        average = round((relevance + consistency) / 2, 2)

                        prompt_data.setdefault("hallucination_scores", {})[model] = {
                            "relevance": relevance,
                            "consistency": consistency,
                            "average": average
                        }

                        model_scores.append(average)
                    else:
                        tqdm.write(f"⚠️ Failed to parse scores from {model} response for {entry_id} → {prompt_id}:\n{response}")

            if model_scores:
                # Compute per-model average scores
                relevance_vals = []
                consistency_vals = []

                for prompts in entry["tailored_texts"][model].values():
                    for p in prompts.values():
                        scores = p.get("hallucination_scores", {}).get(model)
                        if scores:
                            relevance_vals.append(scores["relevance"])
                            consistency_vals.append(scores["consistency"])

                hallucination_scores[model] = {
                    "relevance": round(mean(relevance_vals), 2),
                    "consistency": round(mean(consistency_vals), 2),
                    "average": round(mean(model_scores), 2)
                }

                all_avg_scores.append(hallucination_scores[model]["average"])

        entry["hallucination_scores"] = hallucination_scores
        entry["overall_hallucination_average"] = round(mean(all_avg_scores), 2) if all_avg_scores else None

        tqdm.write(f"✅ Processed entry {entry_id}")

    # Save updated file
    with open(LINGUISTIC_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Hallucination scoring complete.")

# Save updated file
with open(LINGUISTIC_ANALYSIS_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Hallucination scoring complete.")"""
