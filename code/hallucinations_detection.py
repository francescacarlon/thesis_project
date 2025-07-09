"""
Multi‑Model Hallucination Scoring Script

This script evaluates tailored (paraphrased) texts for hallucinations by
comparing them with their original source texts.  For each paraphrase, it asks
several LLMs to rate:

• **Relevance** (1 – 5): how well the tailored text covers the key points  
• **Consistency** (1 – 5): factual faithfulness to the original text  

The script:

1. Loads the project‑wide `linguistic_analysis.json` file (path set in
   `config.py`).
2. Iterates through every tailored text (or a single example in `TEST_MODE`).
3. Sends a hallucination‑detection prompt to each evaluator LLM.
4. Extracts numeric scores with regex, stores them under
   `prompt_data["hallucination_scores"][eval_model]`.
5. Computes an average per evaluator **and** an overall average per prompt.
6. Saves progress after each entry; final results go to the same JSON file.

Key toggles & inputs
--------------------
• `TEST_MODE` – when `True`, scores a single hard‑coded entry for quick checks.  
• `llms` – list of generator/evaluator model names supported by `call_llm()`.  
• Requires valid API keys / env variables for OpenAI, Anthropic, and Mistral.

Outputs
-------
Adds this block to every prompt that gets scored:

"hallucination_scores": {
  "gpt4o":  { "relevance": 4, "consistency": 5, "average": 4.5 },
  "claude": { "relevance": 3, "consistency": 4, "average": 3.5 },
  …
  "hallucinations_overall_average": 4.00
}
"""

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
You are an expert judge tasked with evaluating the faithfulness of AI-generated tailored texts for different target audiences from the original text, originally aimed to a specific target audience. 
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

# Models to evaluate
llms = ["mistral", "llama", "gpt4o", "claude", "deepseek"]

if TEST_MODE:

    entry_id = list(data.keys())[1]
    entry = data[entry_id]
    original_text = entry["original_text"]

    print(f" Running TEST_MODE on entry: {entry_id}")

    for gen_model in llms:
        model_data = entry.get("tailored_texts", {}).get(gen_model)
        if not model_data:
            print(f" No tailored texts for generator model {gen_model} in entry {entry_id}")
            continue

        for user_category, prompts in model_data.items():
            for prompt_id, prompt_data in prompts.items():
                if prompt_id != "prompt2":
                    continue

                for eval_model in llms:
                    # Skip if already scored by this evaluator
                    if eval_model in prompt_data.get("hallucination_scores", {}):
                        print(f" Already scored by {eval_model}: {gen_model} → {user_category} → {prompt_id}")
                        continue

                    tailored_text = prompt_data.get("text", "").strip()
                    if not tailored_text:
                        print(f" Missing tailored text for {entry_id} → {gen_model} → {user_category} → {prompt_id}")
                        continue

                    print(f"Evaluating {entry_id} → Gen: {gen_model} → Eval: {eval_model} → {user_category} → {prompt_id}")
                    prompt = detect_hallucination_prompt.format(
                        original=original_text,
                        tailored=tailored_text
                    )

                    try:
                        response = call_llm(eval_model, prompt)
                    except Exception as e:
                        print(f" Error calling LLM '{eval_model}' for {entry_id} → {prompt_id}: {e}")
                        continue

                    if not response:
                        print(f" No response from {eval_model} for {entry_id} → {prompt_id}")
                        continue

                    relevance_match = re.search(r"Relevance\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                    consistency_match = re.search(r"Consistency\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)

                    if relevance_match and consistency_match:
                        relevance = float(relevance_match.group(1))
                        consistency = float(consistency_match.group(1))
                        average = round((relevance + consistency) / 2, 2)

                        prompt_data.setdefault("hallucination_scores", {})[eval_model] = {
                            "relevance": relevance,
                            "consistency": consistency,
                            "average": average
                        }

                        print(f" Scored by {eval_model}: relevance={relevance}, consistency={consistency}, avg={average}")
                    else:
                        print(f" Failed to parse scores from {eval_model} for {entry_id} → {prompt_id}:\n{response}")

                #  Compute hallucinations_overall_average after all evaluators have scored this prompt
                avg_scores = [
                    s["average"]
                    for s in prompt_data.get("hallucination_scores", {}).values()
                    if isinstance(s, dict) and "average" in s
                ]

                if avg_scores:
                    prompt_data["hallucination_scores"]["hallucinations_overall_average"] = round(mean(avg_scores), 2)
                    print(f" Overall average for {gen_model} → {user_category} → {prompt_id}: {prompt_data['hallucination_scores']['hallucinations_overall_average']}")


    # Overwrite full data back to main file
    with open(LINGUISTIC_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" Test results written to {LINGUISTIC_ANALYSIS_PATH}")


else:
    print(f" Starting full hallucination scoring on {len(data)} entries...\n")

    total_scored = 0

    for entry_id, entry in tqdm(data.items(), desc="Scoring entries"):
        original_text = entry.get("original_text", "").strip()

        for gen_model in llms:
            model_data = entry.get("tailored_texts", {}).get(gen_model)
            if not model_data:
                tqdm.write(f" No tailored texts for generator model {gen_model} in entry {entry_id}")
                continue

            for user_category, prompts in model_data.items():
                for prompt_id, prompt_data in prompts.items():
                    tailored_text = prompt_data.get("text", "").strip()
                    if not tailored_text:
                        tqdm.write(f" Missing tailored text for {entry_id} → {gen_model} → {user_category} → {prompt_id}")
                        continue

                    for eval_model in llms:
                        if eval_model in prompt_data.get("hallucination_scores", {}):
                            continue  # Already scored by this model

                        tqdm.write(f" Scoring {entry_id} → Gen: {gen_model} → Eval: {eval_model} → {user_category} → {prompt_id}")

                        prompt = detect_hallucination_prompt.format(
                            original=original_text,
                            tailored=tailored_text
                        )

                        try:
                            response = call_llm(eval_model, prompt)
                        except Exception as e:
                            tqdm.write(f" Error calling LLM '{eval_model}' for {entry_id} → {prompt_id}: {e}")
                            continue

                        if not response:
                            tqdm.write(f" No response from {eval_model} for {entry_id} → {prompt_id}")
                            continue

                        relevance_match = re.search(r"Relevance\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)
                        consistency_match = re.search(r"Consistency\s*[:=]\s*(\d(?:\.\d)?)", response, re.IGNORECASE)

                        if relevance_match and consistency_match:
                            relevance = float(relevance_match.group(1))
                            consistency = float(consistency_match.group(1))
                            average = round((relevance + consistency) / 2, 2)

                            prompt_data.setdefault("hallucination_scores", {})[eval_model] = {
                                "relevance": relevance,
                                "consistency": consistency,
                                "average": average
                            }

                            total_scored += 1
                            tqdm.write(f" Scored → relevance={relevance}, consistency={consistency}, avg={average}")
                        else:
                            tqdm.write(f" Failed to parse scores from {eval_model} for {entry_id} → {prompt_id}:\n{response}")

                    # After all eval models scored this prompt → compute overall average
                    avg_scores = [
                        s["average"]
                        for s in prompt_data.get("hallucination_scores", {}).values()
                        if isinstance(s, dict) and "average" in s
                    ]
                    if avg_scores:
                        prompt_data["hallucination_scores"]["hallucinations_overall_average"] = round(mean(avg_scores), 2)
                        tqdm.write(
                            f" Overall average for {entry_id} → {gen_model} → {user_category} → {prompt_id}: "
                            f"{prompt_data['hallucination_scores']['hallucinations_overall_average']}"
                        )

        # Save progress after each entry
        with open(LINGUISTIC_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            tqdm.write(f"Progress saved after entry {entry_id}")

    # ✅ Final message
    print(f"\n Hallucination scoring complete. Total prompts scored: {total_scored}")
