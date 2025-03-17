from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text, compute_cosine_similarity
from llm_caller import call_llm
from prompts import *  # Import all prompts
from prompts import PROMPT_FUNCTIONS
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH

from openai.types.chat import ChatCompletion
import re

def has_long_dash_run(text, threshold=20):
    max_run = 0
    current_run = 0
    for char in text:
        if char == '-':
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run >= threshold

def is_valid_existing_text(text, dash_threshold=20, min_word_count=100, min_char_count=30):
    if text is None or text.strip() in {"", "FAILED", "None", "#"}:
        return False
    text = text.strip()
    if len(text) < min_char_count or len(text.split()) < min_word_count:
        return False
    if has_long_dash_run(text, threshold=dash_threshold):
        return False
    if any(text.endswith(marker) for marker in {"...", "END OF OUTPUT", "### END OF INPUT ###", "### END OF FILE ###"}):
        return False
    if "Original text:" in text and text.count("Original text:") > 1:
        return False
    return True

def get_invalid_reason(text):
    if text is None or text.strip() in {"", "FAILED", "None"}:
        return "text is empty, marked as FAILED, or None"
    if len(text.strip()) < 30:
        return "text is too short (fewer than 30 characters)"
    if len(text.split()) < 100:
        return "text is too short (<100 words)"
    if has_long_dash_run(text):
        return "text contains a long run of dashes"
    if any(text.strip().endswith(marker) for marker in {"...", "END OF OUTPUT", "### END OF INPUT ###", "### END OF FILE ###"}):
        return "text ends with suspicious marker"
    if "Original text:" in text and text.count("Original text:") > 1:
        return "text contains repeated 'Original text:' marker"
    return "unknown validation failure"

def extract_text_from_llm_response(response):
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, ChatCompletion):
        return response.choices[0].message.content.strip()
    if isinstance(response, dict):
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"].strip()
        if "generated_text" in response:
            return response["generated_text"].strip()
    raise ValueError(f"Unexpected response format: {response}")

def clean_text(text):
    patterns = [
        r"(?i)###\s*END OF OUTPUT.*",
        r"(?i)###\s*END OF FILE.*",
        r"(?i)###\s*END OF INPUT.*",
        r"(?i)\[System note.*?\]",
        r"Original text:.*?",
        r"### END OF INPUT ###.*?",
        r"### END OF OUTPUT ###.*?",
        r"### END OF FILE ###.*?",
        r"(?:```python\s*```)?\s*# No code in this file(?:\n|.)*?```(?:\s*```python\s*```)?"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^-{5,}", "", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_existing_texts(benchmark):
    cleaned_count = 0
    for key, entry in benchmark.items():
        for llm, categories in entry.get("tailored_texts", {}).items():
            for category, prompts in categories.items():
                for prompt_key, text in prompts.items():
                    cleaned_text = clean_text(text)
                    if cleaned_text != text:
                        cleaned_count += 1
                    benchmark[key]["tailored_texts"][llm][category][prompt_key] = cleaned_text
    print(f"\n✅ Post-run cleanup applied to {cleaned_count} texts in benchmark.")

def clean_existing_texts_in_linguistic_analysis(linguistic_analysis, benchmark):
    """Cleans up and reanalyzes texts in linguistic_analysis, ensuring cosine similarity is computed for all."""
    cleaned_count = 0
    updated_similarities = 0

    for key, entry in linguistic_analysis.items():
        original_text = entry["original_text"]

        for llm, categories in entry.get("tailored_texts", {}).items():
            for category, prompts in categories.items():
                for prompt_key, analysis in prompts.items():
                    if "text" in analysis:
                        cleaned_text = clean_text(analysis["text"])
                        updated_analysis = analyze_text(cleaned_text)

                        if cleaned_text != analysis["text"]:
                            cleaned_count += 1

                        # Compute cosine similarity if missing
                        if "cosine_similarity" not in analysis:
                            cosine_sim = compute_cosine_similarity(original_text, cleaned_text)["cosine_similarity"]
                            updated_similarities += 1
                        else:
                            cosine_sim = analysis["cosine_similarity"]

                        linguistic_analysis[key]["tailored_texts"][llm][category][prompt_key] = {
                            "text": cleaned_text,
                            "token_count": updated_analysis["token_count"],
                            "readability": updated_analysis["readability"],
                            "pos": updated_analysis["pos"],
                            "cosine_similarity": cosine_sim
                        }

    print(f"\n✅ Post-run cleanup applied to {cleaned_count} texts in linguistic_analysis.")
    print(f"\n✅ Cosine similarity computed/updated for {updated_similarities} text pairs.")


def create_benchmark(llm_model, prompt_function_name, max_entries=None, target_key=None):
    """Creates a benchmark, generating new tailored texts and updating linguistic analysis with cosine similarity."""
    
    dataset = load_dataset(DATASET_PATH)

    try:
        benchmark = load_dataset(BENCHMARK_PATH)
    except FileNotFoundError:
        benchmark = {}

    try:
        linguistic_analysis = load_dataset(LINGUISTIC_ANALYSIS_PATH)
    except FileNotFoundError:
        linguistic_analysis = {}

    TAILORING_MATRIX = {
        "L": ["CS", "CL"],
        "CS": ["L", "CL"],
        "CL": ["L", "CS"]
    }

    prompt_number = next((key for key, value in PROMPT_FUNCTIONS.items() if value == prompt_function_name), None)
    if prompt_number is None:
        raise ValueError(f"Prompt function '{prompt_function_name}' is not in PROMPT_FUNCTIONS.")

    prompt_key = f"prompt{prompt_number}"

    for i, (key, value) in enumerate(dataset.items()):
        if max_entries and i >= max_entries:
            break

        if target_key and key != target_key:
            continue

        print(f"\nProcessing entry {i+1}/{max_entries or len(dataset)}: {key}")

        original_text = value["original_text"]
        original_category = value["original_category"]

        benchmark.setdefault(key, {
            "chapter": value["chapter"],
            "sections": value["sections"],
            "topic": value["topic"],
            "original_category": original_category,
            "original_text": original_text,
            "tailored_texts": {}
        })

        original_analysis = analyze_text(original_text)
        linguistic_analysis.setdefault(key, {
            "original_category": original_category,
            "original_text": original_text,
            "token_count": original_analysis["token_count"],
            "readability": original_analysis["readability"],
            "pos": original_analysis["pos"],
            "tailored_texts": {}
        })

        for target_category in TAILORING_MATRIX.get(original_category, []):
            tailored_texts = benchmark[key].setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})
            existing_text = (tailored_texts.get(prompt_key) or "").strip()

            if is_valid_existing_text(existing_text):
                print(f"Skipping generation for {target_category} in {key}, already exists and passed validation.")

                # Ensure cosine similarity is computed if missing
                existing_analysis = linguistic_analysis[key]["tailored_texts"].get(llm_model, {}).get(target_category, {}).get(prompt_key, {})
                if "cosine_similarity" not in existing_analysis:
                    print(f"Computing cosine similarity for {target_category} in {key} (existing text).")
                    cosine_sim = compute_cosine_similarity(original_text, existing_text)["cosine_similarity"]
                    linguistic_analysis[key].setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})[prompt_key] = {
                        "text": existing_text,
                        "cosine_similarity": cosine_sim
                    }
                continue

            reason = get_invalid_reason(existing_text)
            print(f"Regenerating {target_category} for {key} because {reason}.")

            prompt_function = globals().get(prompt_function_name)
            if not prompt_function:
                raise ValueError(f"Prompt function '{prompt_function_name}' does not exist.")

            prompt = prompt_function(target_category, original_text)

            MAX_RETRIES = 3

            for attempt in range(MAX_RETRIES):
                response = call_llm(llm_model, prompt)
                response_text = extract_text_from_llm_response(response)
                response_text = clean_text(response_text)

                if is_valid_existing_text(response_text):
                    break

            tailored_texts[prompt_key] = response_text
            tailored_analysis = analyze_text(response_text)

            # Compute cosine similarity for the newly generated text
            cosine_sim = compute_cosine_similarity(original_text, response_text)["cosine_similarity"]

            linguistic_analysis[key].setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})[prompt_key] = {
                "text": response_text,
                "token_count": tailored_analysis["token_count"],
                "readability": tailored_analysis["readability"],
                "pos": tailored_analysis["pos"],
                "cosine_similarity": cosine_sim
            }

    clean_existing_texts(benchmark)
    clean_existing_texts_in_linguistic_analysis(linguistic_analysis, benchmark)

    save_dataset(benchmark, BENCHMARK_PATH)
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)
