from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from llm_caller import call_llm
from prompts import *  # Import all prompts
from prompts import PROMPT_FUNCTIONS
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH

from openai.types.chat import ChatCompletion

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
    if text is None:
        return False

    text = text.strip()

    if text in {"", "FAILED", "None", "#"}:
        return False

    if len(text) < min_char_count:
        return False  # Too short, e.g., just "In"

    if len(text.split()) < min_word_count:
        return False

    if has_long_dash_run(text, threshold=dash_threshold):
        return False

    suspicious_endings = {"...", "END OF OUTPUT", "### END OF INPUT ###"}
    if any(text.endswith(marker) for marker in suspicious_endings):
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
    if any(text.strip().endswith(marker) for marker in {"...", "END OF OUTPUT", "### END OF INPUT ###"}):
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

def clean_response_text(text):
    if "Original text:" in text:
        text = text.split("Original text:", 1)[-1].strip()
    if "### END OF INPUT ###" in text:
        text = text.split("### END OF INPUT ###", 1)[-1].strip()
    return text

def create_benchmark(llm_model, prompt_function_name, max_entries=None, target_key=None):
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
                print(f"Skipping {target_category} for {key}, already exists and passed validation.")
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
                response_text = clean_response_text(response_text)

                if is_valid_existing_text(response_text):
                    print(f"✅ Text is valid on attempt {attempt+1}")
                    break
                else:
                    reason = get_invalid_reason(response_text)
                    print(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed due to: {reason}")

            # else:
            #     print(f"❌ All attempts failed — keeping null for {key}-{target_category}-{prompt_key}")
            #     response_text = None

            if response_text:
                tailored_texts[prompt_key] = response_text
                tailored_analysis = analyze_text(response_text)
                linguistic_analysis[key].setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})[prompt_key] = {
                    "text": response_text,
                    "token_count": tailored_analysis["token_count"],
                    "readability": tailored_analysis["readability"],
                    "pos": tailored_analysis["pos"]
                }
                print(f"Successfully generated {target_category} for {key} (Tokens: {tailored_analysis['token_count']})")
            else:
                print(f"Warning: Generated text is invalid for {target_category} in entry {key}. Keeping null.")
                tailored_texts[prompt_key] = None

    save_dataset(benchmark, BENCHMARK_PATH)
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)
    print(f"\nBenchmark updated and saved (Processed {min(len(dataset), max_entries or len(dataset))} entries).")