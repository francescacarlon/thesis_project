from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from llm_caller import call_llm
from prompts import *  # Import all prompts
from prompts import PROMPT_FUNCTIONS  # Ensure PROMPT_FUNCTIONS is imported
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH

def has_long_dash_run(text, threshold=20):
    """
    Checks if the text contains a run of dashes that is >= threshold length.
    This helps detect invalid or corrupted LLM responses.
    """
    max_run = 0
    current_run = 0

    for char in text:
        if char == '-':
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run >= threshold

def is_valid_existing_text(text, dash_threshold=20):
    """
    Checks if the existing text is valid.
    A valid text:
    - is not empty
    - is not the special "FAILED" marker
    - is not the string "None"
    - does not contain long runs of dashes
    """
    if not text or text.strip() in {"", "FAILED", "None"}:
        return False  # Empty, failed, or explicitly broken is invalid
    if has_long_dash_run(text, threshold=dash_threshold):
        return False  # Contains too many dashes (likely corrupted)
    return True


def create_benchmark(llm_model, prompt_function_name, max_entries=None, target_key=None):
    """
    Generates tailored paraphrases using the selected LLM and prompt,
    and performs linguistic analysis for each entry in the dataset.
    """

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
        raise ValueError(f"❌ Error: Prompt function '{prompt_function_name}' is not in PROMPT_FUNCTIONS.")
    prompt_key = f"prompt{prompt_number}"

    for i, (key, value) in enumerate(dataset.items()):
        if max_entries and i >= max_entries:
            break

        if target_key and key != target_key:  # Add this line to skip unwanted entries
            continue

        print(f"\n⚡ Processing entry {i+1}/{max_entries or len(dataset)}: {key}")

        original_text = value["original_text"]
        original_category = value["original_category"]

        if key not in benchmark:
            benchmark[key] = {
                "chapter": value["chapter"],
                "sections": value["sections"],
                "topic": value["topic"],
                "original_category": original_category,
                "original_text": original_text,
                "tailored_texts": {}
            }

        original_analysis = analyze_text(original_text)
        linguistic_analysis[key] = {
            "original_category": original_category,
            "original_text": original_text,
            "token_count": original_analysis["token_count"],
            "readability": original_analysis["readability"],
            "pos": original_analysis["pos"],
            "tailored_texts": linguistic_analysis.get(key, {}).get("tailored_texts", {})
        }

        for target_category in TAILORING_MATRIX.get(original_category, []):
            benchmark.setdefault(key, {}).setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})

            existing_text = benchmark[key]["tailored_texts"][llm_model][target_category].get(prompt_key, "").strip()

            if is_valid_existing_text(existing_text):
                print(f"⏭️ Skipping {target_category} for {key}, already exists and passed validation.")
                continue
            else:
                print(f"♻️ Regenerating {target_category} for {key} because it failed validation (empty, failed, or has long dashes).")

            print(f"\n📝 Generating tailored text for {target_category} in entry {key} using {llm_model} with {prompt_key}...")

            try:
                prompt_function = globals().get(prompt_function_name)
                if not prompt_function or not callable(prompt_function):
                    raise ValueError(f"❌ Error: The prompt function '{prompt_function_name}' does not exist or is not callable.")

                prompt = prompt_function(target_category, original_text)
                response = call_llm(llm_model, prompt)

                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, "choices") and response.choices:
                    response_text = response.choices[0].message.content.strip()
                else:
                    response_text = str(response).strip()

                if "Original text:" in response_text:
                    response_text = response_text.split("Original text:", 1)[-1].strip()
                if "### END OF INPUT ###" in response_text:
                    response_text = response_text.split("### END OF INPUT ###")[-1].strip()

                if isinstance(response_text, str) and response_text.strip():
                    benchmark[key]["tailored_texts"][llm_model][target_category][prompt_key] = response_text

                    tailored_analysis = analyze_text(response_text)
                    linguistic_analysis[key]["tailored_texts"][llm_model][target_category][prompt_key] = {
                        "text": response_text,
                        "token_count": tailored_analysis["token_count"],
                        "readability": tailored_analysis["readability"],
                        "pos": tailored_analysis["pos"]
                    }

                    print(f"✅ Successfully generated and updated for {target_category} in entry {key}")
                else:
                    print(f"⚠️ Warning: Generated text is invalid for {target_category} in entry {key}. Keeping null.")
                    benchmark[key]["tailored_texts"][llm_model][target_category][prompt_key] = None

            except Exception as e:
                print(f"❌ Error generating for {target_category} in entry {key}: {e}")
                benchmark[key]["tailored_texts"][llm_model][target_category][prompt_key] = None

    save_dataset(benchmark, BENCHMARK_PATH)
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)
    print(f"\n✅ Benchmark updated with tailored texts and saved (Processed {min(len(dataset), max_entries or len(dataset))} entries).")
