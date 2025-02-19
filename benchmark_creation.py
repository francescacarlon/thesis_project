"""
Functions to create the benchmark (original texts & tailored texts) and perform linguistic analysis.
It retrieves `original_text` from the dataset, paraphrases it using an LLM based on the target category, 
and analyzes linguistic features.
"""

from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from llm_caller import call_llm
from prompts import *  # Import all prompts
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH  

def create_benchmark(llm_model, prompt_function_name):
    """
    Generates tailored paraphrases using the selected LLM and prompt, 
    and performs linguistic analysis for each entry in the dataset.

    Args:
        llm_model (str): The selected LLM model to generate paraphrases.
        prompt_function_name (str): The name of the prompt function to use.
    """

    dataset = load_dataset(DATASET_PATH)

    # ✅ Load existing benchmark and linguistic analysis data
    try:
        benchmark_data = load_dataset(BENCHMARK_PATH)
    except FileNotFoundError:
        benchmark_data = {}

    try:
        linguistic_analysis = load_dataset(LINGUISTIC_ANALYSIS_PATH)
    except FileNotFoundError:
        linguistic_analysis = {}

    # ✅ Tailoring matrix: Defines which categories get tailored to which
    TAILORING_MATRIX = {
        "L": ["CS", "CL"],  
        "CS": ["L", "CL"],  
        "CL": ["L", "CS"]   
    }

    # ✅ Normalize model name to maintain consistent keys in `benchmark.json`
    if llm_model in "o1-preview":
        llm_model_key = "o1-preview"
    else:
        llm_model_key = llm_model  # Keep original model name for other cases

    # ✅ Fetch the selected prompt function dynamically
    prompt_function = globals().get(prompt_function_name)
    if not prompt_function:
        raise ValueError(f"❌ Error: The prompt function '{prompt_function_name}' does not exist. Check main.py.")

    for key, value in dataset.items():
        original_text = value["original_text"]
        original_category = value["original_category"]

        print(f"\n🔹 Processing Entry {key} ({original_category})...")

        # ✅ Ensure the entry exists in benchmark data
        if key not in benchmark_data:
            benchmark_data[key] = {
                "chapter": value["chapter"],
                "sections": value["sections"],
                "topic": value["topic"],
                "original_category": original_category,
                "original_text": original_text
            }

        # ✅ Perform linguistic analysis for the original text if missing
        if key not in linguistic_analysis:
            original_analysis = analyze_text(original_text)  
            linguistic_analysis[key] = {
                "original_category": original_category,
                "original_text": original_text,
                "readability": original_analysis["readability"],
                "pos": original_analysis["pos"],
                "tailored_texts": {}  
            }

        # ✅ Generate tailored explanations for the target categories
        for target_category in TAILORING_MATRIX.get(original_category, []):
            tailored_key = f"{target_category}_tailored_{llm_model_key}"

            # ✅ Check if tailored text already exists and is valid
            tailored_text = benchmark_data[key].get(tailored_key)

            if tailored_text is not None and isinstance(tailored_text, str) and tailored_text.strip():
                print(f"⏭️ Skipping {target_category} for {key}, already exists.")

                # ✅ Ensure linguistic analysis is performed
                if tailored_key not in linguistic_analysis[key]["tailored_texts"]:
                    print(f"📊 Performing linguistic analysis for existing {target_category} entry {key}...")
                    linguistic_analysis[key]["tailored_texts"][tailored_key] = {
                        "text": tailored_text,
                        "readability": analyze_text(tailored_text)["readability"],
                        "pos": analyze_text(tailored_text)["pos"]
                    }
                continue  

            # ✅ If tailored text is missing (None or empty), generate it
            print(f"⚠️ No valid tailored text for {target_category} in entry {key}. Generating...")

            try:
                # ✅ Generate the missing tailored text
                prompt = prompt_function(target_category, original_text)
                print(f"📝 Generated Prompt for {target_category} in entry {key}: {prompt}")

                response = call_llm(llm_model, prompt)
                print(f"🔄 LLM Response for {target_category} in entry {key}: {response}")  # Debugging output

                # ✅ Ensure response is valid before saving
                if isinstance(response, str) and response.strip():
                    # ✅ Ensure the key exists in benchmark_data before updating
                    if key not in benchmark_data:
                        benchmark_data[key] = {}

                    benchmark_data[key][tailored_key] = response  # Save generated text in benchmark.json
                    
                    # ✅ Ensure entry exists in linguistic_analysis before storing analysis
                    if key not in linguistic_analysis:
                        linguistic_analysis[key] = {"tailored_texts": {}}
                    if "tailored_texts" not in linguistic_analysis[key]:
                        linguistic_analysis[key]["tailored_texts"] = {}

                    # ✅ Perform linguistic analysis
                    linguistic_analysis[key]["tailored_texts"][tailored_key] = {
                        "text": response,
                        "readability": analyze_text(response)["readability"],
                        "pos": analyze_text(response)["pos"]
                    }

                    print(f"✅ Successfully generated and updated for {target_category} in entry {key}")

                else:
                    print(f"⚠️ Warning: Generated text is invalid for {target_category} in entry {key}. Keeping null.")
                    benchmark_data[key][tailored_key] = None

            except Exception as e:
                print(f"❌ Error generating for {target_category} in entry {key}: {e}")
                benchmark_data[key][tailored_key] = None  # Keep null if there's an error

    # ✅ Save the updated benchmark and linguistic analysis data
    save_dataset(benchmark_data, BENCHMARK_PATH)  # Ensure benchmark.json is updated
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)

    print("\n✅ Benchmark updated with tailored texts and saved.")
