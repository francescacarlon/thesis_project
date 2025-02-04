"""
Functions to create benchmark (original texts and tailored texts) and linguistic_analysis (linguistic metrics results) files. 
It takes the original_text from the dataset, makes the tailored paraphrasis according to LLM and target category, performs linguistic analysis.
"""

from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from llm_caller import call_llm
from prompts import create_prompt2  # add other prompt functions if using other prompts
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH  

def create_benchmark():
    """Processes the dataset, performs readability and POS analysis, and saves linguistic analysis separately."""
    dataset = load_dataset(DATASET_PATH)

    # ✅ Load existing benchmark data or initialize new
    try:
        benchmark_data = load_dataset(BENCHMARK_PATH)
    except FileNotFoundError:
        benchmark_data = {}

    # ✅ Load existing linguistic analysis or initialize new
    try:
        linguistic_analysis = load_dataset(LINGUISTIC_ANALYSIS_PATH)
    except FileNotFoundError:
        linguistic_analysis = {}

    # Tailoring matrix: Defines which categories get tailored to which
    TAILORING_MATRIX = {
        "L": ["CS", "CL"],  # Linguistics → tailored for CS and CL
        "CS": ["L", "CL"],  # Computer Science → tailored for L and CL
        "CL": ["L", "CS"]   # Computational Linguistics → tailored for L and CS
    }

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

        # ✅ First perform linguistic analysis for the original text if missing
        if key not in linguistic_analysis:
            original_analysis = analyze_text(original_text)  # Call once
            linguistic_analysis[key] = {
                "original_category": original_category,
                "original_text": original_text,
                "readability": original_analysis["readability"],
                "pos": original_analysis["pos"],
                "tailored_texts": {}  
            }


        # ✅ Generate tailored explanations for each target category and LLM
        for target_category in TAILORING_MATRIX.get(original_category, []):
            tailored_key = f"{target_category}_tailored_gpt4o"

            # ✅ Skip if already generated
            if benchmark_data[key].get(tailored_key) is not None:
                print(f"⏭️ Skipping {target_category} for {key}, already exists.")

                # ✅ Ensure linguistic analysis is done for already existing entries
                if tailored_key not in linguistic_analysis[key]["tailored_texts"]:
                    print(f"📊 Performing linguistic analysis for existing {target_category} entry {key}...")
                    tailored_text = benchmark_data[key][tailored_key]
                    linguistic_analysis[key]["tailored_texts"][tailored_key] = {
                        "text": tailored_text,
                        "readability": analyze_text(tailored_text)["readability"],
                        "pos": analyze_text(tailored_text)["pos"]
                    }
                continue  # Skip generating a new paraphrase


            # ✅ Use the tailored prompt
            prompt = create_prompt2(target_category, original_text)

            try:
                # ✅ Call GPT-4o for paraphrased output
                response = call_llm("gpt4o", prompt)

                # ✅ Save the result in benchmark_data
                benchmark_data[key][tailored_key] = response

                # ✅ Perform linguistic analysis on the generated text
                linguistic_analysis[key]["tailored_texts"][tailored_key] = {
                    "text": response,
                    "readability": analyze_text(response)["readability"],
                    "pos": analyze_text(response)["pos"]
                }

                print(f"✅ Successfully generated for {target_category} in entry {key}")

            except Exception as e:
                print(f"❌ Error generating for {target_category} in entry {key}: {e}")
                benchmark_data[key][tailored_key] = None  # Mark as None if generation fails

    # ✅ Save the modified benchmark data
    save_dataset(benchmark_data, BENCHMARK_PATH)

    # ✅ Save linguistic analysis separately
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)

    print("\n✅ Benchmark updated with tailored texts and saved.")

if __name__ == "__main__":
    create_benchmark()
