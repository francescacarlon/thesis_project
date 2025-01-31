from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH  # Import path for linguistic analysis

def create_benchmark():
    """Processes the dataset, performs readability and POS analysis, and saves linguistic analysis separately."""
    dataset = load_dataset(DATASET_PATH)

    benchmark_data = {}
    linguistic_analysis = {}  # Dictionary to store linguistic analysis separately

    for key, value in dataset.items():
        original_text = value["original_text"]
        original_category = value["original_category"]

        # Perform linguistic analysis for the original text
        linguistic_analysis[key] = {
            "original_category": original_category,
            "original_text": original_text,
            "readability": analyze_text(original_text)["readability"],
            "pos": analyze_text(original_text)["pos"],
            "tailored_texts": {}  # Placeholder for tailored analyses
        }

        # Tailoring matrix
        TAILORING_MATRIX = {
            "L": ["CS", "CL"],  # Linguistics → tailored for CS and CL
            "CS": ["L", "CL"],  # Computer Science → tailored for L and CL
            "CL": ["L", "CS"]   # Computational Linguistics → tailored for L and CS
        }

        # Prepare benchmark structure without linguistic analysis
        benchmark_data[key] = {
            "chapter": value["chapter"],
            "sections": value["sections"],
            "topic": value["topic"],
            "original_category": original_category,
            "original_text": original_text
        }

        # Dynamically add tailored placeholders based on the tailoring matrix
        for target_category in TAILORING_MATRIX[value["original_category"]]:
            for llm in ["gpto1", "gpt4o", "claude_sonnet", "llama", "mistral"]:
                tailored_key = f"{target_category}_tailored_{llm}"
                benchmark_data[key][tailored_key] = None  # Placeholder in benchmark

                # If a tailored text exists, perform analysis and store it in linguistic_analysis.json
                if value.get(tailored_key):  
                    tailored_text = value[tailored_key]
                    linguistic_analysis[key]["tailored_texts"][tailored_key] = {
                        "text": tailored_text,
                        "readability": analyze_text(tailored_text)["readability"],
                        "pos": analyze_text(tailored_text)["pos"]
                    }

    # Save the modified benchmark data without linguistic analysis
    save_dataset(benchmark_data, BENCHMARK_PATH)

    # Save linguistic analysis separately with tailored texts included
    save_dataset(linguistic_analysis, LINGUISTIC_ANALYSIS_PATH)

    print(f"Benchmark saved to {BENCHMARK_PATH}")
    print(f"Linguistic analysis (with tailored texts) saved to {LINGUISTIC_ANALYSIS_PATH}")


if __name__ == "__main__":
    create_benchmark()
