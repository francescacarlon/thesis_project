from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from config import DATASET_PATH, BENCHMARK_PATH

def create_benchmark():
    """Processes the dataset and performs readability analysis."""
    dataset = load_dataset(DATASET_PATH)

    benchmark_data = {}
    for key, value in dataset.items():
        original_text = value["original_text"]

        # Perform readability, POS analysis
        text_analysis = analyze_text(original_text)

        # Tailoring matrix
        TAILORING_MATRIX = {
            "L": ["CS", "CL"],  # Linguistics → tailored for CS and CL
            "CS": ["L", "CL"],  # Computer Science → tailored for L and CL
            "CL": ["L", "CS"]   # Computational Linguistics → tailored for L and CS
            }

        # Prepare benchmark structure with readability analysis
        benchmark_data[key] = {
            "chapter": value["chapter"],
            "sections": value["sections"],
            "topic": value["topic"],
            "original_category": value["original_category"],
            "original_text": original_text,
            "original_text_analysis": text_analysis
        }

        # Dynamically add tailored placeholders based on the tailoring matrix
        for target_category in TAILORING_MATRIX[value["original_category"]]:
            for llm in ["gpto1", "gpt4o", "claude_sonnet", "gemini", "llama", "mistral"]:
                benchmark_data[key][f"{target_category}_tailored_{llm}"] = None


    save_dataset(benchmark_data, BENCHMARK_PATH)
    print(f"Benchmark with text analysis saved to {BENCHMARK_PATH}")


if __name__ == "__main__":
    create_benchmark()
