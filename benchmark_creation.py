from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from config import DATASET_PATH, BENCHMARK_PATH

def create_benchmark():
    """Processes the dataset and performs readability analysis."""
    dataset = load_dataset(DATASET_PATH)

    benchmark_data = {}
    for key, value in dataset.items():
        original_text = value["original_text"]

        # Perform readability analysis
        readability_analysis = analyze_text(original_text)

        # Prepare benchmark structure with readability analysis
        benchmark_data[key] = {
            "chapter": value["chapter"],
            "sections": value["sections"],
            "topic": value["topic"],
            "original_category": value["original_category"],
            "original_text": original_text,
            "original_text_analysis": readability_analysis,  # Add only readability scores
            "CS_tailored_gpto1": None,
            "CS_tailored_gpt4o": None,
            "CS_tailored_claude_sonnet": None,
            "CS_tailored_gemini": None
        }

    save_dataset(benchmark_data, BENCHMARK_PATH)
    print(f"Benchmark with readability scores saved to {BENCHMARK_PATH}")

if __name__ == "__main__":
    create_benchmark()
