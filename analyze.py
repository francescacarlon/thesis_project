from src.utils import load_dataset, save_dataset, log_message
from src.linguistic_analysis import analyze_dataset
from src.config import DATA_PATH, OUTPUT_PATH
from datetime import datetime

def main():
    log_message("Loading dataset for linguistic analysis...")
    dataset = load_dataset(DATA_PATH)

    log_message("Performing linguistic analysis...")
    analysis_results = analyze_dataset(dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_PATH / f"linguistic_analysis_{timestamp}.json"
    save_dataset(analysis_results, output_file)

    log_message(f"Linguistic analysis completed. Results saved in {output_file}")

if __name__ == "__main__":
    main()
