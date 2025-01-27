from src.config import DATA_PATH, OUTPUT_PATH, CATEGORIES, LLM_MODELS
from src.utils import load_dataset, save_dataset, log_message
from src.prompt_generator import generate_prompt
from src.llm_caller import call_llm
from datetime import datetime

def process_dataset():
    """Iterate through the dataset and generate tailored texts."""
    dataset = load_dataset(DATA_PATH)

    for key, value in dataset.items():
        original_text = value['original_text']
        
        for category in CATEGORIES:
            for model in LLM_MODELS:
                prompt = generate_prompt(original_text, category)
                
                log_message(f"Processing: {value['topic']} for {category} using {model}...")
                try:
                    tailored_text = call_llm(model, prompt)
                    value[f"{category}_tailored_{model}"] = tailored_text
                except Exception as e:
                    log_message(f"Error with {model} for {category}: {e}")
                    value[f"{category}_tailored_{model}"] = "ERROR"
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_PATH / f"benchmark_results_{timestamp}.json"
    save_dataset(dataset, output_file)
    log_message(f"Benchmarking completed. Results saved in {output_file}")
