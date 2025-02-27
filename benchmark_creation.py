from utils import load_dataset, save_dataset
from linguistic_analysis import analyze_text
from llm_caller import call_llm
from prompts import *  # Import all prompts
from prompts import PROMPT_FUNCTIONS  # Ensure PROMPT_FUNCTIONS is imported
from config import DATASET_PATH, BENCHMARK_PATH, LINGUISTIC_ANALYSIS_PATH

def create_benchmark(llm_model, prompt_function_name):
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

    # Validate that the prompt function exists in PROMPT_FUNCTIONS
    prompt_number = next((key for key, value in PROMPT_FUNCTIONS.items() if value == prompt_function_name), None)
    if prompt_number is None:
        raise ValueError(f"❌ Error: Prompt function '{prompt_function_name}' is not in PROMPT_FUNCTIONS.")
    prompt_key = f"prompt{prompt_number}"
    
    for key, value in dataset.items():
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

        if key not in linguistic_analysis:
            original_analysis = analyze_text(original_text)
            linguistic_analysis[key] = {
                "original_category": original_category,
                "original_text": original_text,
                "readability": original_analysis["readability"],
                "pos": original_analysis["pos"],
                "tailored_texts": {}
            }

        for target_category in TAILORING_MATRIX.get(original_category, []):
            benchmark.setdefault(key, {}).setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})
            linguistic_analysis.setdefault(key, {}).setdefault("tailored_texts", {}).setdefault(llm_model, {}).setdefault(target_category, {})

            if prompt_key in benchmark[key]["tailored_texts"][llm_model][target_category]:
                print(f"⏭️ Skipping {target_category} for {key}, already exists.")
                continue  

            print(f"\n📝 Generating tailored text for {target_category} in entry {key} using {llm_model} with {prompt_key}...")

            try:
                prompt_function = globals().get(prompt_function_name)
                if not prompt_function or not callable(prompt_function):
                    raise ValueError(f"❌ Error: The prompt function '{prompt_function_name}' does not exist or is not callable.")
                
                prompt = prompt_function(target_category, original_text)
                response = call_llm(llm_model, prompt)
                
                if hasattr(response, "choices") and response.choices:
                    response_text = response.choices[0].message.content.strip()

                    # ✅ Remove the original text if echoed in the output
                    if original_text in response_text:
                        response_text = response_text.replace(original_text, "").strip()
                    
                    # ✅ Remove anything before or including "### END OF INPUT ###"
                    if "### END OF INPUT ###" in response_text:
                        response_text = response_text.split("### END OF INPUT ###")[-1].strip()

                else:
                    response_text = str(response).strip()

                
                # ✅ Check if the response is valid
                if isinstance(response_text, str) and response_text.strip():
                    print(f"🔍 Checking LLM output for {key} ({target_category}):\n{response_text[:500]}\n{'-'*80}")
                    benchmark[key]["tailored_texts"][llm_model][target_category][prompt_key] = response_text
                    

                    linguistic_analysis[key]["tailored_texts"][llm_model][target_category][prompt_key] = {
                        "text": response_text,
                        "readability": analyze_text(response_text)["readability"],
                        "pos": analyze_text(response_text)["pos"]
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
    print("\n✅ Benchmark updated with tailored texts and saved.")