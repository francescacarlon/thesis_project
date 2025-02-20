import argparse
from benchmark_creation import create_benchmark

# ✅ Define available LLMs and prompts
LLM_MODELS = {
    "gpt4o": "gpt-4o",
    "o1-preview": "o1-preview",
    "claude" : "claude-3-5-sonnet-20241022"
}

PROMPT_FUNCTIONS = {
    2: "create_prompt2",
    3: "create_prompt3",
    4: "create_prompt4"
}

def main():
    # ✅ Argument parser for command-line flexibility
    parser = argparse.ArgumentParser(description="Run benchmark generation with selected LLM and prompt.")
    parser.add_argument("--llm", type=str, choices=LLM_MODELS.keys(), required=True, help="Select LLM model.")
    parser.add_argument("--prompt", type=int, choices=PROMPT_FUNCTIONS.keys(), required=True, help="Select prompt version.")
    
    args = parser.parse_args()
    
    # ✅ Get selected LLM and prompt function
    llm_model = LLM_MODELS.get(args.llm)
    prompt_function_name = PROMPT_FUNCTIONS.get(args.prompt)

    if llm_model is None or prompt_function_name is None:
        print("\n❌ Error: Invalid LLM or prompt selection.")
        print("   Available LLMs:", list(LLM_MODELS.keys()))
        print("   Available Prompts:", list(PROMPT_FUNCTIONS.keys()))
        exit(1)

    # ✅ Normalize model name for consistency in stored keys
    if llm_model in ["o1-preview"]:
        llm_model_key = "o1-preview"  # Consistent with benchmark JSON format
    else:
        llm_model_key = llm_model  # Use actual model name if not a special case

    print(f"\n🚀 Running benchmark with:")
    print(f"   🧠 LLM Model: {llm_model} (stored as {llm_model_key})")
    print(f"   ✏️  Prompt Function: {prompt_function_name}\n")

    # ✅ Call the benchmark function with the normalized LLM name
    create_benchmark(llm_model_key, prompt_function_name)

if __name__ == "__main__":
    main()
