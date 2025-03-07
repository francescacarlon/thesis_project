from benchmark_creation import create_benchmark
from prompts import PROMPT_FUNCTIONS

# ✅ Define available LLMs and prompts
LLM_MODELS = {
    "deepseek": "deepseek",
    "gpt4o": "gpt4o",
    "claude": "claude",
    "mistral": "mistral",
    "llama": "llama"
}

# ✅ Toggle between test mode and full benchmark
TEST_MODE = False  # Set to False to run the whole benchmark

def main():
    print("\n🚀 Starting automatic benchmark generation...\n")

    if TEST_MODE:
        # ✅ Run a single test prompt
        llm_model = "llama"  # Set the model for testing
        prompt_function_name = PROMPT_FUNCTIONS[3]  # Choose the specific prompt function

        print(f"\n📝 Running test benchmark for:")
        print(f"   🧠 LLM Model: {llm_model}")
        print(f"   ✏️  Prompt Function: {prompt_function_name}")

        # ✅ Call the benchmark function for the test case
        create_benchmark(llm_model, prompt_function_name, target_key="4")  # Optionally: max_entries=2

    else:
        # ✅ Run the full benchmark loop for all LLMs and prompts
        for llm_model in LLM_MODELS.values():
            for prompt_function_name in PROMPT_FUNCTIONS.values():
                print(f"\n📝 Running benchmark for:")
                print(f"   🧠 LLM Model: {llm_model}")
                print(f"   ✏️  Prompt Function: {prompt_function_name}")

                # ✅ Call the benchmark function for each LLM and each prompt
                create_benchmark(llm_model, prompt_function_name)

        print("\n✅ All benchmarks have been generated successfully!")

if __name__ == "__main__":
    main()
