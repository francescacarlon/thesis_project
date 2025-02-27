from benchmark_creation import create_benchmark
from prompts import PROMPT_FUNCTIONS

# ✅ Define available LLMs and prompts
LLM_MODELS = {
    # "gpt4o": "gpt4o"
    # "claude" : "claude",
    #"deepseek" : "deepseek",
    # "llama" : "llama",
    "mistral" : "mistral"
}


def main():
    print("\n🚀 Starting automatic benchmark generation for all LLMs and prompts...\n")

    for llm_model in LLM_MODELS.values():  # ✅ Only keep values (model names)
        for prompt_function_name in PROMPT_FUNCTIONS.values():  # ✅ Only keep function names
            print(f"\n📝 Running benchmark for:")
            print(f"   🧠 LLM Model: {llm_model}")
            print(f"   ✏️  Prompt Function: {prompt_function_name}")

            # ✅ Call the benchmark function for each LLM and each prompt
            create_benchmark(llm_model, prompt_function_name)

    print("\n✅ All benchmarks have been generated successfully!")

if __name__ == "__main__":
    main()
