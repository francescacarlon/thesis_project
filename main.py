from benchmark_creation import create_benchmark
from prompts import PROMPT_FUNCTIONS

# ✅ Define available LLMs and prompts
LLM_MODELS = {
    # "gpt4o": "gpt4o"
    # "claude" : "claude",
    # "deepseek" : "deepseek"
    # "mistral" : "mistral"
    # "llama" : "llama"    
}


def main():
    print("\n🚀 Starting automatic benchmark generation for all LLMs and prompts...\n")

    # TEST: ONLY N PROMPT 

    llm_model = "llama"
    prompt_function_name = PROMPT_FUNCTIONS[1]  # Choose prompt n. to test

    print(f"\n📝 Running benchmark for:")
    print(f"   🧠 LLM Model: {llm_model}")
    print(f"   ✏️  Prompt Function: {prompt_function_name}")

    # ✅ Call the benchmark function for each LLM and each prompt
    create_benchmark(llm_model, prompt_function_name, target_key="12") # max_entries=15) #,  # max_entries=from 1 to n. OR # target_key=only that key

    # WHOLE BENCHMARK
    # for llm_model in LLM_MODELS.values():  # ✅ Only keep values (model names)
    #     for prompt_function_name in PROMPT_FUNCTIONS.values():  # ✅ Only keep function names
# 
    #         print(f"\n📝 Running benchmark for:")
    #         print(f"   🧠 LLM Model: {llm_model}")
    #         print(f"   ✏️  Prompt Function: {prompt_function_name}")

    # ✅ Call the benchmark function for each LLM and each prompt
            # create_benchmark(llm_model, prompt_function_name) # WHOLE BENCHMARK

    print("\n✅ All benchmarks have been generated successfully!")

if __name__ == "__main__":
    main()
