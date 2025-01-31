from utils import load_dataset  # Import function to load benchmark.json
from llm_caller import call_llm
from config import BENCHMARK_PATH

# ✅ Load benchmark data
benchmark_data = load_dataset(BENCHMARK_PATH)

# ✅ Display available entries
print("\n📌 Available Text Entries:")
for key, value in benchmark_data.items():
    print(f"{key}: {value['topic']} (Category: {value['original_category']})")

# ✅ Let user pick an entry
entry_id = input("\nEnter the entry number to test: ").strip()

if entry_id not in benchmark_data:
    print("\n❌ Invalid entry number. Please try again.")
    exit()

# ✅ Retrieve `original_text`
original_text = benchmark_data[entry_id]["original_text"]

# ✅ Let user pick two target categories
target_category_1 = input("Enter first target category (L, CS, CL): ").strip().upper()
target_category_2 = input("Enter second target category (L, CS, CL): ").strip().upper()

# ✅ Define the prompt template
def create_prompt(target_category, text):
    return f"""
    You are an expert in {target_category}. Rewrite the following explanation so that it is best understood by {target_category} students.
    - Use terminology specific to their field.
    - Provide examples they are familiar with.
    - Avoid unnecessary complexity while maintaining accuracy.

    Original text: 
    {text}

    Tailored text:
    """

# ✅ Call GPT-4o with both prompts
try:
    # First tailored explanation
    prompt_1 = create_prompt(target_category_1, original_text)
    response_1 = call_llm("gpt4o", prompt_1)

    # Second tailored explanation
    prompt_2 = create_prompt(target_category_2, original_text)
    response_2 = call_llm("gpt4o", prompt_2)

    # ✅ Print the prompt & response for easy comparison
    print("\n🔹 Prompt Used for", target_category_1, ":\n", prompt_1)
    print("\n✅ Tailored Explanation for", target_category_1, ":\n", response_1)

    print("\n🔹 Prompt Used for", target_category_2, ":\n", prompt_2)
    print("\n✅ Tailored Explanation for", target_category_2, ":\n", response_2)

except Exception as e:
    print("\n❌ Error calling GPT-4o:", str(e))
