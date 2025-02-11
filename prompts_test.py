"""
To test different prompts on selected original_texts from the 3 categories. 
This file allows me to choose one entry from benchmark, and have it tailored to other categories. 
After reading the output, I choose the prompt giving the best texts to tailor the whole benchmark. 
"""

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

# ✅ Define the prompt template and define the categories
def create_prompt1(target_category, text):

    category_definitions = {
        "L": "Linguistics (L) students have a strong background on language structure, phonetics, syntax, and semantics. They have little or no technical knowledge.",
        "CS": "Computer Science (CS) students have strong technical backgrounds, including programming, algorithms, and machine learning. They have little or no linguistic knowledge.",
        "CL": "Computational Linguistics (CL) students bridge linguistics and computer science, NLP, corpus linguistics, AI, and LLMs."
    }
    
    category_explanation = category_definitions.get(target_category, "Unknown category")

    return f"""
    You are an expert professor in {target_category}. Rewrite the following explanation so that it is best understood by {target_category} students.
    The most important thing is that your students understand the concept that is different from their background knowledge, so that they can pass your exam and your salary increases. 
    You must consider the information on the {category_explanation} to adapt the tailored texts to the students' backgrounds. 
    - Read the text carefully.
    - Identify the key concepts.
    - Use terminology specific to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy. 
    - Provide examples they are familiar with.
    - Provide analogies they can relate their knowledge with and transfer it to new concepts.
    - Integrate background information if needed.    

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """

# ✅ Define the prompt template and define the categories
def create_prompt2(target_category, text):

    category_definitions = {
        "L": "Linguistics (L) clients have a strong background on language structure, phonetics, syntax, and semantics. They have little or no technical knowledge.",
        "CS": "Computer Science (CS) clients have strong technical backgrounds, including programming, algorithms, and machine learning. They have little or no linguistic knowledge.",
        "CL": "Computational Linguistics (CL) clients bridge linguistics and computer science, NLP, corpus linguistics, AI, and LLMs."
    }
    
    category_explanation = category_definitions.get(target_category, "Unknown category") # unknown if the category does not appear in category_definitions

    return f"""
    You are an expert in AI sales, responsible for selling AI products developed by your company to clients with a background in {target_category}.
    To successfully sell your products, your clients must first understand the following concepts. Rewrite the explanation below in a way that is most comprehensible to {target_category} clients.
    The key objective is to ensure that your clients grasp concepts outside their existing background knowledge so they can make an informed purchase. By doing so, you will increase your sales percentage and achieve greater success.
    Consider the information on {category_explanation} to tailor the explanation to your clients' specific backgrounds 
    - Read the text carefully.
    - Identify the key concepts.
    - Use terminology specific to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy. 
    - Provide examples they are familiar with.
    - Provide analogies they can relate their knowledge with and transfer it to new concepts.
    - Integrate background information if needed.    

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """


# ✅ Call models with selected prompts
try:
    # First tailored explanation
    prompt_1 = create_prompt1(target_category_1, original_text)
    response_1 = call_llm("llama", prompt_1)

    # Second tailored explanation
    prompt_2 = create_prompt1(target_category_2, original_text)
    response_2 = call_llm("llama", prompt_2)

    # ✅ Print the prompt & response for easy comparison
    #print("\n🔹 Prompt Used for", target_category_1, ":\n", prompt_1)
    print("\n✅ Tailored Explanation for", target_category_1, ":\n", response_1)

    #print("\n🔹 Prompt Used for", target_category_2, ":\n", prompt_2)
    print("\n✅ Tailored Explanation for", target_category_2, ":\n", response_2)

except Exception as e:
    print(f"\n❌ Error calling model:", str(e))

