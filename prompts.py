"""
This file automates the creation of tailored texts for all the benchmark file entries.
It automatically retrieves the original texts and original categories. 
It tailors the texts for the target categories and saves them in benchmark, in the LLM's corresponding entry. 
"""

import json
from utils import load_dataset, save_dataset  # Ensure save_dataset is implemented
from llm_caller import call_llm
from config import BENCHMARK_PATH

# ✅ Load benchmark data
benchmark_data = load_dataset(BENCHMARK_PATH)

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

# ✅ Iterate over all entries in the benchmark dataset
for entry_id, entry_data in benchmark_data.items():
    original_text = entry_data["original_text"]
    original_category = entry_data["original_category"]

    # ✅ Determine target categories
    if original_category == "L":
        target_category_1, target_category_2 = "CS", "CL"
    elif original_category == "CS":
        target_category_1, target_category_2 = "L", "CL"
    elif original_category == "CL":
        target_category_1, target_category_2 = "L", "CS"
    else:
        print(f"\n❌ Unknown category for entry {entry_id}. Skipping.")
        continue

    try:
        # ✅ Generate first tailored explanation
        prompt_1 = create_prompt2(target_category_1, original_text)
        response_1 = call_llm("gpt4o", prompt_1) # insert llm name

        # ✅ Generate second tailored explanation
        prompt_2 = create_prompt2(target_category_2, original_text)
        response_2 = call_llm("gpt4o", prompt_2) # insert llm name

        # ✅ Store the responses in the benchmark data
        benchmark_data[entry_id][f"{target_category_1}_tailored_gpt4o"] = response_1
        benchmark_data[entry_id][f"{target_category_2}_tailored_gpt4o"] = response_2

        print(f"\n✅ Tailored explanations generated for Entry {entry_id} ({entry_data['topic']})")

    except Exception as e:
        print(f"\n❌ Error processing Entry {entry_id}: {str(e)}")

# ✅ Save the updated benchmark data
with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=4, ensure_ascii=False)

print("\n🎉 All entries have been paraphrased and saved successfully!")
