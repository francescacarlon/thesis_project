from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup # to extract data from HTML or XML documents
from pathlib import Path
import json
from llm_caller import call_llm

# ========================
# 🔐 Load & Validate All API Keys
# ========================
load_dotenv()

required_keys = {
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic (Claude)",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "HF_API_KEY": "Hugging Face (Mistral, LLaMA)"
}

for env_key, service_name in required_keys.items():
    if not os.getenv(env_key):
        print(f"⚠️ {env_key} ({service_name}) not found. Make sure your .env file is set up correctly.")


# ========================
# 📖 Role Descriptions
# ========================

# role_definitions = {
#     "Linguist": (
#         "Linguists have a strong background in language structure, phonetics, syntax, and semantics. "
#         "They are familiar with theoretical concepts in language analysis but typically have little or no experience with programming or technical machine learning methods."
#     ),
#     "Computer Scientist": (
#         "Computer scientists have a strong background in programming, algorithms, data structures, and machine learning. "
#         "They are not typically trained in linguistics, language theory, or phonetics."
#     )
# }


# NEW Role definitions according to the typical human evaluator's background information from survey
role_definitions = {
    "Linguist": (
        "Linguists are students enrolled in the Master of Science Computational Linguistics. They currently study AI, Machine Learning and NLP-related subjects."
        "They have obtained a Bachelor's degree in Linguistics, where they studied phonetics, syntax, and semantics."
        "They have completed University-level courses mainly in Linguistics, a few in Computational Linguistics but none or only few in Computer Science."
        "They have gained work experience mainly in Linguistics, a little in Computational Linguistics, but none or little in Computer Science."
        "Their native language is not English."
    ),
    "Computer Scientist": (
        "Computer Scientists are students enrolled in the Master of Science Computational Linguistics. They currently study AI, Machine Learning and NLP-related subjects."
        "They have obtained a Bachelor's degree in Computer Science, where they studied programming, algorithms and data structures."
        "They have completed University-level courses mainly in Computer Science, a few in Computational Linguistics but none in Linguistics."
        "They have gained work experience mainly in Computer Science, a little in Computational Linguistics, but none in Linguistics."
        "Their native language is not English."
    )
}


# ========================
# 📄 Extract 3 explanations from file
# ========================
def extract_texts_from_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    texts = []
    for box in soup.find_all("div", class_="box"):
        category = box.find("p", class_="category").get_text(strip=True)
        explanation = box.find_all("p")[-1].get_text(separator=' ', strip=True)
        texts.append(f"{category}: {explanation}")
    return texts

# ========================
# ✍️ Prompt Constructor
# ========================
def prompt_judge(role, input_texts, instructions, role_definitions):
    role_description = f"You are a {role}.\n\n{role_definitions[role]}"
    
    task_intro = (
        "In this section, you will read 3 texts on a topic. "
        "Which texts helped you understand the topic the most? Rank them according to your preference. "
        "Please provide the reference name of the texts (e.g. 1. CS_o_6, 2. L_l_2_3, 3. CS_c_5_4). "
        "Explain the reasoning behind your selection of the best and worst text. "
        "Please provide the reference for the texts (e.g. best: CS_o_6, worst: L_l_2_3) and share your thoughts on why they were easy or difficult to understand. "
        "In your comments, consider both the content and the form."
    )
    
    numbered_texts = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(input_texts)])
    
    full_prompt = (
        f"{role_description}\n\n"
        f"{task_intro}\n\n"
        f"**Evaluation Criteria**:\n{instructions}\n\n"
        f"**Texts to evaluate:**\n{numbered_texts}\n\n"
        "Return your answer in the following format:\n"
        "Rankings: [e.g., 1. CS_o_6, 2. L_l_2_3, 3. CS_c_5_4]\n"
        "Best: <reference>\n"
        "Best Comment: <why it was helpful>\n"
        "Worst: <reference>\n"
        "Worst Comment: <why it was hard to understand>"
        "**Important**: Do *not* repeat the prompt or the input texts in your answer. Only return your evaluation in the format above."

    )
    
    return full_prompt


# ========================
# 🧪 Run for all files in folder
# ========================
def evaluate_folder(role, folder_path: Path, save_dir: Path, model_name="gpt4o"):
    instructions = "Clarity, technical accuracy, and accessibility to someone from your background."
    html_files = list(folder_path.glob("*.html"))

    for file in html_files:
        output_path = save_dir / f"{file.stem}_{model_name}_response.json"
        
        if output_path.exists():
            print(f"⏭ Skipping {file.name} for {model_name} (already processed)")
            continue

        texts = extract_texts_from_html_file(file)
        if len(texts) != 3:
            print(f"⚠️ Skipping {file.name} — does not contain 3 texts.")
            continue

        prompt = prompt_judge(role, texts, instructions, role_definitions)
        response = call_llm(model_name, prompt)

        if response is None:
            print(f"❌ No response for {file.name} with model {model_name}")
            continue

        output_data = {
            "model": model_name,
            "filename": file.name,
            "response": response
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved JSON: {output_path.name}")


def load_results_from_folder(folder):
    result_dict = {}
    for json_file in folder.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            filename = data.get("filename", json_file.name.split("_response")[0] + ".html")
            model = data.get("model", "unknown")

            if filename not in result_dict:
                result_dict[filename] = {}

            result_dict[filename][model] = {
                "response": data["response"]
            }
    return result_dict

# ========================
# 🚀 Run for both Linguist and Computer Scientist
# ========================
if __name__ == "__main__":
    BASE_PATH = Path.cwd()
    #models_who_judge = ["gpt4o", "claude", "deepseek", "llama", "mistral"]

    role_configs = [
        {
            "role": "Linguist",
            "input_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics",
            # "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics/results"
            "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics/new_results" # new results
        },
        {
            "role": "Computer Scientist",
            "input_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_computer_science",
            # "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_computer_science/results"
            "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_computer_science/new_results" # new results
        }
    ]

    for model_name in models_who_judge:
        for config in role_configs:
            role = config["role"]
            input_folder = config["input_folder"]
            output_folder = config["output_folder"]
            output_folder.mkdir(parents=True, exist_ok=True)

            print(f"\n🚀 Running {model_name} evaluation for: {role}")
            evaluate_folder(role, input_folder, output_folder, model_name=model_name)


    # ========================
    # 📦 Aggregate All Results into One File
    # ========================
    aggregated_results = {}
    for config in role_configs:
        role = config["role"]
        results_folder = config["output_folder"]
        print(f"📥 Aggregating results for: {role}")
        aggregated_results[role] = load_results_from_folder(results_folder)

    # output_path = BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/evaluation_results.json"
    output_path = BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/new_evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Combined results saved to: {output_path}")
