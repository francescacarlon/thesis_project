import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pathlib import Path
import json

# ========================
# 🔐 Load API Key
# ========================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key="OPENAI_API_KEY")

# ========================
# 📖 Role Descriptions
# ========================
role_definitions = {
    "Linguist": (
        "Linguists have a strong background in language structure, phonetics, syntax, and semantics. "
        "They are familiar with theoretical concepts in language analysis but typically have little or no experience with programming or technical machine learning methods."
    ),
    "Computer Scientist": (
        "Computer scientists have a strong background in programming, algorithms, data structures, and machine learning. "
        "They are not typically trained in linguistics, language theory, or phonetics."
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
    )
    
    return full_prompt

# ========================
# 🤖 Query LLM
# ========================
def query_llm(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a current MSc Computational Linguistics student who has been selected for the evaluation of texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message["content"]

# ========================
# 🧪 Run for all files in folder
# ========================
def evaluate_folder(role, folder_path: Path, save_dir: Path):
    instructions = "Clarity, technical accuracy, and accessibility to someone from your background."
    html_files = list(folder_path.glob("*.html"))
    
    for file in html_files:
        texts = extract_texts_from_html_file(file)
        if len(texts) != 3:
            print(f"⚠️ Skipping {file.name} — does not contain 3 texts.")
            continue

        prompt = prompt_judge(role, texts, instructions, role_definitions)
        response = query_llm(prompt)

        # ✅ Just save filename and response
        output_data = {
            "filename": file.name,
            "response": response
        }

        output_path = save_dir / f"{file.stem}_response.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved JSON: {output_path.name}")



### MAIN X TEST ###

if __name__ == "__main__":
    BASE_PATH = Path.cwd()

    role = "Linguist"
    input_folder = BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics"
    output_folder = input_folder / "results"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Run only on the first file
    html_files = sorted(input_folder.glob("*.html"))
    if not html_files:
        print("❌ No HTML files found.")
    else:
        test_file = html_files[0]
        print(f"🚀 Running test on: {test_file.name}")

        texts = extract_texts_from_html_file(test_file)
        if len(texts) != 3:
            print(f"⚠️ Skipping {test_file.name} — does not contain 3 texts.")
        else:
            instructions = "Clarity, technical accuracy, and accessibility to someone from your background."
            prompt = prompt_judge(role, texts, instructions, role_definitions)
            response = query_llm(prompt)

            output_data = {
                "filename": test_file.name,
                "response": response
            }

            output_path = output_folder / f"{test_file.stem}_response.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved JSON: {output_path.name}")


"""# ========================
# 🚀 Run for both Linguist and Computer Scientist
# ========================
if __name__ == "__main__":
    BASE_PATH = Path.cwd()

    role_configs = [
        {
            "role": "Linguist",
            "input_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics",
            "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_linguistics/results"
        },
        {
            "role": "Computer Scientist",
            "input_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_computer_science",
            "output_folder": BASE_PATH / "data/texts_survey/llm_as_a_judge_texts/target_computer_science/results"
        }
    ]

    for config in role_configs:
        role = config["role"]
        input_folder = config["input_folder"]
        output_folder = config["output_folder"]
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Running evaluation for: {role}")
        evaluate_folder(role, input_folder, output_folder)
"""

