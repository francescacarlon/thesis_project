import requests
import os
from dotenv import load_dotenv
import json
from config import LINGUISTIC_ANALYSIS_PATH

HF_API_URL = "https://huggingface.co/api/models"
search_query = "mistral"  # Searches for Mistral models

params = {
    "search": search_query,  # Search for Mistral models
    "limit": 20,  # Number of models to retrieve
}

response = requests.get(HF_API_URL, params=params)

if response.status_code == 200:
    models = response.json()
    if models:
        print("\n✅ Available Mistral Models on Hugging Face:")
        for model in models:
            print("-", model["modelId"])
    else:
        print("\n❌ No Mistral models found.")
else:
    print(f"\n❌ Failed to retrieve models: {response.status_code} - {response.text}")


# Load Hugging Face API Key from .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Check if API Key is loaded
if not HF_API_KEY:
    print("❌ Error: Hugging Face API Key Not Found. Check your .env file.")
    exit()

# Define the API request
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": "Explain deep learning in simple terms.",
    "parameters": {"max_new_tokens": 100, "temperature": 0.7},
}

# Call Hugging Face API for Mistral
url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3" # WORKS

try:
    response = requests.post(url, headers=headers, json=payload)
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())  # Display API response
except Exception as e:
    print("❌ Error:", e)


### CHECK ON COSINE SIMILARITY

# ✅ Load the dataset
with open(LINGUISTIC_ANALYSIS_PATH, "r", encoding="utf-8") as f:
    linguistic_analysis = json.load(f)

# ✅ Iterate through the dataset
found_compared_texts = False  # Track if any comparisons exist

for key, entry in linguistic_analysis.items():
    original_text = entry["original_text"]  # The original text

    for llm in entry.get("tailored_texts", {}):
        for category in entry["tailored_texts"][llm]:
            for prompt_key, analysis in entry["tailored_texts"][llm][category].items():
                if "cosine_similarity" in analysis:
                    found_compared_texts = True  # Mark that at least one comparison exists
                    tailored_text = analysis["text"]  # The tailored text
                    similarity = analysis["cosine_similarity"]

                    # ✅ Print the text pairs
                    print(f"\n🔍 Already compared:")
                    print(f"📌 Entry: {key} | LLM: {llm} | Category: {category} | Prompt: {prompt_key}")
                    print(f"📝 Original: {original_text[:300]}...")  # Show first 300 characters
                    print(f"📝 Tailored: {tailored_text[:300]}...")  # Show first 300 characters
                    print(f"📊 Cosine Similarity: {similarity:.4f}")
                    print("-" * 80)  # Separator for readability

# ✅ If no compared texts found, inform the user
if not found_compared_texts:
    print("\n🚨 No texts with cosine similarity found!")




