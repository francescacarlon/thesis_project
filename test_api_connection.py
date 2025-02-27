import requests
import os
from dotenv import load_dotenv

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


