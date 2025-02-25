import openai
import anthropic
import os
from dotenv import load_dotenv
import requests

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Create OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def test_openai_key():
    """Test if the OpenAI API key is loaded correctly."""
    if OPENAI_API_KEY:
        print("✅ OpenAI API Key Loaded Successfully!")
    else:
        print("❌ Error: OpenAI API Key Not Found. Check your .env file.")

def test_deepseek_key():
    """Test if the DeepSeek API key is loaded correctly."""
    if DEEPSEEK_API_KEY:
        print("✅ DeepSeek API Key Loaded Successfully!")
    else:
        print("❌ Error: DeepSeek API Key Not Found. Check your .env file.")

# def test_anthropic_key():
#     """Test if the Anthropic API key is loaded correctly."""
#     if ANTHROPIC_API_KEY:
#         print("✅ Anthropic API Key Loaded Successfully!")
#     else:
#         print("❌ Error: Anthropic API Key Not Found. Check your .env file.")

# def test_anthropic_api():
#     """Test a simple API call to Claude 3.5 Sonnet."""
#     try:
#         response = anthropic_client.messages.create(
#                 model="claude-3.5-sonnet-20240626",
#                 max_tokens=100,
#                 temperature=0.7,
#                 messages=[{"role": "user", "content": [{"type": "text", "text": "Hello! Can you confirm my Anthropic API key is working?"}]}]
#             )
#         print("\n✅ Anthropic API Test Response:")
#         print(response.content[0]["text"])
#     except Exception as e:
#         print("\n❌ Anthropic API Call Failed:", e)

# def list_anthropic_models():
#     """Lists available Anthropic models."""
#     try:
#         models = anthropic_client.models.list()
#         print("\n✅ Available Anthropic Models:")
#         for model in models.data:  # ✅ Access model list properly
#             print("-", model.id)  # ✅ Extract model ID correctly
#     except Exception as e:
#         print("\n❌ Error fetching model list:", e)

# if __name__ == "__main__":
#     print("\n🔹 Running API Connection Tests...\n")
#     test_openai_key()
#     # test_anthropic_key()
    # list_anthropic_models()
    
    # ✅ Uncomment this if you want to test a real API call
    # test_anthropic_api()
    

# def test_gpt4o():
#     """Test a simple API call to GPT-4o (Updated for openai>=1.0.0)."""
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Explain deep learning in simple terms."}
#             ],
#             temperature=1
#         )
#         print("\n✅ GPT-4o Test Response:")
#         print(response.choices[0].message.content)
#     except Exception as e:
#         print("\n❌ GPT-4o API Call Failed:", e)

def test_openai_models():
    """List available OpenAI models."""
    try:
        models = client.models.list()
        print("\n✅ Available OpenAI Models:")
        for model in models.data:  # ✅ Corrected `.data`
            print("-", model.id)  # ✅ Corrected `.id`
    except Exception as e:
        print("\n❌ Error fetching OpenAI model list:", e)

def test_o1():
    """Test an API call to OpenAI's o1 model."""
    try:
        response = client.chat.completions.create(
            model="o1",
            messages=[{"role": "user", "content": "Hello, can you confirm if I have access to o1?"}]
        )
        print("\n✅ 'o1' Test Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("\n❌ 'o1' API Call Failed:", e)

def list_deepseek_models():
    """Lists available DeepSeek models using the API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get("https://api.deepseek.com/v1/models", headers=headers)

        if response.status_code == 200:
            print("\n✅ Available DeepSeek Models:")
            models = response.json().get("data", [])
            for model in models:
                print("-", model["id"])  # Correctly extract model IDs
        else:
            print(f"\n❌ Error fetching DeepSeek model list: {response.status_code} - {response.text}")
    except Exception as e:
        print("\n❌ DeepSeek API Call Failed:", e)

def test_deepseek_api():
        """Test a simple API call to deepSeek-reasoner."""
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-reasoner",  # Replace with the correct model name if different
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you confirm my DeepSeek API key is working?"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",  # Replace with the correct API endpoint
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                print("\n✅ DeepSeek API Test Response:")
                print(response.json()["choices"][0]["message"]["content"])
            else:
                print(f"\n❌ DeepSeek API Call Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print("\n❌ DeepSeek API Call Failed:", e)

if __name__ == "__main__":
    print("\n🔹 Running OpenAI API Tests...\n")
    # test_openai_key()
    # test_openai_models()
    list_deepseek_models()
    test_deepseek_key()
    
    # ✅ Uncomment to test model directly
    # test_o1()
    test_deepseek_api()




url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "model": "deepseek-reasoner",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 10,
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("❌ Error:", e)