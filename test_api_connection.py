import openai
import anthropic
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Create OpenAI client
client = openai.Client(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def test_openai_key():
    """Test if the OpenAI API key is loaded correctly."""
    if OPENAI_API_KEY:
        print("✅ OpenAI API Key Loaded Successfully!")
    else:
        print("❌ Error: OpenAI API Key Not Found. Check your .env file.")

def test_anthropic_key():
    """Test if the Anthropic API key is loaded correctly."""
    if ANTHROPIC_API_KEY:
        print("✅ Anthropic API Key Loaded Successfully!")
    else:
        print("❌ Error: Anthropic API Key Not Found. Check your .env file.")

def test_anthropic_api():
    """Test a simple API call to Claude 3.5 Sonnet."""
    try:
        response = anthropic_client.messages.create(
                model="claude-3.5-sonnet-20240626",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": [{"type": "text", "text": "Hello! Can you confirm my Anthropic API key is working?"}]}]
            )
        print("\n✅ Anthropic API Test Response:")
        print(response.content[0]["text"])
    except Exception as e:
        print("\n❌ Anthropic API Call Failed:", e)

def list_anthropic_models():
    """Lists available Anthropic models."""
    try:
        models = anthropic_client.models.list()
        print("\n✅ Available Anthropic Models:")
        for model in models.data:  # ✅ Access model list properly
            print("-", model.id)  # ✅ Extract model ID correctly
    except Exception as e:
        print("\n❌ Error fetching model list:", e)

if __name__ == "__main__":
    print("\n🔹 Running API Connection Tests...\n")
    test_openai_key()
    test_anthropic_key()
    list_anthropic_models()
    
    # ✅ Uncomment this if you want to test a real API call
    test_anthropic_api()
    

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



import openai

client = openai.OpenAI()  # Ensure your OpenAI client is properly initialized
models = client.models.list()

for model in models:
    print(model.id)


response = client.chat.completions.create(
    model="o1",
    messages=[{"role": "user", "content": "Hello, can you confirm if I have access to o1?"}]
)

print(response)
