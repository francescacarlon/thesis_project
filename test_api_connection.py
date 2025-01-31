import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = openai.Client(api_key=OPENAI_API_KEY)

def test_api_key():
    """Test if the OpenAI API key is loaded correctly."""
    if OPENAI_API_KEY:
        print("✅ OpenAI API Key Loaded Successfully!")
    else:
        print("❌ Error: OpenAI API Key Not Found. Check your .env file.")

def test_gpt4o():
    """Test a simple API call to GPT-4o (Updated for openai>=1.0.0)."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain deep learning in simple terms."}
            ],
            temperature=1
        )
        print("\n✅ GPT-4o Test Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("\n❌ GPT-4o API Call Failed:", e)

if __name__ == "__main__":
    print("\n🔹 Running API Connection Tests...\n")
    test_api_key()
    test_gpt4o()
