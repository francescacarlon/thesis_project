import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print the key (for testing only, avoid in production)
print("Your OpenAI API Key:", os.getenv("OPENAI_API_KEY"))
