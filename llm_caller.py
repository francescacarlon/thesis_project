"""
This file calls the LLMs with the corresponding API keys. 
"""

#from config import LLM_MODELS
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# add API keys for other models:
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# HF FOR LLAMA? 

def get_api_function_llm(model):
    """Selects the appropriate API function based on the model name."""
    
    if model == "gpt4o":
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        def openai_gpt4o_call(messages):
            """Calls OpenAI GPT-4o API."""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            return response

        return openai_gpt4o_call

    """elif model == "claude_sonnet":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        def anthropic_claude_call(messages):
            #Calls Claude API.
            response = client.messages.create(
                model="claude-2.1",
                max_tokens=1024,
                temperature=0.7,
                messages=messages
            )
            return response

        return anthropic_claude_call

    elif model == "mistral":
        import mistralai
        client = mistralai.Client(api_key=MISTRAL_API_KEY)

        def mistral_call(messages):
            #Calls Mistral API.
            response = client.chat_completions.create(
                model="mistral-large",
                messages=messages,
                temperature=0.7
            )
            return response

        return mistral_call
        

    else:
        raise ValueError(f"Model '{model}' is not supported.")
        """
        

def call_llm(model, prompt):
    """Call the LLM API using the selected model."""
    
    llm_function = get_api_function_llm(model)

    # Format messages properly
    messages = [
        {"role": "system", "content": "You are an expert in explaining concepts."}, # General instruction to the model, overwritten with the specific prompt
        {"role": "user", "content": prompt}
    ]

    # Call the API
    response = llm_function(messages)

    # ✅ Extract only the generated text
    try:
        return response.choices[0].message.content.strip()
    except AttributeError:
        print("\n❌ Error: Unexpected response format from OpenAI API")
        print(response)  # Debugging info
        return None
