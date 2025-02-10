"""
This file calls the different LLMs with the corresponding API keys. 
"""

from dotenv import load_dotenv
import os
import openai

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # for GPT-4o and o1

# add API keys for other models:
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# HF FOR LLAMA? 

def get_api_function_llm(model):
    """Selects the appropriate API function based on the model name."""
    
    if model == "gpt4o":
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
    
    elif model == "o1":  # Explicitly handling "o1"
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        def openai_o1_call(messages):
            """Calls OpenAI GPT-4-Turbo API (formerly o1)."""
            response = client.chat.completions.create(
                model="gpt-4-turbo",  # Correct API name for o1
                messages=messages,
                temperature=0.7
            )
            return response

        return openai_o1_call

    elif model == "claude":
        pass

    else:
        raise ValueError(f"Model '{model}' is not supported.")
        
        

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
