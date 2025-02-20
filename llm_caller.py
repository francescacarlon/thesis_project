"""
This file calls the different LLMs with the corresponding API keys. 
"""

from dotenv import load_dotenv
import os
import openai
import anthropic
import requests

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # for GPT-4o and o1
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") # for Claude
HF_API_KEY=os.getenv("HF_API_KEY") # for llama

# add API keys for other models:
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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
        
    elif model == "o1-preview":  # Explicitly handling "o1-preview"
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        def openai_o1_call(messages):
            """Calls OpenAI's o1-preview model, ensuring it does not include 'system' messages."""
            
            # ✅ Filter out system messages
            filtered_messages = [msg for msg in messages if msg.get("role") != "system"]

            response = client.chat.completions.create(
                model="o1-preview",  
                messages=filtered_messages  # ✅ Use filtered messages
                #temperature=0.7
            )
            return response

        return openai_o1_call



    elif model == "claude":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        def claude_call(user_message):
            """Calls Anthropic Claude API correctly."""
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Make sure this model is available in your account
                    max_tokens=1024,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": str(user_message)}]  # ✅ Ensure user_message is a string
                        }
                    ]  
                )
                return response.content[0].text.strip()  # ✅ Extract text from response
            except Exception as e:
                print(f"\n❌ Error calling Anthropic API: {e}")
                return None

  
        return claude_call

    
    elif model == "llama":
        def llama_call(user_message):
            """Calls a LLaMA model hosted on Hugging Face's API."""
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {
                "inputs": user_message,
                "parameters": {"max_length": 1024, "temperature": 0.7},
            }

            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct",  # Model chosen
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    return response.json()[0]["generated_text"].strip()
                else:
                    print(f"\n❌ Error calling LLaMA API: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                return None

        return llama_call

    else:
        raise ValueError(f"Model '{model}' is not supported.")
        
        

def call_llm(model, prompt):
    """Call the LLM API using the selected model and extract responses correctly."""
    
    llm_function = get_api_function_llm(model)

    if model in ["gpt4o", "o1-preview"]:
        # ✅ OpenAI models require formatted message lists
        messages = [
            {"role": "system", "content": "You are an expert in explaining concepts."}, 
            {"role": "user", "content": prompt}
        ]
        response = llm_function(messages)

        # ✅ Extract response correctly for OpenAI API
        try:
            return response.choices[0].message.content.strip()
        except AttributeError:
            print("\n❌ Error: Unexpected response format from OpenAI API")
            print(response)  # Debugging info
            return None

    elif model == "claude":
        # ✅ Claude only needs a plain string input (no role messages needed)
        response = llm_function(prompt)

        # ✅ Extract response correctly for Claude API
        try:
            if response:
                return response.strip()  # Claude responses are already strings
            else:
                print("\n❌ Error: Claude API returned None")
                return None
        except AttributeError:
            print("\n❌ Error: Unexpected response format from Claude API")
            print(response)  # Debugging info
            return None

    elif model == "llama":
        # ✅ LLaMA also takes a plain string input
        response = llm_function(prompt)

        # ✅ Extract response correctly for LLaMA API
        try:
            if response:
                return response.strip()  # LLaMA responses are already strings
            else:
                print("\n❌ Error: LLaMA API returned None")
                return None
        except AttributeError:
            print("\n❌ Error: Unexpected response format from LLaMA API")
            print(response)  # Debugging info
            return None

    else:
        print("\n❌ Error: Unsupported model:", model)
        return None

