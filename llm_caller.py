from src.config import LLM_MODELS
from your_module import get_api_function_llm  # Import your function selector

def call_llm(model, prompt):
    """Call the LLM API using the selected model."""
    if model not in LLM_MODELS:
        raise ValueError(f"Invalid model: {model}")
    
    llm_function = get_api_function_llm(model)
    response = llm_function(prompt=prompt)
    return response
