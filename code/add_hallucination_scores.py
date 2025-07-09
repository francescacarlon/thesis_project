"""
LLM-Based Hallucination Scoring Script

This script evaluates the hallucination levels of paraphrased (tailored) texts 
by comparing them to original texts using multiple LLMs (GPT-4, Claude, Mistral). 
Each model provides scores for:
- **Relevance**: how well the tailored text matches the topic of the original
- **Consistency**: how factually and logically aligned it is with the original

Main features:
- Sends input to each LLM with a hallucination detection prompt
- Extracts and stores relevance and consistency scores (1–5)
- Computes an average score per LLM and overall
- Saves results to `linguistic_analysis_with_scores.json`

Requires valid API keys for OpenAI, Anthropic, and Mistral.
"""

import json
import os
from openai import OpenAI
from anthropic import Anthropic
import requests
import time

# Initialize API clients
client = OpenAI()
anthropic = Anthropic()

def get_gpt4_score(text, original_text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": detect_hallucination_prompt},
            {"role": "user", "content": f"Original text: {original_text}\n\nTailored text: {text}\n\nPlease provide relevance and consistency scores (1-5) and explain your reasoning."}
        ]
    )
    return response.choices[0].message.content

def get_claude_score(text, original_text):
    response = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Original text: {original_text}\n\nTailored text: {text}\n\nPlease provide relevance and consistency scores (1-5) and explain your reasoning."}
        ]
    )
    return response.content[0].text

def get_mistral_score(text, original_text):
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": detect_hallucination_prompt},
                {"role": "user", "content": f"Original text: {original_text}\n\nTailored text: {text}\n\nPlease provide relevance and consistency scores (1-5) and explain your reasoning."}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def extract_scores(response_text):
    # Simple extraction of scores from response text
    # This might need to be adjusted based on actual response format
    try:
        # Look for numbers between 1-5 in the text
        import re
        scores = re.findall(r'\b[1-5]\b', response_text)
        if len(scores) >= 2:
            relevance = float(scores[0])
            consistency = float(scores[1])
            return relevance, consistency
    except:
        pass
    return None, None

def process_json_file():
    # Read the existing JSON file
    with open('data/linguistic_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each entry
    for entry in data:
        original_text = entry.get('original_text', '')
        tailored_texts = entry.get('tailored_texts', {})
        
        # Initialize scores dictionary
        entry['hallucination_scores'] = {}
        
        # Process each LLM's tailored text
        for llm, text in tailored_texts.items():
            if not text:
                continue
                
            print(f"Processing {llm} for text...")
            
            # Get scores based on LLM
            if llm == 'gpt4o':
                response = get_gpt4_score(text, original_text)
            elif llm == 'claude':
                response = get_claude_score(text, original_text)
            elif llm == 'mistral':
                response = get_mistral_score(text, original_text)
            else:
                continue
                
            relevance, consistency = extract_scores(response)
            
            if relevance and consistency:
                entry['hallucination_scores'][llm] = {
                    'relevance': relevance,
                    'consistency': consistency,
                    'average': (relevance + consistency) / 2
                }
            
            # Add delay to avoid rate limits
            time.sleep(1)
        
        # Calculate overall average across all LLMs
        if entry['hallucination_scores']:
            all_averages = [scores['average'] for scores in entry['hallucination_scores'].values()]
            entry['overall_hallucination_average'] = sum(all_averages) / len(all_averages)

    # Save the modified JSON file
    with open('data/linguistic_analysis_with_scores.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_json_file() 