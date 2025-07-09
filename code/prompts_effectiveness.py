"""
This module analyzes hallucination scores from a JSON dataset containing
LLM-generated tailored texts. It computes average hallucination scores by
category and prompt, both overall and per individual LLM. The module
provides functions to print results, export them to CSV and JSON, and
visualize hallucination scores with grouped bar charts for easy comparison
across LLMs and categories.

Usage:
- Analyze hallucination scores from a JSON file
- Filter analysis by specific LLMs if desired
- Export summarized results to CSV/JSON
- Plot comparative bar charts of hallucination averages
"""

import json
import pandas as pd
from collections import defaultdict
from config import LINGUISTIC_ANALYSIS_PATH
import matplotlib.pyplot as plt
import numpy as np

def analyze_hallucination_scores(json_file_path, llm_filter=None):
    """
    Analyze hallucination scores from JSON data and calculate averages by category and prompt.
    
    Args:
        json_file_path (str): Path to the JSON file containing the data
        llm_filter (list or None): List of LLMs to include (e.g., ['mistral', 'claude']). 
                                   If None, includes all LLMs.
    
    Returns:
        tuple: (averages_dict, detailed_scores_dict, llm_breakdown_dict)
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    # Dictionary to store all scores for each category-prompt combination
    category_prompt_scores = defaultdict(list)
    # Dictionary to store scores broken down by LLM
    llm_breakdown = defaultdict(lambda: defaultdict(list))
    
    # Track which LLMs we find
    found_llms = set()
    
    # Process each entry in the JSON
    for key, entry in data.items():
        if 'tailored_texts' in entry:
            
            # Process each LLM
            for llm_name, llm_data in entry['tailored_texts'].items():
                found_llms.add(llm_name)
                
                # Skip if we're filtering LLMs and this one isn't included
                if llm_filter and llm_name not in llm_filter:
                    continue
                
                # Process each category (CS, CL, L, etc.)
                for category, category_data in llm_data.items():
                    
                    # Process each prompt within the category
                    for prompt_key, prompt_data in category_data.items():
                        if ('hallucination_scores' in prompt_data and 
                            'hallucinations_overall_average' in prompt_data['hallucination_scores']):
                            
                            score = prompt_data['hallucination_scores']['hallucinations_overall_average']
                            combination_key = f"{prompt_key}_{category}"
                            
                            # Add to overall scores
                            category_prompt_scores[combination_key].append(score)
                            
                            # Add to LLM-specific breakdown
                            llm_breakdown[llm_name][combination_key].append(score)
    
    # Calculate overall averages
    averages = {}
    for combination, scores in category_prompt_scores.items():
        avg_score = sum(scores) / len(scores)
        averages[f"avg_{combination}"] = round(avg_score, 3)
    
    # Calculate LLM-specific averages
    llm_averages = {}
    for llm_name, llm_scores in llm_breakdown.items():
        llm_averages[llm_name] = {}
        for combination, scores in llm_scores.items():
            avg_score = sum(scores) / len(scores)
            llm_averages[llm_name][f"avg_{combination}"] = round(avg_score, 3)
    
    print(f"Found LLMs in data: {sorted(found_llms)}")
    if llm_filter:
        print(f"Filtering to: {llm_filter}")
    
    return averages, category_prompt_scores, llm_averages

def print_results(averages, detailed_scores, llm_averages=None):
    """Print the results in a formatted way"""
    
    print("=== OVERALL HALLUCINATION SCORE AVERAGES (ALL LLMs) ===\n")
    
    # Group by category for better readability
    categories = {}
    for key, avg in averages.items():
        # Extract category from key (e.g., avg_prompt1_CS -> CS)
        parts = key.split('_')
        if len(parts) >= 3:
            category = parts[-1]  # Last part is category
            prompt = '_'.join(parts[1:-1])  # Middle parts are prompt
            
            if category not in categories:
                categories[category] = {}
            categories[category][prompt] = avg
    
    # Print grouped results
    for category in sorted(categories.keys()):
        print(f"Category: {category}")
        print("-" * 20)
        for prompt in sorted(categories[category].keys()):
            key = f"avg_{prompt}_{category}"
            score_count = len(detailed_scores[f"{prompt}_{category}"])
            print(f"  {key}: {categories[category][prompt]} (based on {score_count} entries)")
        print()
    
    # Print LLM-specific results if available
    if llm_averages:
        print("\n=== LLM-SPECIFIC AVERAGES ===\n")
        for llm_name in sorted(llm_averages.keys()):
            print(f"LLM: {llm_name.upper()}")
            print("-" * 30)
            
            # Group by category for this LLM
            llm_categories = {}
            for key, avg in llm_averages[llm_name].items():
                parts = key.split('_')
                if len(parts) >= 3:
                    category = parts[-1]
                    prompt = '_'.join(parts[1:-1])
                    
                    if category not in llm_categories:
                        llm_categories[category] = {}
                    llm_categories[category][prompt] = avg
            
            for category in sorted(llm_categories.keys()):
                print(f"  Category {category}:")
                for prompt in sorted(llm_categories[category].keys()):
                    print(f"    avg_{prompt}_{category}: {llm_categories[category][prompt]}")
                print()
            print()
    
    # Print all results in order
    print("=== ALL RESULTS (ALPHABETICAL) ===")
    for key in sorted(averages.keys()):
        combination_key = key[4:]  # Remove 'avg_' prefix
        score_count = len(detailed_scores[combination_key])
        print(f"{key}: {averages[key]} (n={score_count})")

def export_to_csv(averages, detailed_scores, llm_averages, output_file):
    """Export results to CSV"""
    
    results_data = []
    
    # Overall results
    for key, avg in averages.items():
        combination_key = key[4:]  # Remove 'avg_' prefix
        parts = combination_key.split('_')
        category = parts[-1]
        prompt = '_'.join(parts[:-1])
        score_count = len(detailed_scores[combination_key])
        scores = detailed_scores[combination_key]
        
        results_data.append({
            'llm': 'ALL',
            'metric': key,
            'category': category,
            'prompt': prompt,
            'average': avg,
            'count': score_count,
            'min_score': min(scores),
            'max_score': max(scores)
        })
    
    # LLM-specific results
    for llm_name, llm_scores in llm_averages.items():
        for key, avg in llm_scores.items():
            combination_key = key[4:]  # Remove 'avg_' prefix
            parts = combination_key.split('_')
            category = parts[-1]
            prompt = '_'.join(parts[:-1])
            
            results_data.append({
                'llm': llm_name,
                'metric': key,
                'category': category,
                'prompt': prompt,
                'average': avg,
                'count': 'N/A',  # Individual LLM counts would need separate tracking
                'min_score': 'N/A',
                'max_score': 'N/A'
            })
    
    df = pd.DataFrame(results_data)
    df.to_csv(output_file, index=False)
    print(f"\nResults exported to {output_file}")

# Example usage
if __name__ == "__main__":
    # Use the path from config
    json_file = LINGUISTIC_ANALYSIS_PATH
    
    try:
        # Analyze the data (all LLMs)
        averages, detailed_scores, llm_averages = analyze_hallucination_scores(json_file)
        
        # Print results
        print_results(averages, detailed_scores, llm_averages)
        
        # Export to CSV (optional)
        export_to_csv(averages, detailed_scores, llm_averages, "hallucination_averages.csv")
        
        # Export averages to JSON (optional)
        export_data = {
            'overall_averages': averages,
            'llm_specific_averages': llm_averages
        }
        with open("hallucination_averages.json", "w") as f:
            json.dump(export_data, f, indent=2)
        print("Averages exported to hallucination_averages.json")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        print("Please check that LINGUISTIC_ANALYSIS_PATH in config.py points to the correct file.")
    except Exception as e:
        print(f"Error processing file: {e}")

# Quick usage functions
def get_averages_from_config(llm_filter=None):
    """Get averages using the path from config file"""
    averages, _, llm_averages = analyze_hallucination_scores(LINGUISTIC_ANALYSIS_PATH, llm_filter)
    return averages, llm_averages

def analyze_and_print(llm_filter=None):
    """Analyze and print results using config path"""
    averages, detailed_scores, llm_averages = analyze_hallucination_scores(LINGUISTIC_ANALYSIS_PATH, llm_filter)
    print_results(averages, detailed_scores, llm_averages)
    return averages, llm_averages

def get_specific_llm_averages(llm_names):
    """Get averages for specific LLMs only"""
    return analyze_and_print(llm_filter=llm_names)

# Alternative: Process from JSON string instead of file
def analyze_from_json_string(json_string):
    """Analyze hallucination scores from JSON string instead of file"""
    data = json.loads(json_string)
    
    category_prompt_scores = defaultdict(list)
    
    for key, entry in data.items():
        if 'tailored_texts' in entry and 'mistral' in entry['tailored_texts']:
            mistral_data = entry['tailored_texts']['mistral']
            
            for category, category_data in mistral_data.items():
                for prompt_key, prompt_data in category_data.items():
                    if ('hallucination_scores' in prompt_data and 
                        'hallucinations_overall_average' in prompt_data['hallucination_scores']):
                        
                        score = prompt_data['hallucination_scores']['hallucinations_overall_average']
                        combination_key = f"{prompt_key}_{category}"
                        category_prompt_scores[combination_key].append(score)
    
    averages = {}
    for combination, scores in category_prompt_scores.items():
        avg_score = sum(scores) / len(scores)
        averages[f"avg_{combination}"] = round(avg_score, 3)
    
    return averages

def plot_grouped_llm_averages(llm_averages, overall_averages=None, title_suffix="Hallucination Scores per Prompt"):
    """
    Create grouped bar charts per category showing hallucination averages for each LLM.
    """
    # Collect all LLMs and prompt-category combinations
    llm_names = sorted(llm_averages.keys())
    all_combinations = set()

    # Build reverse index: {category: {prompt: {llm: avg}}}
    data = defaultdict(lambda: defaultdict(dict))
    for llm, scores in llm_averages.items():
        for key, avg in scores.items():
            parts = key.split('_')
            if len(parts) >= 3:
                category = parts[-1]
                prompt = '_'.join(parts[1:-1])
                data[category][prompt][llm] = avg
                all_combinations.add((prompt, category))

    # Add 'ALL' values if provided
    if overall_averages:
        for key, avg in overall_averages.items():
            parts = key.split('_')
            if len(parts) >= 3:
                category = parts[-1]
                prompt = '_'.join(parts[1:-1])
                data[category][prompt]['ALL'] = avg
                if 'ALL' not in llm_names:
                    llm_names.append('ALL')

    # Plot each category
    for category, prompts in data.items():
        labels = sorted(prompts.keys())
        x = np.arange(len(labels))  # label locations
        width = 0.8 / len(llm_names)  # bar width depends on number of LLMs

        plt.figure(figsize=(12, 6))
        for i, llm in enumerate(llm_names):
            scores = [prompts[prompt].get(llm, 0) for prompt in labels]
            plt.bar(x + i * width, scores, width, label=llm)

        plt.ylabel('Hallucination Avg Score')
        plt.title(f"{category} - {title_suffix}")
        plt.xticks(x + width * len(llm_names) / 2, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    averages, detailed_scores, llm_averages = analyze_hallucination_scores(LINGUISTIC_ANALYSIS_PATH)
    print_results(averages, detailed_scores, llm_averages)

    # Plotting:
    plot_grouped_llm_averages(llm_averages, overall_averages=averages)

