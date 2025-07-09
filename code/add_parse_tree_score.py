"""
=== Parse Tree Depth Computation ===

This script computes the **mean constituency parse tree depth** of each explanation text
(original and tailored) using the Benepar parser integrated with spaCy.

What is computed:
- For each sentence in the text, we generate a constituency parse tree (using Benepar).
- The depth of a parse tree is defined as the number of nodes from the root to the deepest leaf.
- For example, a sentence like "Cats sleep." has a shallow tree, while a complex nested sentence
  will produce a deeper tree.
- For a given text (which may include multiple sentences), the **mean parse tree depth** is calculated
  by averaging the depths of all its sentence-level trees.

This value is used as a proxy for **syntactic complexity**: deeper trees typically indicate more
complex syntax.

Why it's useful:
- Helps compare original vs. paraphrased explanations.
- Useful for linguistic analysis, readability assessment, and tailoring content for users
  from different backgrounds.

"""

from config import RANDOMIZED_BENCHMARK_WITH_SCORES_PATH
import spacy
import benepar
from nltk.tree import Tree
import numpy as np
import json
import traceback
import os
import re

PARTIAL_SAVE_EVERY = 5  # Save progress every N entries

# === SETUP NLP PARSER ===
print("Loading spaCy and benepar...")
nlp = spacy.load("en_core_web_sm")
if not nlp.has_pipe("benepar"):
    benepar.download("benepar_en3")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

print("Pipeline components loaded:", nlp.pipe_names)


# === CLEANING FUNCTION ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\xa0', ' ')
    return text


# === DEPTH COMPUTATION FUNCTION ===
def compute_mean_parse_tree_depth(text):
    """
    Computes the mean depth of constituency parse trees for all sentences in a given text.

    - Uses Benepar (Berkeley Neural Parser) for constituency parsing via spaCy.
    - Each sentence is parsed into a tree, and its depth is calculated as:
        tree height - 1 (to exclude the root node itself).
    - Returns the average depth across all parseable sentences.
    - Returns None if no valid parse trees are found.

    This depth is used to assess the syntactic complexity of the text.
    """
    if not text.strip():
        return None

    text = clean_text(text)
    if not text.endswith('.'):
        text += "."

    try:
        doc = nlp(text)
    except Exception as e:
        print(f"spaCy/Benepar failed to parse:\n'{text[:100]}...'\nReason: {e}")
        return None

    depths = []
    for sent in doc.sents:
        try:
            parse_string = sent._.parse_string
            parse_tree = Tree.fromstring(parse_string)
            depth = parse_tree.height() - 1
            depths.append(depth)
        except Exception as e:
            print(f"⚠️ Parse error on sentence: '{sent.text}'\nReason: {e}")
            continue

    return float(np.mean(depths)) if depths else None


# === MAIN PROCESSING ===
print(f"Loading input file: {RANDOMIZED_BENCHMARK_WITH_SCORES_PATH}")
with open(RANDOMIZED_BENCHMARK_WITH_SCORES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from input file.")
processed_data = {}

for i, (key, instance) in enumerate(data.items()):
    print(f"\nProcessing item {i+1}/{len(data)}: {key}")
    try:
        # Process original texts
        for orig_key in list(instance.keys()):
            if orig_key.startswith("original_text_"):
                content = instance[orig_key]
                text = content.get("text") if isinstance(content, dict) else content
                if isinstance(text, str):
                    depth = compute_mean_parse_tree_depth(text)
                    instance[f"parse_tree_depth_mean_{orig_key}"] = depth
                    print(f"  {orig_key}: depth = {depth}")
                else:
                    print(f"Invalid or missing text in {orig_key}")

        # Process selected (tailored) texts
        selected_texts = instance.get("selected_texts", {})
        for background, variants in selected_texts.items():
            for variant_key, variant in variants.items():
                text = variant.get("text")
                if isinstance(text, str):
                    depth = compute_mean_parse_tree_depth(text)
                    variant["parse_tree_depth_mean"] = depth
                    print(f"  {background}/{variant_key}: depth = {depth}")
                else:
                    print(f"Invalid or missing text in {background}/{variant_key}")

        processed_data[key] = instance
        print(f"Added {key} to processed_data.")

        # Save partial progress
        if (i + 1) % PARTIAL_SAVE_EVERY == 0:
            partial_path = "partial_" + os.path.basename(RANDOMIZED_BENCHMARK_WITH_SCORES_PATH)
            print(f"Saving partial progress to: {partial_path}")
            with open(partial_path, "w", encoding="utf-8") as pf:
                json.dump(processed_data, pf, indent=2)

    except Exception as e:
        print(f"Error processing {key}: {type(e).__name__} — {e}")
        traceback.print_exc()
        with open("parse_errors.log", "a", encoding="utf-8") as logf:
            logf.write(f"{key}: {type(e).__name__} — {e}\n")


# === FINAL SAVE ===
output_path = os.path.splitext(RANDOMIZED_BENCHMARK_WITH_SCORES_PATH)[0] + "_parsed_depths.json"
print(f"\nSaving full output to: {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)

print("Done.")
