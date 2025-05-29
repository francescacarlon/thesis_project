from config import RANDOMIZED_BENCHMARK_WITH_SCORES_PATH
import spacy
import benepar
from nltk.tree import Tree
import numpy as np
import json
import traceback
import os

PARTIAL_SAVE_EVERY = 5  # Save progress every N entries

# === SETUP NLP PARSER ===
print("Loading spaCy and benepar...")
nlp = spacy.load("en_core_web_sm")
if not nlp.has_pipe("benepar"):
    benepar.download("benepar_en3")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

print("Pipeline components loaded:", nlp.pipe_names)

# === DEPTH COMPUTATION FUNCTION ===
def compute_mean_parse_tree_depth(text):
    if not text.strip():
        return None

    # Help spaCy segment properly
    if not text.strip().endswith("."):
        text += "."

    doc = nlp(text)
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
                    print(f"  ➤ {orig_key}: depth = {depth}")
                else:
                    print(f"  ⚠️ Invalid or missing text in {orig_key}")

        # Process selected (tailored) texts
        selected_texts = instance.get("selected_texts", {})
        for background, variants in selected_texts.items():
            for variant_key, variant in variants.items():
                text = variant.get("text")
                if isinstance(text, str):
                    depth = compute_mean_parse_tree_depth(text)
                    variant["parse_tree_depth_mean"] = depth
                    print(f"  ➤ {background}/{variant_key}: depth = {depth}")
                else:
                    print(f"  ⚠️ Invalid or missing text in {background}/{variant_key}")

        processed_data[key] = instance

        # Save partial progress
        if (i + 1) % PARTIAL_SAVE_EVERY == 0:
            print(f"🔃 Saving partial progress after {i+1} entries...")
            with open("partial_" + os.path.basename(RANDOMIZED_BENCHMARK_WITH_SCORES_PATH), "w", encoding="utf-8") as pf:
                json.dump(processed_data, pf, indent=2)

    except Exception as e:
        print(f"❌ Error processing {key}: {type(e).__name__} — {e}")
        traceback.print_exc()

# === FINAL SAVE ===
output_path = RANDOMIZED_BENCHMARK_WITH_SCORES_PATH.replace(".json", "_with_parse_depth.json")
print(f"\n✅ Saving full output to: {output_path}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)

print("✅ Done.")
