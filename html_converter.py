import json
from itertools import islice
from config import RANDOMIZED_BENCHMARK_PATH, TEXT_PAGES_PATH

# Load benchmark data
with open(RANDOMIZED_BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Only generate HTML for CS and L categories, and only if they exist
for instance_code, data in benchmark_data.items():
    original_text_title = data.get("topic", "Unknown Title")
    original_text = data.get("original_text", "No text available")
    selected_texts = data.get("selected_texts", {})

    for target_category in ["CS", "L"]:  # CL excluded
        category_texts = selected_texts.get(target_category)
        if not category_texts:
            print(f"⚠️ Skipping {instance_code} - {target_category}: No texts found.")
            continue

        # --- Write the original file as before ---
        html_file_path = TEXT_PAGES_PATH / f"{instance_code}_{target_category}.html"
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{original_text_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; padding: 20px; line-height: 1.6; }}
                    .container {{ max-width: 800px; margin: auto; }}
                    h1, h2 {{ color: #333; }}
                    .box {{ background: #f4f4f4; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                    .category {{ font-weight: bold; color: #0056b3; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{original_text_title}</h1>
                    <p><strong>Instance Code:</strong> {instance_code}</p>
                    <p><strong>Target Category:</strong> {target_category}</p>
                    <div class="box">
                        <h2>Original Text</h2>
                        <p>{original_text}</p>
                    </div>
                    <h2>{target_category} Tailored Explanations</h2>
            """)

            for prompt_id, entry in category_texts.items():
                text = entry.get("text", "")
                f.write(f"""
                <div class="box">
                    <p class="category">{prompt_id}</p>
                    <p>{text}</p>
                </div>
                """)

            f.write("</div></body></html>")

        print(f"✅ Saved: {html_file_path.name}")

        # --- Build unified list: original + tailored ---
        items = [("ORIGINAL", {"text": original_text})] + list(category_texts.items())
        n = len(items)

        for i in range(3):  # v1, v2, v3
            rotated_items = list(islice(items, i, n)) + list(islice(items, 0, i))

            rotated_html_path = TEXT_PAGES_PATH / f"{instance_code}_{target_category}_v{i+1}.html"
            with open(rotated_html_path, "w", encoding="utf-8") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{original_text_title} (Rotation v{i+1})</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; padding: 20px; line-height: 1.6; }}
                        .container {{ max-width: 800px; margin: auto; }}
                        h1, h2 {{ color: #333; }}
                        .box {{ background: #f4f4f4; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                        .category {{ font-weight: bold; color: #0056b3; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>{original_text_title} (Rotation v{i+1})</h1>
                        <p><strong>Instance Code:</strong> {instance_code}</p>
                        <p><strong>Target Category:</strong> {target_category}</p>
                        
                        <h2>Explanations</h2>
                """)

                for prompt_id, entry in rotated_items:
                    text = entry.get("text", "")
                    # Build the code name title
                    if prompt_id == "ORIGINAL":
                        topic_number = instance_code.replace("T", "")
                        code_name = f"{target_category}_o_{topic_number}"
                    else:
                        try:
                            parts = prompt_id.split("_", 2)  # e.g., CS_GPT4_prompt3
                            model = parts[1]                 # GPT4
                            prompt_key = parts[2]           # prompt3

                            # Extract number from prompt_key (e.g., "prompt3" → "3")
                            prompt_num = ''.join(filter(str.isdigit, prompt_key))

                            # Use first lowercase letter of model
                            code_name = f"{target_category}_{model[0].lower()}_{prompt_num}"
                        except Exception as e:
                            code_name = f"{target_category}_x_unknown"
                            print(f"[WARNING] Could not parse prompt_id '{prompt_id}' → {e}")

                    f.write(f"""
                    <div class="box">
                        <p class="category">{code_name}</p>
                        <p>{text}</p>
                    </div>
                    """)


