import json
from config import RANDOMIZED_BENCHMARK_PATH, TEXT_PAGES_PATH

# Load benchmark data
with open(RANDOMIZED_BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# Only generate HTML for CS and L categories, and only if they exist
for instance_code, data in benchmark_data.items():
    original_text_title = data.get("original_text_title", "Unknown Title")
    original_text = data.get("original_text", "No text available")
    original_category = data.get("original_category", "Unknown Category")
    selected_texts = data.get("selected_texts", {})

    for target_category in ["CS", "L"]:  # CL no longer relevant
        texts = selected_texts.get(target_category)
        if not texts:
            continue  # Skip if no tailored texts for this category

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
                    <p><strong>Original Category:</strong> {original_category}</p>
                    <p><strong>Target Category:</strong> {target_category}</p>
                    <div class="box">
                        <h2>Original Text</h2>
                        <p>{original_text}</p>
                    </div>
                    <h2>{target_category} Tailored Explanations</h2>
            """)

            for key, text in texts.items():
                f.write(f"""
                <div class="box">
                    <p class="category">{key}</p>
                    <p>{text}</p>
                </div>
                """)

            f.write("</div></body></html>")

print(f"Tailored explanation HTML files for CS and L generated in: {TEXT_PAGES_PATH}")
