import pdfplumber
import re
import os
import json

# Define the base path for your project
base_path = r"C:/Users/Francesca Carlon/Desktop/Fran_stuff/MASTER/THESIS/thesis_project"
existing_json_file = os.path.join(base_path, "dataset.json")

# Define the chapters and their sections to extract
chapters_to_extract = {
    "17": ["17.1"],
    "18": ["18.1", "18.2"],
    "21": ["21.1", "21.2"],
    "G": ["G.1", "G.2"],
    "H": ["H.1", "H.2"],
    "7": ["7.3"],
    "7a": ["7.5"],
    "7b": ["7.5"],
    "8": ["8.1"],
    "8a": ["8.5"],
    "4": ["4.1", "4.2"],
    "6": ["6.2", "6.4"],
    "13": ["13.2"],
    "13a": ["13.3"],
    "16": ["16.2"]    # Add more chapters and sections as needed
}

# Define the category mapping for each chapter
category_mapping = {
    "17": "L",
    "18": "L",
    "21": "L",
    "G": "L",
    "H": "L",
    "7": "CS",
    "7a": "CS",
    "7b": "CS",
    "8": "CS",
    "8a": "CS",
    "4": "CL",
    "6": "CL",
    "13": "CL",
    "13a": "CL",
    "16": "CL"
}

# Load the existing JSON file or initialize a new dataset
if os.path.exists(existing_json_file):
    with open(existing_json_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
else:
    data = {}

# Start ID from the next available number
current_id = max(map(int, data.keys()), default=0) + 1

# Keep track of processed chapters to avoid duplicates
existing_chapters = {entry["chapter"] for entry in data.values()}

# Loop through each chapter and its sections
for chapter, sections in chapters_to_extract.items():
    pdf_path = os.path.join(base_path, f"{chapter}.pdf")  # Assuming PDF is named by chapter number
    
    # Skip if the chapter already exists in the dataset
    if chapter in existing_chapters:
        print(f"Skipping existing chapter: {chapter}")
        continue

    # Initialize a list to store all paragraphs for the chapter
    chapter_paragraphs = []

    # Open the chapter PDF
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file {pdf_path} does not exist. Skipping chapter {chapter}.")
        continue

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue  # Skip empty pages
            
            # Check if the page contains any of the target sections
            for section in sections:
                if re.search(rf"\b{section}\b", text):
                    # Split text into paragraphs and store
                    chapter_paragraphs.extend([p.strip() for p in text.split("\n\n") if p.strip()])
    
    # Combine all paragraphs for this chapter
    combined_paragraphs = " ".join(chapter_paragraphs)
    combined_sections = ", ".join(sections)  # Combine sections into a single string

    # Get the category for this chapter
    category = category_mapping.get(chapter, "Unknown")  # Default to "Unknown" if not mapped

    # Append data to the JSON structure
    if combined_paragraphs:
        data[current_id] = {
            "chapter": chapter,
            "sections": combined_sections,  # Store sections as a single string
            "topic": f"Topic of Sections {combined_sections}",  # Replace chapter_title with topic
            "original_category": category,  # Empty field for manual completion
            "original_text": combined_paragraphs
        }
        current_id += 1
    else:
        print(f"No content found for chapter {chapter}, sections {sections}.")

# Save the updated dataset to the JSON file
with open(existing_json_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Updated dataset saved to {existing_json_file}")

