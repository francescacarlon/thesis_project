import pdfplumber
import re
import os
import json

# Define the base path for your project
base_path = r"C:/Users/Francesca Carlon/Desktop/Fran_stuff/MASTER/THESIS/thesis_project"
existing_json_file = os.path.join(base_path, "dataset.json")

# Define the chapters and sections to extract
chapters_to_extract = {
    "17": ["17.1"],
    "18": ["18.1", "18.2"],
    "21": ["21.1", "21.2"],
    "G": ["G.1", "G.2"],
    "H": ["H.1", "H.2"]  # Add more chapters and sections as needed
}

# Load the existing JSON file or initialize a new dataset
if os.path.exists(existing_json_file):
    with open(existing_json_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
else:
    data = {}

# Start ID from the next available number
current_id = max(map(int, data.keys()), default=0) + 1

# Loop through each chapter and its sections
for chapter, sections in chapters_to_extract.items():
    pdf_path = os.path.join(base_path, f"{chapter}.pdf")  # Assuming PDF is named by chapter number
    
    # Initialize a list to store the paragraphs for this chapter's sections
    paragraphs = []
    
    # Open the chapter PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()  # Extract text from the page
            
            # Check if the page contains any of the target sections
            for section in sections:
                if re.search(rf"\b{section}\b", text):
                    # Split text into paragraphs and store
                    for paragraph in text.split("\n\n"):
                        if paragraph.strip():
                            paragraphs.append(paragraph.strip())
    
    # Combine paragraphs for this chapter and its sections
    combined_paragraphs = " ".join(paragraphs)
    
    # Append data to the JSON structure
    data[current_id] = {
        "chapter": chapter,
        "section": " & ".join(sections),
        "chapter_title": f"Title of Sections {', '.join(sections)}",  # Placeholder for manual update
        "original_category": "",  # Empty field for manual completion
        "original_text": combined_paragraphs
    }
    
    # Increment ID for the next entry
    current_id += 1

# Save the updated dataset to the JSON file
with open(existing_json_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Updated dataset saved to {existing_json_file}")



"""import pdfplumber
import re
import pandas as pd
import os
import json

# Define the base path for your project
base_path = r"C:/Users/Francesca Carlon/Desktop/Fran_stuff/MASTER/THESIS/thesis_project"

# Define the PDF file name for the whole book
chapter_file = "chapter17.pdf"

# Construct the full path to the PDF
pdf_path = os.path.join(base_path, chapter_file)

# Initialize a list to store the extracted paragraphs
paragraphs = []

# Open the PDF
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()  # Extract text from the page
        
        # Find the section of interest (e.g., Section 17.1)
        if "17.1" in text:
            # Split text into paragraphs (assuming paragraphs are separated by double newlines)
            for paragraph in text.split("\n\n"):
                if paragraph.strip():  # Ignore empty lines
                    paragraphs.append(paragraph.strip())

# Combine all paragraphs into one text block
combined_paragraphs = " ".join(paragraphs)

# Prepare the data for JSON
data = {
    1: {                    # ID as the key
        "chapter": "17",
        "section": "17.1",
        "chapter_title": "Title of Subsection",  # Placeholder; to be completed manually
        "original_category": "",  # Empty field for manual completion
        "original_text": combined_paragraphs
    }
}

# Save the data to a JSON file
json_output_path = os.path.join(base_path, "chapter_17_1_data.json")
with open(json_output_path, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Filtered paragraphs saved to {json_output_path}")"""
