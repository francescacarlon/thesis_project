import json

def load_dataset(file_path):
    """Load JSON dataset from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data, file_path):
    """Save dataset to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
