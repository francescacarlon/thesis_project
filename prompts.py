from src.config import CATEGORIES

def make_prompt(original_text, category):
    """Generate a tailored prompt based on the category."""
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category}")
    return f"Please tailor this explanation for {category} students:\n{original_text}"
