"""
File with the linguistic metrics functions.
For each text:
- length (token count);
- readability: Flesch scores;
- POS distribution.

For similarity scores between texts:
- Cosine similarity;
- Jaccard similarity;
- BLEU and ROUGE scores. 
"""

import nltk
import textstat
from collections import Counter

nltk.download('averaged_perceptron_tagger_eng')

# Function to compute tokens: length of the text
def count_tokens(text):
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha()]  # No punctuation
    return len(words)

# Function to compute readability scores
def compute_readability(text):
    """Computes readability scores, ensuring text is a valid string."""
    if not isinstance(text, str) or not text.strip():  # Check for None or empty strings
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0}

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }



# Function to compute POS distribution
def compute_pos_distribution(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)
    
    # Filter out unwanted tags and punctuation
    filtered_tags = [
        tag for word, tag in pos_tags 
        if tag not in {"PRP", "PRP$", "WDT", "TO", "POS", "WP", "WRB", "PDT", "EX", "WP$"} and word.isalpha()
    ]
    
    # Group POS tags according to my rules
    grouped_tags = []
    for tag in filtered_tags:
        if tag in {"VB", "VBP", "VBZ"}:
            grouped_tags.append("VB")  # Group all present tense verbs
        elif tag in {"VBN", "VBD"}:
            grouped_tags.append("VBN")  # Group past participles and past tense
        elif tag in {"NN", "NNS"}:
            grouped_tags.append("NN")  # Group singular and plural nouns
        elif tag in {"JJ", "JJS", "JJR"}:
            grouped_tags.append("JJ") # Group adjectives, superlative adjectives and comparative adjectives
        elif tag in {"RB", "RBR", "RBS"}:
            grouped_tags.append("RB") # Group adverbs, comparative adverbs, superlative adverbs
        else:
            grouped_tags.append(tag)  # Keep other tags as-is

    # Count the POS tags
    pos_counts = Counter(grouped_tags)
    
    # Calculate the distribution as percentages
    total_words = sum(pos_counts.values())
    pos_distribution = {tag: count / total_words for tag, count in pos_counts.items()}
    
    return pos_distribution



# Function to analyze a text
def analyze_text(text):
    return {
        "token_count": count_tokens(text),
        "readability": compute_readability(text),
        "pos": compute_pos_distribution(text)
    }