"""
File with the linguistic metrics functions.
For each text:
- readability: Flesch scores;
- POS distribution.

For similarity scores between texts:
- Jaccard similarity;
- Cosine similarity;
- BLEU and ROUGE scores.
"""

import nltk
import textstat
#import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_scorer
from collections import Counter
#
# Download necessary NLTK resources if not already available
#nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

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


"""
# Function to compute Jaccard similarity between two texts
def compute_jaccard_similarity(text1, text2):
    set1 = set(nltk.word_tokenize(text1.lower()))
    set2 = set(nltk.word_tokenize(text2.lower()))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

# Function to compute Cosine similarity with CountVectorizer
def compute_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Function to compute BLEU score
def compute_bleu(reference, candidate):
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    return sentence_bleu(reference_tokens, candidate_tokens)

# Function to compute ROUGE scores
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge-1', 'rouge-2', 'rouge-l'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: value.fmeasure for key, value in scores.items()}
    """

# Function to analyze a text
def analyze_text(text):
    return {
        "readability": compute_readability(text),
        "pos": compute_pos_distribution(text)
    }
