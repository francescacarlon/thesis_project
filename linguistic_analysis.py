# import nltk
import textstat
#import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#from nltk.translate.bleu_score import sentence_bleu
#from rouge_score import rouge_scorer
#from collections import Counter
#
## Download necessary NLTK resources if not already available
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Function to compute readability scores
def compute_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }

"""
# Function to compute POS distribution
def compute_pos_distribution(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total_words = len(words)
    pos_distribution = {tag: count / total_words for tag, count in pos_counts.items()}
    return pos_distribution

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
        "readability": compute_readability(text)
        # "pos_distribution": compute_pos_distribution(text),
    }
