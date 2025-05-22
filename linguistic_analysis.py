"""
File with the linguistic metrics functions.
For each text:
- length (token count);
- readability: Flesch scores;
- POS distribution.

For similarity scores between the original text and the tailored texts:
- Cosine similarity; 
- BERT score; 
- BLEU and ROUGE scores. 

"""

import nltk
import textstat
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from bert_score import score
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

# Load the Sentence Transformer model and tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to compute tokens: length of the text
def count_tokens(text):
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha()]  # No punctuation
    return len(words)

# Function to compute readability scores
def compute_readability(text):
    """Computes readability scores, ensuring text is a valid string."""
    if not isinstance(text, str) or not text.strip():  # Check for None or empty strings
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0, "smog_index": 0}

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text)
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


# Function to split long texts into overlapping chunks
def chunk_text(text, max_tokens=256, overlap=50):
    """Splits a long text into overlapping chunks of max_tokens tokens each."""
    tokens = tokenizer.tokenize(text)
    
    if len(tokens) <= max_tokens:
        return [text]  # If short, return the full text as one chunk
    
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokens[start : start + max_tokens]  # Get max_tokens chunk
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # Move forward with overlap
    
    return chunks

# Function to encode long texts by averaging chunk embeddings 
# as the SentenceBERT model used (all-MiniLM-L6-v2) truncates input text longer than 256 word pieces 
def encode_long_text(text):
    """Encodes a long text by splitting it into chunks and averaging their embeddings."""
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return torch.mean(embeddings, dim=0)  # Average embeddings

# Function to compute cosine similarity without truncation
def compute_cosine_similarity(original_text, tailored_text):
    """Computes cosine similarity between original and paraphrased texts without truncation."""
    
    # Encode full texts without truncation
    embedding1 = encode_long_text(original_text)
    embedding2 = encode_long_text(tailored_text)

    # Compute cosine similarity
    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2).item()

    return {"cosine_similarity": cosine_sim}


# Function to compute BERTScore using microsoft/deberta-xlarge-mnli
# use the embeddings from this strong DeBERTa model to compute token-level similarity.
def compute_bertscore(original_text, tailored_text, model_name="microsoft/deberta-xlarge-mnli"):
    if not original_text.strip() or not tailored_text.strip():
        print("⚠️ Skipping BERTScore computation: Empty input detected.")
        return {"bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None}

    # Split long texts into chunks (to prevent truncation)
    original_chunks = chunk_text(original_text, max_tokens=512, overlap=50)
    tailored_chunks = chunk_text(tailored_text, max_tokens=512, overlap=50)

    # Ensure both lists have the same number of chunks
    max_len = max(len(original_chunks), len(tailored_chunks))
    original_chunks += [""] * (max_len - len(original_chunks))
    tailored_chunks += [""] * (max_len - len(tailored_chunks))

    # Compute BERTScore for each chunk
    scores = {"precision": [], "recall": [], "f1": []}

    for orig_chunk, tail_chunk in zip(original_chunks, tailored_chunks):
        if not orig_chunk.strip() or not tail_chunk.strip():
            continue  # Skip empty chunks

        P, R, F1 = score(
            [tail_chunk], 
            [orig_chunk], 
            model_type=model_name,  
            lang="en", 
            rescale_with_baseline=True,  # ✅ Prevents inflated scores
            batch_size=8  
        )

        scores["precision"].append(P.item())
        scores["recall"].append(R.item())
        scores["f1"].append(F1.item())

    if not scores["precision"]:  # In case all chunks were empty
        return {"bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None}

    # Return the averaged BERTScore
    return {
        "bertscore_precision": sum(scores["precision"]) / len(scores["precision"]),
        "bertscore_recall": sum(scores["recall"]) / len(scores["recall"]),
        "bertscore_f1": sum(scores["f1"]) / len(scores["f1"]),
    }

# Function to compute BLEU score
def compute_bleu_score(original_text, tailored_text):
    """Computes BLEU score for similarity checking.
    BLEU-1 formula:
    BLEU = BP * p1
    where:
      - p1 = modified unigram precision (clipped count of matching unigrams / total unigrams in candidate)
      - BP = brevity penalty = 1 if candidate length > reference length,
                                else exp(1 - reference_length / candidate_length)
    Smoothing is applied using nltk.translate.bleu_score.SmoothingFunction().method1
"""
    if not original_text.strip() or not tailored_text.strip():
        return {"bleu_score": None}  

    original_tokens = [nltk.word_tokenize(original_text)]  
    tailored_tokens = nltk.word_tokenize(tailored_text)  

    smoothie = SmoothingFunction().method1

    # BLEU uses n-grams, so we set weights evenly across 1-gram to 4-gram
    # Here I used 1-gram
    bleu_score = sentence_bleu(original_tokens, tailored_tokens, 
                               weights=(1, 0, 0, 0),  
                               smoothing_function=smoothie)
    
    return {"bleu_score": bleu_score}

# Function to compute ROUGE scores
def compute_rouge_scores(original_text, tailored_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, tailored_text)
    
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_L": scores["rougeL"].fmeasure
    }

# Function to analyze a text
def analyze_text(text):
    return {
        "token_count": count_tokens(text),
        "readability": compute_readability(text),
        "pos": compute_pos_distribution(text)
    }

# Function to analyze similarity between two texts
def analyze_similarity(original_text, tailored_text):
    """Analyzes linguistic metrics and similarity scores between two texts."""
    return {
        "original_text_analysis": analyze_text(original_text),
        "tailored_text_analysis": analyze_text(tailored_text),
        "cosine_similarity": compute_cosine_similarity(original_text, tailored_text),
        "bertscore": compute_bertscore(original_text, tailored_text),
        "bleu_score": compute_bleu_score(original_text, tailored_text),
        "rouge_scores": compute_rouge_scores(original_text, tailored_text)
    }
