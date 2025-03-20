import requests
import os
from dotenv import load_dotenv
import json
from config import LINGUISTIC_ANALYSIS_PATH

HF_API_URL = "https://huggingface.co/api/models"
search_query = "mistral"  # Searches for Mistral models

params = {
    "search": search_query,  # Search for Mistral models
    "limit": 20,  # Number of models to retrieve
}

response = requests.get(HF_API_URL, params=params)

if response.status_code == 200:
    models = response.json()
    if models:
        print("\n✅ Available Mistral Models on Hugging Face:")
        for model in models:
            print("-", model["modelId"])
    else:
        print("\n❌ No Mistral models found.")
else:
    print(f"\n❌ Failed to retrieve models: {response.status_code} - {response.text}")


# Load Hugging Face API Key from .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Check if API Key is loaded
if not HF_API_KEY:
    print("❌ Error: Hugging Face API Key Not Found. Check your .env file.")
    exit()

# Define the API request
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": "Explain deep learning in simple terms.",
    "parameters": {"max_new_tokens": 100, "temperature": 0.7},
}

# Call Hugging Face API for Mistral
url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3" # WORKS

try:
    response = requests.post(url, headers=headers, json=payload)
    
    print("Status Code:", response.status_code)
    #print("Response:", response.json())  # Display API response
except Exception as e:
    print("❌ Error:", e)

########################################à

from bert_score import score

# Run a quick test with a known example
original_text = "The cat is on the mat."
paraphrased_text = "A feline sits on a rug."

P, R, F1 = score(
    [paraphrased_text], 
    [original_text], 
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",
    rescale_with_baseline=False
)

# print("\n🔍 **Quick Test Example:**")
# print(f"Original: {original_text}")
# print(f"Paraphrase: {paraphrased_text}")
# print(f"BERTScore - Precision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}\n")

# Define test cases with varying similarity levels
test_pairs = [
    ("Parts of speech fall into two broad categories: closed class and open class. Closed classes are those with relatively fixed membership, such as prepositions [...]. By contrast, nouns and verbs are open classes [...]. Closed class words are generally function words like of, it, and, or you, which tend to be very short, occur frequently, and often have structuring uses in grammar. Four major open classes occur in the languages of the world: nouns [...], verbs, adjectives, and adverbs, [...]. Nouns are words for people, places, or things,  [...]. Verbs refer to actions and processes, including main verbs like draw, provide, and go. English verbs have inflections (non-third-person-singular (eat), third-person singular (eats), progressive (eating), past participle (eaten)). [...]. Adjectives often describe properties or qualities of nouns, like color (white, black), age (old, young), and value (good, bad), [...]. Adverbs generally modify something. [...] A particle [...] is used in combination with a verb. Particles often have extended meanings that aren't quite the same as the prepositions they resemble, as in the particle 'over' in 'she turned the paper over'. A phrasal verb and a particle acting as a single unit is called a phrasal verb. The meaning of phrasal verbs is often non-compositional - not predictable from the individual meanings of the verb and the particle", "In the realm of linguistics, words can be categorized into two main types: closed and open classes. Closed classes, such as prepositions (like 'in' or 'on'), have a limited set of members. On the other hand, open classes, including nouns (e.g., 'person' or 'car') and verbs (e.g., 'draw' or 'go'), have a more dynamic and flexible membership.\n\n     Closed class words are typically function words, such as 'the,' 'and,' or 'or.' They are short, common, and often play a crucial role in structuring sentences. For instance, English function words like 'the' and 'and' help to denote the subject and connect sentences, respectively.\n\n     English has four major open classes: nouns, verbs, adjectives, and adverbs.\n\n     Nouns represent entities like people, places, or things. For example, 'student,' 'classroom,' or 'book.'\n\n     Verbs signify actions or processes. Some examples include 'draw,' 'provide,' or 'go.' English verbs can have inflections, such as the third-person singular form (eats) and the progressive form (eating).\n\n     Adjectives usually describe the properties or qualities of nouns. For instance, 'red' or 'old' can describe a 'car.'\n\n     Adverbs generally modify verbs, adjectives, or other adverbs. An example of an adverb is 'quickly.'\n\n     A particle is a type of word that combines with a verb, like 'over' in 'she turned the paper over.' Particles often have meanings that differ from their prepositional counterparts, as in the case of the particle 'over' in the example above.\n\n     When a particle becomes associated with a verb, forming a single unit, it is called a phrasal verb. Phrasal verbs often have meanings that are not straightforward combinations of the individual verb and particle meanings. For example, the phrasal verb 'turn over' does not mean simply 'to turn' and 'over.' Instead, it implies a complete action or process, such as revealing or changing the position of something."),  # High similarity
    ("A feedforward network is a multilayer network in which the units are connected with no cycles; the outputs from units in each layer are passed to units in the next higher layer, and no outputs are passed back to lower layers. [...] Simple feedforward networks have three kinds of nodes: input units, hidden units, and output units. [...] The input layer x is a vector of simple scalar values. [...] The core of the neural network is the hidden layer h formed of hidden units hi, each of which is a neural unit [...]. In the standard architecture, each layer is fully-connected, meaning that each unit in each layer takes as input the outputs from all the units in the previous layer, and there is a link between every pair of units from two adjacent layers. [...] a single hidden unit has as parameters a weight vector and a bias. We represent the parameters for the entire hidden layer by combining the weight vector and bias for each unit i into a single weight matrix W and a single bias vector b for the whole layer. [...] the hidden layer computation for a feedforward network can be done very efficiently with simple matrix operations. In fact, the computation only has three steps: multiplying the weight matrix by the input vector x, adding the bias vector b, and applying the activation function g (such as the sigmoid, tanh, or ReLU activation […]). The output of the hidden layer, the vector h, is thus the following (for this example we'll use the sigmoid function σ as our activation function): h = σ(Wx+b). [...] The role of the output layer is to take this new representation h and compute a final output. This output could be a real valued number, but in many cases the goal of the network is to make some sort of classification decision,", "In the field of Linguistics, you're likely familiar with the concept of hierarchical structures, where elements are organized in a specific order to convey meaning. A feedforward network can be thought of as a similar hierarchical structure, where units are connected in a specific way to process and transform information. \n\nImagine a simple sentence with a subject, verb, and object. In a feedforward network, the input layer would be like the subject, providing the initial information. The hidden layer would be like the verb, taking the input and transforming it into a new representation, much like how a verb changes the subject in a sentence. The output layer would be like the object, receiving the transformed information and producing a final result.\n\nIn a feedforward network, the units are connected in a specific way, with no cycles or feedback loops. The output from each unit in one layer is passed to the units in the next layer, but not back to the previous layer. This is similar to how words in a sentence are arranged in a specific order to convey meaning, with"),  # Medium similarity
    ("The weather is nice today.", "I love ice cream."),  # Low similarity
]

# print("\n📊 **Testing BERTScore with Multiple Pairs:**")
# for orig, tail in test_pairs:
#     P, R, F1 = score([tail], [orig], model_type="microsoft/deberta-xlarge-mnli", lang="en")
#     
#     print(f"📌 **Original:** {orig}")
#     print(f"🔄 **Paraphrase:** {tail}")
#     print(f"✅ Precision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}\n")


##################################

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')

# Function to compute BLEU score
def compute_bleu_score(original_text, tailored_text):
    """Computes BLEU score for similarity checking."""
    if not original_text.strip() or not tailored_text.strip():
        return {"bleu_score": None}  

    original_tokens = [nltk.word_tokenize(original_text)]  
    tailored_tokens = nltk.word_tokenize(tailored_text)  

    smoothie = SmoothingFunction().method1

    # BLEU uses n-grams, so we set weights evenly across 1-gram to 4-gram
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

# Function to analyze similarity using BLEU and ROUGE
def analyze_similarity(original_text, tailored_text):
    return {
        "bleu_score": compute_bleu_score(original_text, tailored_text),
        "rouge_scores": compute_rouge_scores(original_text, tailored_text)
    }

# Example original text
original_text = "Parts of speech fall into two broad categories: closed class and open class. Closed classes are those with relatively fixed membership, such as prepositions [...]. By contrast, nouns and verbs are open classes [...]. Closed class words are generally function words like of, it, and, or you, which tend to be very short, occur frequently, and often have structuring uses in grammar. Four major open classes occur in the languages of the world: nouns [...], verbs, adjectives, and adverbs, [...]. Nouns are words for people, places, or things,  [...]. Verbs refer to actions and processes, including main verbs like draw, provide, and go. English verbs have inflections (non-third-person-singular (eat), third-person singular (eats), progressive (eating), past participle (eaten)). [...]. Adjectives often describe properties or qualities of nouns, like color (white, black), age (old, young), and value (good, bad), [...]. Adverbs generally modify something. [...] A particle [...] is used in combination with a verb. Particles often have extended meanings that aren't quite the same as the prepositions they resemble, as in the particle 'over' in 'she turned the paper over'. A phrasal verb and a particle acting as a single unit is called a phrasal verb. The meaning of phrasal verbs is often non-compositional - not predictable from the individual meanings of the verb and the particle."

# Example tailored paraphrase
tailored_text = "In the realm of linguistics, words can be categorized into two main types: closed and open classes. Closed classes, such as prepositions (like 'in' or 'on'), have a limited set of members. On the other hand, open classes, including nouns (e.g., 'person' or 'car') and verbs (e.g., 'draw' or 'go'), have a more dynamic and flexible membership.\n\n     Closed class words are typically function words, such as 'the,' 'and,' or 'or.' They are short, common, and often play a crucial role in structuring sentences. For instance, English function words like 'the' and 'and' help to denote the subject and connect sentences, respectively.\n\n     English has four major open classes: nouns, verbs, adjectives, and adverbs.\n\n     Nouns represent entities like people, places, or things. For example, 'student,' 'classroom,' or 'book.'\n\n     Verbs signify actions or processes. Some examples include 'draw,' 'provide,' or 'go.' English verbs can have inflections, such as the third-person singular form (eats) and the progressive form (eating).\n\n     Adjectives usually describe the properties or qualities of nouns. For instance, 'red' or 'old' can describe a 'car.'\n\n     Adverbs generally modify verbs, adjectives, or other adverbs. An example of an adverb is 'quickly.'\n\n     A particle is a type of word that combines with a verb, like 'over' in 'she turned the paper over.' Particles often have meanings that differ from their prepositional counterparts, as in the case of the particle 'over' in the example above.\n\n     When a particle becomes associated with a verb, forming a single unit, it is called a phrasal verb. Phrasal verbs often have meanings that are not straightforward combinations of the individual verb and particle meanings. For example, the phrasal verb 'turn over' does not mean simply 'to turn' and 'over.' Instead, it implies a complete action or process, such as revealing or changing the position of something."

# Compute BLEU and ROUGE scores
results = analyze_similarity(original_text, tailored_text)

# Print results
print("BLEU & ROUGE Results:")
print(results)




