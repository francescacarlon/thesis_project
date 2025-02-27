"""
This file defines prompt templates for generating tailored paraphrases.
Each function returns a customized prompt based on the target category.
"""

# ✅ Define category descriptions
CATEGORY_DEFINITIONS = {
    "L": "Linguistics (L) students have a strong background in language structure, phonetics, syntax, and semantics. They have little or no technical knowledge.",
    "CS": "Computer Science (CS) students have strong technical backgrounds, including programming, algorithms, and machine learning. They have little or no linguistic knowledge.",
    "CL": "Computational Linguistics (CL) students bridge linguistics and computer science, NLP, corpus linguistics, AI, and LLMs."
}

def create_prompt1(target_category, text):
    """
    Generates a professor-style explanation tailored to a specific audience.
    """
    category_explanation = CATEGORY_DEFINITIONS.get(target_category, "Unknown category")

    return f"""
    You are an expert professor in {target_category}. Rewrite the following explanation so that it is best understood by {target_category} students.
    The most important goal is that your students understand concepts that are outside their existing background knowledge, so they can pass your exam and improve their academic performance.
    You must consider the information on {category_explanation} to adapt the explanation to the students' backgrounds.

    - Read the text carefully.
    - Identify key concepts.
    - Use terminology specific to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy.
    - Provide relatable examples.
    - Use analogies that help transfer their knowledge to new concepts.
    - Integrate background information if needed.    

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """

def create_prompt2(target_category, text):
    """
    Generates an AI sales expert-style explanation tailored to clients.
    """
    category_explanation = CATEGORY_DEFINITIONS.get(target_category, "Unknown category")

    return f"""
    You are an AI sales expert, responsible for selling AI products to clients with a background in {target_category}.
    To successfully sell your products, your clients must first understand the following concepts.
    Rewrite the explanation below in a way that is most comprehensible to {target_category} clients.
    
    Your goal is to ensure that your clients grasp concepts outside their existing background knowledge so they can make an informed purchase.
    By doing so, you will increase your sales percentage and achieve greater success.

    Consider the information on {category_explanation} to tailor the explanation to your clients' specific backgrounds.

    - Read the text carefully.
    - Identify key concepts.
    - Use terminology specific to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy.
    - Provide relatable examples.
    - Use analogies that help transfer their knowledge to new concepts.
    - Integrate background information if needed.    

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """

def create_prompt3(target_category, text):
    """
    Generates a copywriter-style explanation tailored to an audience.
    """
    category_explanation = CATEGORY_DEFINITIONS.get(target_category, "Unknown category")

    return f"""
    You are a freelance copywriter responsible for adapting field-specific academic texts for an audience with a background in {target_category}. 
    Your client at the publishing house will decide whether to publish your adapted texts in the new edition of the book, making you famous in the copywriting field.

    Rewrite the explanation below in a way that is most comprehensible to {target_category} audience.
    The key objective is to ensure that your audience grasps concepts outside their existing background knowledge so your client will publish your work.
    If you succeed, many more clients will work with you, and you will become extremely successful.

    Consider the information on {category_explanation} to tailor the explanation to your audience’s specific background.

    - Read the text carefully.
    - Identify key concepts.
    - Use terminology that is familiar to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy.
    - Provide relatable examples.
    - Use analogies that help transfer their knowledge to new concepts.
    - Integrate background information if needed.    

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """

# ✅ Define the prompt template and define the categories
def create_prompt4(target_category, text):

    category_definitions = {
        "L": "Linguistics (L) jury have a strong background on language structure, phonetics, syntax, and semantics. They have little or no technical knowledge.",
        "CS": "Computer Science (CS) jury have strong technical backgrounds, including programming, algorithms, and machine learning. They have little or no linguistic knowledge.",
        "CL": "Computational Linguistics (CL) jury bridge linguistics and computer science, NLP, corpus linguistics, AI, and LLMs."
    }
    
    category_explanation = category_definitions.get(target_category, "Unknown category") # unknown if the category does not appear in category_definitions

    return f"""
    We are playing an escape room game against each other. Your goal is to gain more points than me to be the first one escaping hence winning the game. 
    In order to escape the room, you must paraphrase field-specific concepts to the jury of the game in the {target_category}. 
    They will assign you points from one to ten, according to how good their understanding of the new explanation is. 
    Therefore, you want your adaptation to be the best possible, and definitely better than mine, so that you can gain ten points for each paraphrasis and win more quickly. 
    Consider the information on {category_explanation} to tailor the explanation to each jury specific background. 
    The winning steps for good paraphrasis are:
    - Read the text carefully.
    - Identify the key concepts.
    - Use terminology that is familiar to the jury's field.
    - Avoid unnecessary sentence complexity while maintaining accuracy. 
    - Provide examples they are familiar with.
    - Provide analogies they can relate their knowledge with and transfer it to new concepts.
    - Integrate background information if needed.    
    - For the CL jury in {category_explanation}, only provide one paraphrasis for NLP/Computational Linguistics Practitioners.

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """

# ✅ Define the prompt template and define the categories
def create_prompt5(target_category, text): # this with no role

    category_definitions = {
        "L": "Linguistics (L) audience have a strong background on language structure, phonetics, syntax, and semantics. They have little or no technical knowledge.",
        "CS": "Computer Science (CS) audience have strong technical backgrounds, including programming, algorithms, and machine learning. They have little or no linguistic knowledge.",
        "CL": "Computational Linguistics (CL) audience bridge linguistics and computer science, NLP, corpus linguistics, AI, and LLMs."
    }
    
    category_explanation = category_definitions.get(target_category, "Unknown category") # unknown if the category does not appear in category_definitions

    return f"""
    Paraphrase the given concepts in order for the audience of the {target_category} to understand concepts outside of their field of knowledge. 
    You must consider the information on the {category_explanation} to adapt the tailored texts to the audience backgrounds. 
    - Read the text carefully.
    - Identify the key concepts.
    - Use terminology specific to their field.
    - Avoid unnecessary sentence complexity while maintaining accuracy. 
    - Provide examples they are familiar with.
    - Provide analogies they can relate their knowledge with and transfer it to new concepts.
    - Integrate background information if needed.
    - Do not consider other background per category. 
    - Output only one text per category.

    Original text: 
    {text}

    ### END OF INPUT ###

    Tailored text (provide one text per target category only):
    """