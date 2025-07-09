# Tailoring Language for User Backgrounds: Human and LLM Evaluations of Paraphrased Texts

This repository contains the code and resources for my Master's thesis,  
**"Tailoring Language for User Backgrounds: Human and LLM Evaluations of Paraphrased Texts"**,  
written at the University of Stuttgart, Germany.

---

Francesca Carlon 

MSc. Computational Linguistics

francesca.carlon8@gmail.com

## Description

Modern research and industry teams are increasingly interdisciplinary, so explaining domain‑specific concepts in terms that diverse audiences can understand is becoming extremely important.
This project investigates how Large Language Models (LLMs) can automatically paraphrase technical concepts for readers with different academic backgrounds—specifically in Linguistics, Computer Science, and Computational Linguistics.

Key contributions:

- Paraphrase generation: I design Prompt Engineering pipelines to have LLMs to rewrite specialist texts for non‑experts.

- Linguistic feature analysis: by measuring semantic, syntactic and lexical properties, I identify which features most strongly enhance readers' comprehension.

- Evaluation: I compare human judgments with LLM‑as‑a‑Judge scores to see when synthetic ratings can stand in for costly human annotation.

Findings show that the combined prompting strategy lead to high‑quality paraphrases, and that carefully chosen syntactic and lexical tweaks make unfamiliar concepts more accessible to a diverse audience. LLM‑based evaluations track human preferences reasonably well—though alignment varies by audience—suggesting a path toward scalable quality control for tailored technical writing.

---

## Environment Setup

To create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the code

## Dataset Creation
To create initial dataset with original texts from selected chapters of your book:

```bash
cd code/
python dataset_creation.py
```

## Benchmark Creation
To create tailored paraphrases with different prompts-LLMs combinations and compute linguistic metrics for each text:
```bash
cd code/
python create_benchmark_main.py
```

(Optional) Hallucination Scores can be added with: 
```bash
cd code/
python add_hallucination_scores.py
```

To implement LLM-as-a-judge Survey:
```bash
cd code/
python llm_as_a_judge_survey.py
```

To compute Correlation Analysis (scores and visuals):
```bash
cd code/
python correlations_analysis.py
```

# (Optional) Human annotation setup: 

To randomize a subset from Benchmark:
```bash
cd code/
python randomize_benchmark.py
```

To convert files into HTML for better readability and survey distribution:
```bash
cd code/
python html_converter.py
```