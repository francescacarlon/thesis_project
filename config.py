from pathlib import Path
import os
from dotenv import load_dotenv

# Define dataset paths
BASE_PATH = Path("C:/Users/Francesca Carlon/Desktop/Fran_stuff/MASTER/THESIS/thesis_project/")
DATASET_PATH = BASE_PATH / "data/dataset.json"
BENCHMARK_PATH = BASE_PATH / "data/benchmark.json"
LINGUISTIC_ANALYSIS_PATH = BASE_PATH / "data/linguistic_analysis.json"  # New file only for linguistic analysis


# Load environment variables from .env file
load_dotenv()

# Retrieve API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if key is loaded correctly
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found. Make sure to add it to your .env file.")

print("API Key Loaded Successfully!")
