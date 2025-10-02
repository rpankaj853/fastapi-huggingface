import os
from dotenv import load_dotenv
from transformers import pipeline

# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Load model once
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn",token=hf_token)

def generate_summary(text: str) -> str:
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

