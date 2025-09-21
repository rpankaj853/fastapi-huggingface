import os
from dotenv import load_dotenv
from transformers import pipeline

# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Load model once
text_generation = pipeline(task="text-generation", model="openai-community/gpt2",token=hf_token)

def generate_text(query:str) -> str:
    text_response = text_generation(query,max_new_tokens=50)
    text_updated_response = text_response[0]["generated_text"]
    return " ".join(line.strip() for line in text_updated_response.split("\n"))