import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Text generation pipeline
generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        token=hf_token,
        return_full_text=False,
        trust_remote_code=True,
        
    )

# Summarization pipeline
summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        token=hf_token
    )


# 🔹 TEXT GENERATION MODEL
def load_text_generation_model():
    
    return HuggingFacePipeline(pipeline=generator)


# 🔹 SUMMARIZATION MODEL
def load_summarization_model():
    
    return HuggingFacePipeline(pipeline=summarizer)
