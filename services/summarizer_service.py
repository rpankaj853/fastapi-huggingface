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




# Path: generate_text.py from transformers import pipeline

# Load the model and set device to GPU (device=0)
# model = pipeline(
#     "text-generation",
#     model="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",    device=0,
# )

# # Generate text
# output = model("What is LangChain?")
# print("text generation input",output)
