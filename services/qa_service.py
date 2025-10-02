import os
from dotenv import load_dotenv
from transformers import pipeline

# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Load model once
question_answer = pipeline(task="question-answering", model="deepset/roberta-base-squad2",token=hf_token)

def generate_qa(context: str,question: str) -> str:
    qa_response = question_answer(question=question, context=context)
    return qa_response["answer"]