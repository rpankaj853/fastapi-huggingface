import os
from dotenv import load_dotenv
from ..services.text_generation_service import generate_text
from ..services.qa_service import generate_qa


# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")


def combine_generate_text(query:str) -> str:
    context = generate_text(query)
    qa_response = generate_qa(question=query, context=context)
    return qa_response