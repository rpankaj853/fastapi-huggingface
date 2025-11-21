import os
from dotenv import load_dotenv
from transformers import pipeline

# Load .env variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Load model once
text_generation = pipeline(task="text-generation", model="Qwen/Qwen2.5-3B-Instruct",token=hf_token,trust_remote_code=True)

def generate_text(query: str) -> str:
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    text_response = text_generation(
        prompt,
        max_new_tokens=100,
        do_sample=False  # factual answers → deterministic
    )

    output = text_response[0]["generated_text"]

    # Remove the prompt part to keep only the model's answer
    answer = output.split("<|im_start|>assistant\n")[-1].strip()

    return answer
