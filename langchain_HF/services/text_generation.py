from ..services.model_config import load_text_generation_model

# Load model once at startup
llm_generate = load_text_generation_model()

def generate_text(user_query: str) -> str:
    """
    Generate creative or factual text response using Qwen Chat Template.
    """

    # Qwen Chat Format (recommended)
    qwen_prompt = (
        f"<|im_start|>user\n"
        f"{user_query}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Run through HuggingFacePipeline
    raw_output = llm_generate.invoke(qwen_prompt)

    # Remove prompt part and keep only assistant answer
    cleaned = raw_output.split("<|im_start|>assistant\n")[-1].strip()

    return cleaned

