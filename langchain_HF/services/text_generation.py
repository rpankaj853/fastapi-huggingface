from ..services.model_config import load_text_generation_model

# Load model once at startup
llm_generate = load_text_generation_model()

def generate_text(prompt: str) -> str:
    """
    Generate creative text response from prompt.
    """
    response = llm_generate.invoke(prompt)
    return response
