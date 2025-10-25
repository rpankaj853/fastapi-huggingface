from ..services.model_config import load_summarization_model

llm_summarizer = load_summarization_model()

def summarize_text(content: str) -> str:
    """
    Summarize a long passage.
    """
    response = llm_summarizer.invoke(content)
    return response
