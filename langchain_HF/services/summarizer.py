from ..services.model_config import load_summarization_model
import json

llm_summarizer = load_summarization_model()


def clean_title(title: str) -> str:
    # Split by new lines and remove empty entries
    lines = [line.strip() for line in title.split("\n") if line.strip()]
    return lines[0]


def summarize_text(content: str) -> str:
    """
    Summarize a long passage.
    """
    content = clean_title(content)
    response = llm_summarizer.invoke(content)
    return response
