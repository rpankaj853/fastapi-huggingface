# Import necessary components
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

from ..services.model_config import load_text_generation_model, load_summarization_model

llm = load_text_generation_model()
llm_summarizer = load_summarization_model()


# # Create a prompt template — LangChain replaces {content} dynamically
# summarize_prompt = PromptTemplate(
#     input_variables=["content"],  # expects an input key named "content"
#     template="Summarize this text in one short paragraph:\n\n{content}",
# )

summarize_prompt = PromptTemplate(input_variables=["content"], template="{content}")

# output_key = name given to store the result from this chain
summarize_chain = LLMChain(
    llm=llm_summarizer,
    prompt=summarize_prompt,
    output_key="summary",
)

# This will take the output of the previous step ("summary") as input

title_prompt = PromptTemplate(
    input_variables=["summary"],
    template=(
        "Based on the summary below, generate ONLY a short, catchy title."
        "Do NOT add explanations, do NOT repeat the summary, do NOT add multiple titles.\n\n"
        "Summary:\n{summary}\n\n"
        "Title:"
    ),
)


title_chain = LLMChain(
    llm=llm,
    prompt=title_prompt,
    output_key="title",
)


chain = SequentialChain(
    chains=[summarize_chain, title_chain],
    input_variables=["content"],
    output_variables=["summary", "title"],
)


def squential_chain(text: str):
    """
    Runs the sequential chain to:
    1. Summarize the given text
    2. Generate a title from the summary
    """
    return chain.invoke({"content": text})
