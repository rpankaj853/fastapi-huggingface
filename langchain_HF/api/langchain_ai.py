import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from ..schema.model_schema import TextGenRequest, SummarizeRequest, LLMChainRequest
from ..services.text_generation import generate_text
from ..services.summarizer import summarize_text
from ..services.simple_chain import generate_llm_chain_quote
from ..services.sequential_chain import squential_chain

# Load .env variables
load_dotenv()

router = APIRouter()
service_token = os.getenv("SERVICE_TOKEN")


@router.post(
    "/generate",
    summary="Generate Text",
    description="Generates text based on the provided prompt using LLM. Requires valid service token for authentication.",
    response_description="Returns the generated text response",
)
async def generate_text_api(request: TextGenRequest):
    """
    Generate text from a prompt.

    - **prompt**: Input prompt for text generation
    - **service_token**: Authentication token for API access

    Returns AI-generated text based on your prompt.
    """
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid service code. Access denied.",
        )
    result = generate_text(request.prompt)
    return {"response": result}


@router.post(
    "/summarize",
    summary="Summarize Text",
    description="Summarizes the provided text into a concise version. Requires valid service token for authentication.",
    response_description="Returns the summarized text",
)
async def summarize_text_api(request: SummarizeRequest):
    """
    Summarize long text into key points.

    - **text**: Input text to summarize
    - **service_token**: Authentication token for API access

    Returns a concise summary of the input text.
    """
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid service code. Access denied.",
        )
    result = summarize_text(request.text)
    return {"summary": result}


@router.post("/chain")
async def simple_llm_chain(request: LLMChainRequest):
    """
    Generate a quote using LLM chain.

    - **text**: Input text to generate quote from
    - **service_token**: Authentication token for API access

    Returns a generated quote based on the input text.
    """
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid service code. Access denied.",
        )
    result = generate_llm_chain_quote(request.topic)
    return {"quote": result}


@router.post("/sequential-chain")
async def sequential_llm_chain(request: LLMChainRequest):
    """
    Generate a quote using LLM chain.

    - **text**: Input text to generate quote from
    - **service_token**: Authentication token for API access

    Returns a generated quote based on the input text.
    """
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid service code. Access denied.",
        )
    result = squential_chain(request.topic)
    return {"quote": result}
