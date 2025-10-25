import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

# Text Generation
from ..schemas.text_generation_schema import (
    TextGenerationRequest,
    TextGenerationResponse,
)
from ..services.text_generation_service import generate_text

# Summarization
from ..schemas.summarizer import SummaryRequest, SummaryResponse
from ..services.summarizer_service import generate_summary

# Question Answering
from ..schemas.qa_schema import QARequest, QAResponse
from ..services.qa_service import generate_qa

# Combined Text Generation and QA
from ..schemas.combine_text_qa_schema import CombineTextQARequest, CombineTextQAResponse
from ..services.combine_text_qa_service import combine_generate_text

router = APIRouter()

# Load .env variables
load_dotenv()

service_token = os.getenv("SERVICE_TOKEN")


# Text Generation Endpoint
@router.post("/hf_generate", response_model=TextGenerationResponse)
def text_generation(request: TextGenerationRequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied.",
        )
    result = generate_text(query=request.query)
    return TextGenerationResponse(output=result)


# Summarization Endpoint
@router.post("/hf_summarize", response_model=SummaryResponse)
def summarize_text(request: SummaryRequest):
    # Validate service code
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied.",
        )
    result = generate_summary(request.text)
    return SummaryResponse(summary=result)


# Question Answering Endpoint
@router.post("/hf_qa", response_model=QAResponse)
def question_answer(request: QARequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied.",
        )
    result = generate_qa(question=request.question, context=request.context)
    return QAResponse(answer=result)


# Combined Text Generation and QA Endpoint
@router.post("/hf_sequential", response_model=CombineTextQAResponse)
def sequential(request: CombineTextQARequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied.",
        )
    result = combine_generate_text(query=request.query)
    return CombineTextQAResponse(output=result)
