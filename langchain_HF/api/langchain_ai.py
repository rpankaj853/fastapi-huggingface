import os
from dotenv import load_dotenv
from fastapi import APIRouter,HTTPException
from fastapi import APIRouter
from ..schema.model_schema import TextGenRequest, SummarizeRequest
from ..services.text_generation import generate_text
from ..services.summarizer import summarize_text

# Load .env variables
load_dotenv()

router = APIRouter()
service_token = os.getenv("SERVICE_TOKEN")

@router.post("/generate")
async def generate_text_api(request: TextGenRequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = generate_text(request.prompt)
    return {"response": result}

@router.post("/summarize")
async def summarize_text_api(request: SummarizeRequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = summarize_text(request.text)
    return {"summary": result}
