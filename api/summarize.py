import os
from dotenv import load_dotenv
from fastapi import APIRouter,HTTPException
from schemas.summarizer import SummaryRequest, SummaryResponse
from services.summarizer_service import generate_summary

router = APIRouter()

# Load .env variables
load_dotenv()

service_token = os.getenv("SERVICE_TOKEN")

@router.post("/", response_model=SummaryResponse)
def summarize_text(request: SummaryRequest):
    # Validate service code
    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = generate_summary(request.text)
    return SummaryResponse(summary=result)
