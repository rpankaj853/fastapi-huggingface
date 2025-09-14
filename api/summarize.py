from fastapi import APIRouter
from schemas.summarizer import SummaryRequest, SummaryResponse
from services.summarizer_service import generate_summary

router = APIRouter()

@router.post("/", response_model=SummaryResponse)
def summarize_text(request: SummaryRequest):
    result = generate_summary(request.text)
    return SummaryResponse(summary=result)
