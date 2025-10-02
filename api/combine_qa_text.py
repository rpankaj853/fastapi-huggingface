import os
from dotenv import load_dotenv
from fastapi import APIRouter,HTTPException
from schemas.combine_text_qa_schema import CombineTextQARequest, CombineTextQAResponse
from services.combine_text_qa_service import combine_generate_text

router = APIRouter()

# Load .env variables
load_dotenv()

service_token = os.getenv("SERVICE_TOKEN")

@router.post("/", response_model=CombineTextQAResponse)
def text_generation(request: CombineTextQARequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = combine_generate_text(query=request.query)
    return CombineTextQAResponse(output=result)
