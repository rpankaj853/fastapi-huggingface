import os
from dotenv import load_dotenv
from fastapi import APIRouter,HTTPException
from schemas.text_generation_schema import TextGenerationRequest, TextGenerationResponse 
from services.text_generation_service import generate_text

router = APIRouter()

# Load .env variables
load_dotenv()

service_token = os.getenv("SERVICE_TOKEN")

@router.post("/", response_model=TextGenerationResponse)
def text_generation(request: TextGenerationRequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = generate_text(query=request.query)
    return TextGenerationResponse(output=result)
