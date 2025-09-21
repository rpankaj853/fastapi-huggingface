import os
from dotenv import load_dotenv
from fastapi import APIRouter,HTTPException
from schemas.qa_schema import QARequest, QAResponse 
from services.qa_service import generate_qa

router = APIRouter()

# Load .env variables
load_dotenv()

service_token = os.getenv("SERVICE_TOKEN")

@router.post("/", response_model=QAResponse)
def question_answer(request: QARequest):
    # Validate service code

    if request.service_token != service_token:
        raise HTTPException(
            status_code=403,  # Forbidden
            detail="Invalid service code. Access denied."
        )
    result = generate_qa(question=request.question, context=request.context)
    return QAResponse(answer=result)
