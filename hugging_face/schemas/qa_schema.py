from pydantic import BaseModel

class QARequest(BaseModel):
    context: str
    question: str
    service_token: str


class QAResponse(BaseModel):
    answer: str