from pydantic import BaseModel


class TextGenerationRequest(BaseModel): 
    query: str
    service_token: str


class TextGenerationResponse(BaseModel):
    output: str