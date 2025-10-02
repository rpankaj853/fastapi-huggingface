from pydantic import BaseModel

class CombineTextQARequest(BaseModel):
    query: str
    service_token: str

class CombineTextQAResponse(BaseModel):
    output: str