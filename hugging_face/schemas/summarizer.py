from pydantic import BaseModel

class SummaryRequest(BaseModel):
    text: str   # only text now
    service_token: str

class SummaryResponse(BaseModel):
    summary: str
