from pydantic import BaseModel

class SummaryRequest(BaseModel):
    text: str   # only text now

class SummaryResponse(BaseModel):
    summary: str
