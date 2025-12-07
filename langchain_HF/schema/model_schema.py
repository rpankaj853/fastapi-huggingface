from pydantic import BaseModel
from typing import Optional


class TextGenRequest(BaseModel):
    prompt: str
    service_token: str


class SummarizeRequest(BaseModel):
    text: str
    service_token: str


class LLMChainRequest(BaseModel):
    topic: str
    service_token: str


class AddOptions(BaseModel):
    pass


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5


class AskRequest(BaseModel):
    prompt: str
