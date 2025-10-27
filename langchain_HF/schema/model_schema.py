from pydantic import BaseModel


class TextGenRequest(BaseModel):
    prompt: str
    service_token: str


class SummarizeRequest(BaseModel):
    text: str
    service_token: str


class LLMChainRequest(BaseModel):
    topic: str
    service_token: str
