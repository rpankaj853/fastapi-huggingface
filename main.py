from fastapi import FastAPI
from hugging_face.api import hugging_face_ai
from langchain_HF.api import langchain_ai

# Initialize FastAPI app

app = FastAPI(title="GEN AI", docs_url="/api/pr/docs", redoc_url="/api/pr/redoc")

# Register route
app.include_router(hugging_face_ai.router, prefix="/api/v1/hugging-ai", tags=["Hugging Face"])

app.include_router(
    langchain_ai.router, prefix="/api/v1/langchain-ai", tags=["LangChain AI"]
)
