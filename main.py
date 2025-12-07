from fastapi import FastAPI
from hugging_face.api import hugging_face_ai
from langchain_HF.api.langchain_sample_apis import langchain_ai
from langchain_HF.api.rag_pipeline_apis import rag_apis

# Initialize FastAPI app

app = FastAPI(title="GEN AI", docs_url="/api/pr/docs", redoc_url="/api/pr/redoc")

# Register route
app.include_router(
    hugging_face_ai.router, prefix="/api/v1/hugging-ai", tags=["Hugging Face"]
)

app.include_router(
    langchain_ai.router, prefix="/api/v1/langchain-ai", tags=["LangChain AI"]
)

app.include_router(
    rag_apis.router, prefix="/api/v1/langchain-rag-ai", tags=["RAG Pipeline"]
)
