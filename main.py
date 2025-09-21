from fastapi import FastAPI
from api import summarize, qa, text_generation
# Initialize FastAPI app

app = FastAPI(title="GEN AI",docs_url="/api/pr/docs",redoc_url="/api/pr/redoc")

# Register route
app.include_router(summarize.router, prefix="/api/v1/summarize", tags=["Summarization"])
app.include_router(qa.router, prefix="/api/v1/qa", tags=["Question Answering"])
app.include_router(text_generation.router, prefix="/api/v1/text-generation", tags=["Text Generation"])
