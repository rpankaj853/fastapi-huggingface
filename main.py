from fastapi import FastAPI
from api import summarize

app = FastAPI(title="Text Summarizer API",docs_url="/api/pr/docs",redoc_url="/api/pr/redoc")

# Register route
app.include_router(summarize.router, prefix="/api/v1/summarize", tags=["Summarization"])
