````markdown
# Proposal: FastAPI-HuggingFace Integration for AI-Powered APIs

## Overview

This project, **FastAPI-HuggingFace**, is a backend solution designed to integrate the power of Hugging Face's state-of-the-art machine learning models with the high-performance capabilities of FastAPI. The goal is to provide a scalable, modular, and efficient platform for building AI-powered APIs that cater to various use cases such as text generation, summarization, and question answering.

---

## What is LangChain?

LangChain is a framework designed to simplify the development of applications powered by large language models (LLMs). It provides tools to integrate LLMs with external data sources, enabling advanced functionalities such as document retrieval, conversational agents, and more. LangChain supports modularity and extensibility, making it easier to build and manage complex AI workflows.

---

## What is Hugging Face?

Hugging Face is a leading platform in the field of natural language processing (NLP) and machine learning. It offers a vast repository of pre-trained models for tasks like text generation, summarization, translation, and more. Hugging Face also provides tools like the Transformers library, which simplifies the use of these models in various programming environments.

---

## Objectives

1. **Seamless Integration**: Combine the strengths of FastAPI and Hugging Face to create a robust backend for AI applications.
2. **Scalability**: Ensure the platform is scalable to handle large datasets and high traffic.
3. **Modularity**: Design the system to allow easy customization and integration of additional models or features.
4. **Ease of Use**: Provide a user-friendly interface for developers to interact with the APIs.

---

## Features

- **Text Generation**: Generate creative and contextually relevant text using pre-trained Hugging Face models.
- **Text Summarization**: Summarize long passages into concise and meaningful summaries.
- **Question Answering**: Provide accurate answers to questions based on a given context.
- **Custom Model Integration**: Easily integrate custom Hugging Face models for specific use cases.
- **Authentication**: Secure the APIs with service tokens for controlled access, including additional checks for `SERVICE_TOKEN` in the environment file and code.

---

# Retrieval-Augmented Generation (RAG) Integration

This project now includes a complete RAG (Retrieval-Augmented Generation) pipeline built using LangChain, Hugging Face, and FastAPI. The RAG pipeline allows the system to load documents (PDFs, DOCX files), split them into smaller chunks, generate embeddings, store them in a vector database, retrieve the most relevant chunks for a user query, and generate answers grounded in those retrieved documents.

This dramatically improves accuracy and reduces hallucinations by ensuring the LLM (Qwen2.5 / HuggingFace model) responds using ONLY your actual document data.

---

## PHASE 1 — DOCUMENT PROCESSING PIPELINE

1. **Document Loading**:

   - Uses LangChain community loaders, e.g., PyPDFLoader.
   - Loads all PDF files from a specified folder (e.g., `/docs` directory).

2. **Text Splitting**:

   - Uses RecursiveCharacterTextSplitter.
   - Splits long documents into smaller, overlapping chunks.
   - Helps manage token limits and improves retrieval quality.

3. **Embeddings Generation**:

   - Uses a HuggingFace embedding model (such as `all-MiniLM-L6-v2`).
   - Each chunk is converted into a numerical vector representation.
   - Optional normalization to improve vector similarity results.

4. **Vector Database Storage**:
   - Two vector storage options supported:
     - **FAISS**: In-memory, fast local development database.
     - **ChromaDB**: Persistent, on-disk vector database.
   - Chunks + metadata + embeddings are stored in the vector DB.

---

## PHASE 2 — RETRIEVAL & GENERATION PIPELINE

5. **Retriever Service**:

   - Performs top-k semantic search in FAISS or ChromaDB.
   - Returns the most relevant document chunks as context.

   Example output:

   ```json
   [
     { "text": "...", "metadata": { "page": 1, "source": "AIRag.pdf" }, "distance": 0.18 },
     ...
   ]
   ```

6. **Generation Service (LLM Answering with Context)**:

   - Builds a prompt that includes:
     - System instruction: "Answer using ONLY the provided context."
     - Retrieved context blocks with provenance (file + page number).
     - User question.
   - Calls the HuggingFacePipeline using `.invoke(prompt, **kwargs)`.
   - Produces an answer grounded in the retrieved document content.

   Example:

   ```python
   raw_output = llm_generate.invoke(prompt, max_new_tokens=256)
   ```

   The answer is extracted and cleaned (removing repeated assistant markers).

---

## FASTAPI ENDPOINT FOR RAG

A dedicated endpoint has been added:

**POST** `/rag/ask`

Request Body:

```json
{
  "prompt": "<user question>"
}
```

This endpoint:

- Retrieves relevant chunks from the vector DB.
- Builds a RAG prompt.
- Calls the Qwen2.5 or other HF model using `.invoke()`.
- Returns a structured response:
  ```json
  {
    "query": "...",
    "answer": "...",
    "raw_generation": "...",
    "used_contexts": [...]
  }
  ```

This keeps the API extremely simple for client applications.

---

## KEY COMPONENTS ADDED FOR RAG

### Services:

- `loader_service.py`: Loads PDFs/documents.
- `splitter_service.py`: Splits text into chunks.
- `embeddings_service.py`: Generates embeddings with HF models.
- `vectorstore_service_chroma.py`: Stores embeddings in ChromaDB.
- `retriever_service.py`: Retrieves top-k relevant chunks.
- `generation_service.py`: Builds prompt + calls LLM using `.invoke()`.

### API:

- `api_generation.py`: Contains `/rag/ask` endpoint.
- `set_generation_service()`: Injects retriever + LLM at startup.

### LLM Model Loader:

- `load_text_generation_model()`: Returns HuggingFacePipeline wrapper using HF generator.
- Always uses the `.invoke(prompt, **kwargs)` method.

---

## WHY THIS MATTERS

With RAG, your API can now:

- Answer questions based on your private PDFs.
- Reduce hallucinations.
- Provide citations.
- Support enterprise knowledge-base features.
- Scale across hundreds of documents.
- Enable chatbots and document assistants.

---

## NEXT POSSIBLE EXTENSIONS

- Add streaming responses (Server-Sent Events).
- Add chat-style conversation memory.
- Add document upload API for live ingestion.
- Add multiple vector store support (Weaviate, Pinecone, Qdrant).
- Build UI for “Chat With Your PDFs.”

---

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/rpankaj853/fastapi-huggingface.git
   cd fastapi-huggingface
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Fetch Hugging Face Token**:

   - Visit [Hugging Face Settings](https://huggingface.co/settings/tokens).
   - Log in or create an account if you don’t have one.
   - Scroll to the "Access Tokens" section and click "New Token."
   - Provide a name for the token, select the required permissions, and generate it.
   - Copy the generated token for the next step.

5. **Set Up Environment Variables**:

   - Export the Hugging Face token as an environment variable:
     ```bash
     export HF_ACCESS_TOKEN=your_huggingface_token
     ```
   - Define a custom service token for additional authentication:
     ```bash
     export CUSTOM_SERVICE_TOKEN=your_custom_service_token
     ```
   - Ensure the `service_token` is validated in the code for secure API access.

6. **Run the Server**:

   ```bash
   uvicorn main:app --reload
   ```

7. **Access the API Documentation**:
   - Open your browser and navigate to `http://127.0.0.1:8000/api/pr/docs` for Swagger UI.
   - Alternatively, access the Redoc documentation at `http://127.0.0.1:8000/api/pr/redoc`.

---

## Project Structure

The project is organized as follows:

```
fastapi-huggingface/
├── hugging_face/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── text_generation.py
│   │   │   ├── text_summarization.py
│   │   │   └── question_answering.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── model_service.py
│   │   └── token_service.py
│   ├── schema/
│   │   ├── request_schema.py
│   │   └── response_schema.py
│   └── __init__.py
├── langchain_HF/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── combined_generation.py
│   │   │   └── qa_generation.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── langchain_service.py
│   │   └── token_service.py
│   ├── schema/
│   │   ├── request_schema.py
│   │   └── response_schema.py
│   └── __init__.py
├── app/
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── main.py
│   └── __init__.py
├── tests/
│   ├── test_endpoints.py
│   └── __init__.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Example Endpoints

### Hugging Face Endpoints

- **Text Generation**: `/api/v1/hugging_face/generate`
- **Text Summarization**: `/api/v1/hugging_face/summarize`
- **Question Answering**: `/api/v1/hugging_face/qa`

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain Hugging Face Integration](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
- [Uvicorn ASGI Server](https://www.uvicorn.org/)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
````
