import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from ...schema.model_schema import AddOptions, QueryRequest, AskRequest
from ...services.model_config import load_text_generation_model
from ...rag_pipeline_services.loader_service import DocumentLoaderServices
from ...rag_pipeline_services.splitter_service import DocumentSplitterService
from ...rag_pipeline_services.embeddings_service import EmbeddingsService
from ...rag_pipeline_services.vectorstore_service import ChromaVectorStoreService
from ...rag_pipeline_services.retriever_service import RetrieverService
from ...rag_pipeline_services.generation_query_service import GenerationService

# Load .env variables
load_dotenv()

# doc path
docs_path = "Add your docs path here"

router = APIRouter()
service_token = os.getenv("SERVICE_TOKEN")

# Initialize document loader service
loader_service = DocumentLoaderServices()

# Initialize document splitter service
splitter_service = DocumentSplitterService()

# Initialize embeddings service
embedder = EmbeddingsService(
    model_name="all-MiniLM-L6-v2", device="cpu", normalize=True
)

# Initialize vector store service
chroma_store = ChromaVectorStoreService()

# Initialize retriever service
retriever = RetrieverService(embedder, chroma_store, k=5)


# Initialize generation service
llm = load_text_generation_model()
gen_service = GenerationService(retriever, llm)


@router.get("/rag-document-loader")
def load_documents():
    """Endpoint to load documents for RAG pipeline."""
    docs = loader_service.load_pdfs_from_folder(docs_path)

    if not docs:
        return {"message": "No PDFs found in docs folder"}

    return {
        "total_pages_loaded": len(docs),
        "first_page_preview": docs[0].page_content[:300],
        "metadata": docs[0].metadata,
    }


@router.get("/split-pdfs")
def split_pdfs():
    docs = loader_service.load_pdfs_from_folder(docs_path)
    chunks = splitter_service.split_documents(docs)

    return {
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0].page_content,
        "metadata": chunks[0].metadata,
    }


@router.get("/embed-test")
def embed_test():
    docs = loader_service.load_pdfs_from_folder(docs_path)
    chunks = splitter_service.split_documents(docs)
    if not chunks:
        return {"message": "no chunks"}

    pairs = embedder.embed_documents(chunks[:5])
    return {
        "embedded": len(pairs),
        "dim": len(pairs[0][0]),
        "sample_preview": pairs[0][1].page_content[:200],
        "sample_metadata": pairs[0][1].metadata,
    }


@router.get("/embed-chunks")
def embed_process():
    # 1. Load PDFs
    docs = loader_service.load_pdfs_from_folder(docs_path)

    # 2. Split into chunks
    chunks = splitter_service.split_documents(docs)

    if not chunks:
        return {"message": "No chunks created"}

    # ⭐ Embed ALL chunks (change this to chunks[:5] if you want only 5)
    embedded_pairs = embedder.embed_documents(chunks)

    # ⭐ Build full output
    results = []
    for vector, doc in embedded_pairs:
        results.append(
            {
                "embedding_vector": vector,  # full embedding list
                "vector_dimension": len(vector),  # dimension of embedding
                "text": doc.page_content,  # full chunk text
                "metadata": doc.metadata,  # page, source, etc.
            }
        )

    return {
        "total_chunks": len(chunks),
        "total_embedded": len(results),
        "data": results,
    }


@router.post("/add", summary="Load PDFs, split, embed and add to ChromaDB")
def add_all_to_chroma(opts: AddOptions):
    """
    Loads PDFs from `source_folder`, splits them, embeds chunks and adds to ChromaDB.
    Returns counts.
    """
    # 1) load
    docs = loader_service.load_pdfs_from_folder(docs_path)
    if not docs:
        raise HTTPException(status_code=400, detail=f"No PDFs found in `{docs_path}`")

    # 2) split
    chunks = splitter_service.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=500, detail="Splitter produced no chunks")

    # 3) embed
    embedded_pairs = embedder.embed_documents(chunks)

    # 4) add to chroma
    chroma_store.add_embeddings(embedded_pairs)

    return {
        "status": "ok",
        "source_folder": docs_path,
        "pages_loaded": len(docs),
        "chunks_created": len(chunks),
        "items_added": len(embedded_pairs),
    }


@router.post("/query", summary="Query ChromaDB with text")
def query_chroma(req: QueryRequest):
    """
    Embed the query and search ChromaDB, returning documents, metadatas and distances.
    """
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query text required")

    # 1) embed query text
    q_vec_list = embedder.embed_texts([req.query])
    if not q_vec_list:
        raise HTTPException(status_code=500, detail="Failed to embed query")

    q_vec = q_vec_list[0]

    # 2) search chroma
    results = chroma_store.search(q_vec, k=req.k)

    return {"query": req.query, "k": req.k, "results": results}


@router.post("/persist", summary="Persist ChromaDB to disk")
def persist_chroma():
    chroma_store.persist()
    return {"status": "persisted"}


@router.delete("/delete_all", summary="Delete entire Chroma collection")
def delete_collection():
    chroma_store.delete_all()
    return {"status": "deleted"}


@router.get("/status", summary="Chroma status (counts)")
def chroma_status():
    """
    Simple status: returns number of items currently in the collection (best-effort).
    """
    try:
        info = chroma_store.collection.count()
        # chroma returns dict with 'count' depending on version; fallback
        count = info if isinstance(info, int) else info.get("count", None)
    except Exception:
        # fallback: try retrieving collection metadata length
        try:
            count = len(chroma_store.collection.get()["ids"])
        except Exception:
            count = None

    return {
        "collection_name": chroma_store.collection_name,
        "persist_directory": chroma_store.persist_directory,
        "count": count,
    }


@router.post("/retrieve")
def retrieve_chunks(req: QueryRequest):
    result = retriever.retrieve(req.query, req.k)
    return result


@router.post("/ask")
def ask(req: AskRequest):
    return gen_service.generate_answer(req.prompt)
