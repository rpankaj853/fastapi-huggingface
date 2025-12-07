# rag_pipeline_services/retriever_service.py

from ..rag_pipeline_services.embeddings_service import EmbeddingsService
from ..rag_pipeline_services.vectorstore_service import ChromaVectorStoreService


class RetrieverService:
    """
    Retrieves top-k relevant chunks for a given user query.
    Works with ChromaDB vector store.
    """

    def __init__(
        self,
        embedder: EmbeddingsService,
        vector_store: ChromaVectorStoreService,
        k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str, k: int = None):
        """
        Takes user query → embeds it → retrieves top-k chunks from ChromaDB.
        Returns structured result with text, metadata, and scores.
        """
        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty.")

        # Use custom k OR default
        top_k = k if k is not None else self.k

        # Step 1 — Embed user query
        query_vector = self.embedder.embed_texts([query])[0]

        # Step 2 — Retrieve from ChromaDB
        results = self.vector_store.search(query_vector, k=top_k)

        # Step 3 — Prepare cleaner structure for LLM context
        retrieved_contexts = []
        for text, meta, dist in zip(
            results["documents"], results["metadatas"], results["distances"]
        ):
            retrieved_contexts.append(
                {"text": text, "metadata": meta, "distance": dist}
            )

        return {"query": query, "k": top_k, "results": retrieved_contexts}
