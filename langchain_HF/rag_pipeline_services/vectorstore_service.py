# rag_pipeline_services/vectorstore_service_chroma.py

import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient


class ChromaVectorStoreService:
    """
    Simple ChromaDB vector store wrapper.
    Stores embeddings, metadata, and performs similarity search.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "rag_collection",
    ):
        """
        persist_directory : folder to save the Chroma DB
        collection_name   : name of the vector collection
        """

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create/load persistent DB
        self.client = PersistentClient(path=self.persist_directory)

        # create or load collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    # -------------------------------------------------------
    # ADD EMBEDDINGS
    # -------------------------------------------------------
    def add_embeddings(self, embedded_pairs):
        """
        embedded_pairs: List[(vector, Document)]
        """

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for idx, (vector, doc) in enumerate(embedded_pairs):
            ids.append(f"id_{len(self.collection.get()['ids']) + idx}")
            embeddings.append(vector)
            metadatas.append(doc.metadata)
            documents.append(doc.page_content)

        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

        print(f"Added {len(embeddings)} items to ChromaDB.")

    # -------------------------------------------------------
    # SEARCH / RETRIEVE
    # -------------------------------------------------------
    def search(self, query_vector, k=5):
        """
        query_vector: list[float]
        returns top-k results: text, metadata, score
        """

        results = self.collection.query(query_embeddings=[query_vector], n_results=k)

        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

    # -------------------------------------------------------
    # DELETE COLLECTION (optional)
    # -------------------------------------------------------
    def delete_all(self):
        self.client.delete_collection(self.collection_name)
        print("🗑️ Collection deleted.")
