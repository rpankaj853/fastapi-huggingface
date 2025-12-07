import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import numpy as np


class EmbeddingsService:
    """
    Embeddings service using sentence-transformers (Hugging Face models).
    - model_name: any sentence-transformers / HF model (e.g. "all-MiniLM-L6-v2")
    - device: "cpu" or "cuda"
    - normalize: whether to L2-normalize embeddings (common for some vector DBs)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = False,
        hf_token_env: str = "HF_TOKEN",
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

        # If you need to access private models, set HF token in environment before creating the model:
        hf_token = os.getenv(hf_token_env)
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # Load the model (this downloads from HF if not cached)
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def _maybe_normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of strings and return list of python lists (vectors).
        """
        # SentenceTransformer handles batching internally; batch_size param passed through encode
        vectors = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        vectors = self._maybe_normalize(vectors)
        return [v.tolist() for v in vectors]

    def embed_documents(
        self, documents: List[Document], batch_size: int = 32
    ) -> List[Tuple[List[float], Document]]:
        """
        Embed a list of LangChain Documents. Returns list of (vector, document).
        """
        texts = [doc.page_content for doc in documents]
        vectors = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        vectors = self._maybe_normalize(vectors)
        return [(vec.tolist(), doc) for vec, doc in zip(vectors, documents)]

    def embed_and_attach(
        self, documents: List[Document], batch_size: int = 32
    ) -> List[Document]:
        """
        (Optional) Attach embedding vector to each Document.metadata['embedding'] and return docs.
        Useful for debugging or small experiments.
        """
        pairs = self.embed_documents(documents, batch_size=batch_size)
        for vec, doc in pairs:
            # copy metadata to avoid mutating unexpected references
            doc.metadata = dict(doc.metadata) if doc.metadata else {}
            doc.metadata["embedding"] = vec
        return documents
