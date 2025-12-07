# rag_pipeline_services/generation_service.py

from typing import List, Dict, Any, Optional


class GenerationService:
    """
    Simple & clean RAG generation service for your setup.
    Uses:
       retriever.retrieve(query, k)
       generator.invoke(prompt, **kwargs)   <-- matches your usage
    """

    def __init__(
        self,
        retriever,
        generator_callable,  # MUST be your HuggingFacePipeline wrapper
        default_max_new_tokens: int = 256,
        default_temperature: float = 0.7,
    ):
        self.retriever = retriever
        self.generator = generator_callable  # you supply load_text_generation_model()
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature

    # -------------------------
    # Prompt Builder
    # -------------------------
    def _build_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        prompt = []
        prompt.append(
            "You are an assistant that answers the user using ONLY the provided context."
        )
        prompt.append(
            """If the answer is fully contained in the context, answer directly.
If the context does NOT contain the answer, then say ONLY: "I don't know."""
        )
        prompt.append("\nCONTEXT:\n")

        for i, item in enumerate(retrieved, start=1):
            text = (item.get("text") or "").strip()
            meta = item.get("metadata") or {}

            src = meta.get("source", "unknown")
            page = meta.get("page", None)
            prov = f"{src}" + (f", page {page}" if page else "")

            prompt.append(f"[{i}] {text}\n({prov})\n")

        prompt.append("\nQUESTION:\n")
        prompt.append(query.strip())
        prompt.append("\n\nAnswer with citations.\n\nAnswer:\n")

        return "\n".join(prompt)

    # -------------------------
    # Generate Answer (RAG)
    # -------------------------
    def generate_answer(
        self,
        query: str,
        k: int = 5,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Retrieval
        res = self.retriever.retrieve(query, k=k)
        retrieved = res.get("results", [])

        # 2. Build RAG Prompt
        prompt = self._build_prompt(query, retrieved)

        # 3. LLM settings
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.default_max_new_tokens,
            "temperature": temperature or self.default_temperature,
        }

        # 4. Call HuggingFacePipeline.invoke(prompt, **kwargs)
        raw_output = self.generator.invoke(prompt, **gen_kwargs)

        # 5. Extract answer text
        answer = self._extract_answer(raw_output)

        return {
            "query": query,
            "answer": answer,
            "raw_generation": raw_output,
            "used_contexts": retrieved,
        }

    # -------------------------
    # Extract text from HF output
    # -------------------------
    def _extract_answer(self, raw):
        if raw is None:
            return ""

        # HF pipelines usually return list of dicts
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            text = raw[0].get("generated_text") or raw[0].get("text") or str(raw[0])
        elif isinstance(raw, str):
            text = raw
        else:
            text = str(raw)

        # Cleanup markers if any (Qwen sometimes echoes prompt)
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1].strip()

        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()

        return text.strip()
