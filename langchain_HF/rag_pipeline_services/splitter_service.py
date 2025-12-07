from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitterService:
    """
    Splits loaded documents into smaller chunks for embeddings.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents):
        """
        Accepts list of LangChain Documents and splits them into chunks.
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks
