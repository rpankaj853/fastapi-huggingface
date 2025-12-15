import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)


class DocumentLoaderServices:
    """Service class for loading documents from various sources."""

    @staticmethod
    def load_any_document(
        file_type: str,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
    ) -> List[Document]:
        """
        Load a document based on file type.

        Args:
            file_type (str): pdf | docx | url
            file_path (str): path to file (required for pdf/docx)
            url (str): URL to load (required for url)

        Returns:
            List[Document]: LangChain Document objects
        """

        file_type = file_type.lower()

        if file_type == "pdf":
            if not file_path:
                raise ValueError("file_path is required for PDF loading.")
            loader = PyPDFLoader(file_path)

        elif file_type == "docx":
            if not file_path:
                raise ValueError("file_path is required for DOCX loading.")
            loader = Docx2txtLoader(file_path)

        elif file_type == "url":
            if not url:
                raise ValueError("url is required for URL loading.")
            loader = WebBaseLoader(url)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return loader.load()

    def load_pdfs_from_folder(self, folder_path: str):
        """Load all PDF documents from the specified folder.

        Args:
            folder_path (str): The path to the folder containing PDF files.

        Returns:
            list: A list of loaded documents.
        """
        all_documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                # Use PyPDFLoader to load the PDF file
                loader = PyPDFLoader(file_path)
                # load documents from the PDF
                docs = loader.load()
                all_documents.extend(docs)
        return all_documents
