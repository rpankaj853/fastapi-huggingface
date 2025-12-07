import os
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoaderServices:
    """Service class for loading documents from various sources."""

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
