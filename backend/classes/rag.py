from typing import Union

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RAGHelper:
    """Helper for Retrieval Augmented Generation (RAG)."""

    def __init__(self):
        self.message = "Hello World, I am a helper class for RAG."
        self.vectorstore_path = r"OmniPDF\vector-database\vectorstore"
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            "all_documents", embedding_function, persist_directory=self.vectorstore_path
        )

    def get(self) -> str:
        return self.message

    def add_docs_to_chromadb(self, docs) -> None:
        return self.vectorstore.add_documents(docs)

    def retrieve_relevant_docs(self, user_query: str) -> list[Document]:
        """Retrieve relevant documents from vector database based on user
        query.

        Parameters
        ----------
        user_query : str
            The user query or prompt in "Chat with Omni".

        Returns
        -------
        pd.DataFrame
            The DataFrame that contains the documents with relevance score.
        """

        # Read vector database as DataFrame
        results = self.vectorstore.similarity_search(
            user_query,
            k=5,
        )

        # Retrieve relevant docs
        return results
