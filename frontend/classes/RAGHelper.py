from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class NomicEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [
            self.client.embeddings.create(input=[_], model=self.model).data[0].embedding
            for _ in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]


class RAGHelper:
    """Helper for Retrieval Augmented Generation (RAG)."""

    def __init__(self):
        self.message = "Hello World, I am a helper class for RAG."
        embedding_function = NomicEmbeddings(
            model="text-embedding-nomic-embed-text-v1.5-embedding"
        )
        self.vectorstore = Chroma("all_documents", embedding_function)

    def get(self) -> str:
        return self.message

    def get_all_documents(self) -> List[Document]:
        if self.vectorstore:
            return self.vectorstore.get()

    def add_docs_to_chromadb(self, docs) -> None:
        if self.vectorstore:
            self.vectorstore.reset_collection()
        return self.vectorstore.add_documents(docs)

    def retrieve_relevant_docs(self, user_query: str, top_k: int) -> list[Document]:
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
            k=top_k,
        )

        # Retrieve relevant docs
        return results
