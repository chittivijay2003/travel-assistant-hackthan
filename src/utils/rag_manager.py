"""RAG system for policy documents using local FAISS or Qdrant"""

from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from src.config import Config
from src.utils.logger import setup_logger
import pickle

logger = setup_logger("rag_manager")


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers (no API calls)"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize local embeddings model"""
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name} (384 dimensions)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist()


class RAGManager:
    """Manager for RAG-based policy retrieval using local FAISS"""

    def __init__(self):
        """Initialize RAG system"""
        # Use local embeddings to avoid API quota issues
        self.embeddings = LocalEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("RAG system initialized with local all-MiniLM-L6-v2 embeddings")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        # Use FAISS for local vector storage
        self.vector_store_path = (
            Config.POLICY_DIR.parent / "vector_store" / "faiss_index"
        )
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Load existing FAISS vector store if available"""
        try:
            if self.vector_store_path.exists():
                logger.info(
                    f"Loading existing FAISS index from {self.vector_store_path}"
                )
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully")
            else:
                logger.info(
                    "No existing FAISS index found, will create on first ingestion"
                )
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
            self.vector_store = None

    def _save_vector_store(self):
        """Save FAISS vector store to disk"""
        if self.vector_store:
            try:
                self.vector_store.save_local(str(self.vector_store_path))
                logger.info(f"FAISS index saved to {self.vector_store_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")

    def ingest_pdf(self, pdf_path: Path) -> None:
        """
        Ingest a PDF document into the vector store

        Args:
            pdf_path: Path to PDF file
        """
        try:
            logger.info(f"Ingesting PDF: {pdf_path}")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            # Split documents
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(splits)} chunks")

            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new FAISS index")
                self.vector_store = FAISS.from_documents(
                    documents=splits, embedding=self.embeddings
                )
                logger.info("FAISS index created successfully")
            else:
                logger.info("Adding documents to existing FAISS index")
                self.vector_store.add_documents(splits)
                logger.info("Documents added successfully")

            # Save to disk
            self._save_vector_store()

        except Exception as e:
            logger.error(f"Error ingesting PDF {pdf_path}: {e}")
            raise

    def ingest_all_policies(self) -> None:
        """Ingest all PDF files from policy directory"""
        policy_files = list(Config.POLICY_DIR.glob("*.pdf"))

        if not policy_files:
            logger.warning("No PDF files found in policy directory")
            return

        for pdf_file in policy_files:
            self.ingest_pdf(pdf_file)

        logger.info(f"Ingested {len(policy_files)} policy documents")

    def query_policies(self, query: str, k: int = 3) -> List[Document]:
        """
        Query the policy documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized - no policies ingested yet")
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(
                f"Found {len(results)} relevant policy chunks for query: {query}"
            )
            return results
        except Exception as e:
            logger.error(f"Error querying policies: {e}")
            return []

    def get_policy_context(self, query: str) -> str:
        """
        Get policy context as formatted string

        Args:
            query: Search query

        Returns:
            Formatted policy context
        """
        docs = self.query_policies(query)
        if not docs:
            return "No relevant policy information found."

        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Relevant Policy Information:\n{context}"
