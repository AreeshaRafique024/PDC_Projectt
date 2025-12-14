import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prefer local sentence-transformers if available to avoid remote HF inference
try:
    from sentence_transformers import SentenceTransformer
    _HAS_LOCAL_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_LOCAL_ST = False

try:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
except Exception:
    HuggingFaceEndpointEmbeddings = None

import numpy as np


def _embed_texts_worker(args):
    """Worker for multiprocessing: loads SentenceTransformer per-process and encodes texts.
    Args: tuple of (model_name, texts_list)
    Returns: list of embedding lists
    """
    model_name, texts = args
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [e.tolist() for e in embs]


class LocalSentenceTransformerEmbeddings:
    """Minimal wrapper providing embed_documents/embed_query using sentence-transformers"""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        if not _HAS_LOCAL_ST:
            raise ImportError('sentence_transformers not installed')
        # for local models, use the model name if it's a repo id or local path
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        # return list of vectors (numpy arrays)
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [np.asarray(vec) for vec in embeddings]

    def embed_query(self, text: str):
        emb = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(emb[0])

class CHROMAVectorStore:
    """LangChain-based ChromaDB Vector Store for RAG Pipeline"""
    
    def __init__(self, collection_name: str = 'legal_texts_pipeline', 
                 persistent_directory: str = './src/vectorStore', 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 200):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_function = None
        self.vectorstore = None
        self.text_splitter = None
        self._initialize_store()

    def _get_hf_token(self) -> str:
        """Read HuggingFace token from backend/.env file"""
        backend_env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'backend', '.env'
        )
        token = None
        
        if os.path.exists(backend_env_path):
            with open(backend_env_path, 'r') as f:
                for line in f:
                    if line.startswith('HUGGINGFACE_API_KEY='):
                        token = line.strip().split('=', 1)[1]
                        break
        
        if not token:
            raise ValueError("HUGGINGFACE_API_KEY not found in backend/.env file")
        
        return token

    def _initialize_store(self):
        """Initialize ChromaDB vector store using LangChain"""
        print("Initializing ChromaDB vector store with LangChain...")
        
        os.makedirs(self.persistent_directory, exist_ok=True)
        
        # Initialize embedding function: prefer local sentence-transformers
        if _HAS_LOCAL_ST:
            try:
                print("Using local sentence-transformers model for embeddings")
                self.embedding_function = LocalSentenceTransformerEmbeddings(
                    model_name=self.embedding_model_name
                )
                print(f"Local embedding model loaded: {self.embedding_model_name}")
            except Exception as e:
                print(f"Failed to load local sentence-transformers: {e}")
                self.embedding_function = None

        if self.embedding_function is None:
            # Fall back to HuggingFace Inference API
            if HuggingFaceEndpointEmbeddings is None:
                raise RuntimeError('No embedding backend available: install sentence-transformers or langchain_huggingface')
            token = self._get_hf_token()
            self.embedding_function = HuggingFaceEndpointEmbeddings(
                repo_id=self.embedding_model_name,
                huggingfacehub_api_token=token,
            )
            print(f"Embedding model loaded via HuggingFace endpoint: {self.embedding_model_name}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )
        
        # Initialize or load ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persistent_directory
        )
        
        print(f"Vector store initialized. Collection: {self.collection_name}")
        print(f"Existing documents in collection: {self.vectorstore._collection.count()}")
    
    def load(self):
        """Load existing vector store"""
        print(f"Vector store loaded. Collection: {self.collection_name}")
        print(f"Existing documents in collection: {self.vectorstore._collection.count()}")

    def load_and_add_docs(self, documents: List[Any]):
        """Split documents and add to vector store using LangChain"""
        print(f"Processing {len(documents)} documents...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Add to vector store (support optional parallel embedding)
        # Default behavior: compute embeddings and add in batches to avoid memory spikes
        batch_size = 64
        from math import ceil
        total = len(chunks)
        batches = ceil(total / batch_size)

        # Helper to add a single batch to Chroma using its low-level collection add
        def _add_batch(batch_chunks, batch_embeddings=None):
            try:
                docs = [c.page_content for c in batch_chunks]
                metadatas = [c.metadata for c in batch_chunks]
                ids = None
                if batch_embeddings is None:
                    # Let LangChain/Chroma compute embeddings
                    self.vectorstore.add_documents(documents=batch_chunks)
                else:
                    # Use low-level collection add to supply precomputed embeddings
                    # Ensure embeddings are lists
                    emb_lists = [e.tolist() if hasattr(e, 'tolist') else e for e in batch_embeddings]
                    try:
                        self.vectorstore._collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=emb_lists)
                    except Exception:
                        # Fallback: call add_documents if low-level add fails
                        self.vectorstore.add_documents(documents=batch_chunks)
            except Exception as e:
                print(f"Error adding batch to collection: {e}")

        # If we have a local SentenceTransformer, we can compute embeddings in parallel using multiprocessing
        if _HAS_LOCAL_ST:
            from multiprocessing import Pool

            print("Computing embeddings in parallel using multiprocessing Pool...")
            for i in range(batches):
                start = i * batch_size
                end = min(start + batch_size, total)
                batch_chunks = chunks[start:end]
                texts = [c.page_content for c in batch_chunks]

                # Use a small Pool for each batch to compute embeddings (keeps memory bounded)
                processes = int(os.getenv('EMBED_POOL_PROCESSES', '2'))
                with Pool(processes=processes) as p:
                    # split texts into sub-batches for workers
                    sub_batch_size = max(1, len(texts) // processes)
                    sub_batches = [texts[j:j+sub_batch_size] for j in range(0, len(texts), sub_batch_size)]
                    args = [(self.embedding_model_name, sb) for sb in sub_batches]
                    results = p.map(_embed_texts_worker, args)
                    # flatten results (each result is a list of embeddings)
                    embeddings = [emb for sub in results for emb in sub]

                _add_batch(batch_chunks, batch_embeddings=embeddings)
                print(f"Added batch {i+1}/{batches} ({end-start} chunks)")
        else:
            # No local model: fall back to sequential batched add (Chroma will call HF endpoint)
            print("No local sentence-transformers available — adding in sequential batches (HF endpoint will be used)")
            for i in range(batches):
                start = i * batch_size
                end = min(start + batch_size, total)
                batch_chunks = chunks[start:end]
                _add_batch(batch_chunks)
                print(f"Added batch {i+1}/{batches} ({end-start} chunks)")

        print(f"Successfully added {len(chunks)} chunks to vector store")
        print(f"Total documents in collection: {self.vectorstore._collection.count()}")
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """Get LangChain retriever interface"""
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    @property
    def collection(self):
        """Backward compatibility: access underlying ChromaDB collection"""
        return self.vectorstore._collection