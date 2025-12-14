import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from src.vectorstore import CHROMAVectorStore

def load_pdfs_from_directory(directory: str = "./data"):
    pdf_dir = Path(directory)
    
    if not pdf_dir.exists():
        print(f"Error: Directory {directory} does not exist!")
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return []
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    all_documents = []
    
    for pdf_file in pdf_files:
        try:
            print(f"\nLoading: {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Add source file to metadata
            for doc in docs:
                doc.metadata['source_file'] = pdf_file.name
            
            all_documents.extend(docs)
            print(f"  Loaded {len(docs)} pages from {pdf_file.name}")
            
        except Exception as e:
            print(f"  Error loading {pdf_file.name}: {e}")
            continue
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents


def create_embeddings():
    print("=" * 60)
    print("Creating Embeddings for RAG System")
    print("=" * 60)
    
    backend_env = Path(__file__).parent.parent / "backend" / ".env"
    if not backend_env.exists():
        print("\n❌ Error: backend/.env file not found!")
        print("Please create backend/.env with your HUGGINGFACE_API_KEY")
        sys.exit(1)
    
    # Read and verify token
    token_found = False
    with open(backend_env, 'r') as f:
        for line in f:
            if line.strip().startswith('HUGGINGFACE_API_KEY='):
                token_found = True
                break
    
    if not token_found:
        print("\n❌ Error: HUGGINGFACE_API_KEY not found in backend/.env")
        print("Please add: HUGGINGFACE_API_KEY=your_token_here")
        sys.exit(1)
    
    print("\n✅ HuggingFace API key found in backend/.env")
    
    # Load PDF documents
    print("\n" + "=" * 60)
    print("Step 1: Loading PDF Documents")
    print("=" * 60)
    
    documents = load_pdfs_from_directory("./data")
    
    if not documents:
        print("\n❌ No documents loaded. Please add PDF files to ./data directory")
        sys.exit(1)
    
    # Initialize vector store
    print("\n" + "=" * 60)
    print("Step 2: Initializing Vector Store")
    print("=" * 60)
    
    vector_store = CHROMAVectorStore(
        collection_name='legal_texts_pipeline',
        persistent_directory='./src/vectorStore',
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=800,
        chunk_overlap=200
    )
    
    # Create embeddings and store
    print("\n" + "=" * 60)
    print("Step 3: Creating Embeddings (This may take a while...)")
    print("=" * 60)
    print("\nUsing sentence-transformers/all-MiniLM-L6-v2 model")
    print("Chunk size: 800 characters")
    print("Chunk overlap: 200 characters\n")
    
    try:
        vector_store.load_and_add_docs(documents)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Embeddings Created Successfully")
        print("=" * 60)
        print(f"\nTotal documents in collection: {vector_store.collection.count()}")
        print(f"Storage location: ./src/vectorStore")
        print(f"\nYou can now start the RAG API server with:")
        print("  python api.py")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Error Creating Embeddings")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_embeddings()
