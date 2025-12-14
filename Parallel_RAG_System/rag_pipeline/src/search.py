from typing import List, Dict, Any
from src.vectorstore import CHROMAVectorStore

class RAGRetriever:
    """LangChain-based retriever for RAG Pipeline"""
    
    def __init__(self, vector_store: CHROMAVectorStore):
        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using LangChain retriever"""
        print(f"Retrieving documents for query: '{query}'")
        
        try:
            # Update retriever with new top_k value
            self.retriever = self.vector_store.get_retriever(k=top_k)
            
            # Use LangChain's invoke method to retrieve documents
            documents = self.retriever.invoke(query)
            
            retrieved_docs = []
            if documents:
                for i, doc in enumerate(documents):
                    retrieved_docs.append({
                        'id': doc.metadata.get('id', f'doc_{i}'),
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'rank': i + 1
                    })
                print(f"Retrieved {len(retrieved_docs)} documents")
            else:
                print("No documents found")

            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []