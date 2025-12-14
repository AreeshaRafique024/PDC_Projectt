from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.search import RAGRetriever
from src.vectorstore import CHROMAVectorStore
from src.re_ranking import parallel_rerank, load_config
import uvicorn
import os

app = FastAPI(title="RAG Pipeline API")

class ChatRequest(BaseModel):
    query: str
    model_id: str = None  # Optional, not used anymore since LLM is in backend
    rerank: bool = False
    rerank_k: int = None

# Global store instance to avoid reloading on every request
store = None

@app.on_event("startup")
async def startup_event():
    global store
    print("Loading Vector Store...")
    store = CHROMAVectorStore()
    store.load()
    print("Vector Store Loaded.")

@app.post("/retrieve")
async def retrieve(request: ChatRequest):
    global store
    if not store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        retriever = RAGRetriever(vector_store=store)

        print(f"Retrieving documents for query: {request.query}")
        # retrieve top candidates (use larger pool when reranking)
        top_k = 5
        if request.rerank:
            cfg = load_config()
            top_k = cfg.get('top_n', 50)

        results = retriever.retrieve(query=request.query, top_k=top_k)

        if request.rerank:
            print("Running parallel re-ranking...")
            reranked = parallel_rerank(request.query, results)
            return {"results": reranked}

        return {"results": results}
    except Exception as e:
        print(f"Error processing retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
