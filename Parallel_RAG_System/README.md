# 🚀 Parallel RAG System

A production-ready RAG (Retrieval-Augmented Generation) system that combines document retrieval with multiple LLM models for intelligent, context-aware responses.

## 📋 Overview

This system allows you to:

- Upload PDF documents and create embeddings
- Query your documents using natural language
- Get AI-powered responses using various HuggingFace models
- All responses are grounded in your document content (no hallucination)

## 🏗️ System Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Frontend  │ ───> │   Backend   │ ───> │ RAG Pipeline│
│   (React)   │ <─── │  (Node.js)  │ <─── │  (FastAPI)  │
└─────────────┘      └─────────────┘      └─────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  HuggingFace    │
                    │  LLM Models     │
                    └─────────────────┘
```

## ✨ Features

- **25+ HuggingFace Models**: DeepSeek, Llama, Qwen, Mistral, and more
- **RAG Integration**: Retrieves relevant context from your documents
- **Smart Embeddings**: Uses Qwen model for high-quality document embeddings
- **Real-time Chat**: Instant responses with conversation history
- **Document Management**: Easy PDF upload and processing
- **No Hallucination**: Responses strictly based on your documents

## 📦 Project Structure

```
Parallel_RAG/
├── frontend/           # React application
│   ├── src/
│   │   ├── components/ # UI components
│   │   └── services/   # API services
│   └── package.json
│
├── backend/            # Node.js API server
│   ├── services/
│   │   └── llm/       # HuggingFace integration
│   ├── routes/        # API endpoints
│   └── package.json
│
└── rag_pipeline/      # Python RAG system
    ├── api.py         # FastAPI server
    ├── src/           # Core RAG components
    ├── data/          # PDF documents
    └── create_embeddings.py
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **HuggingFace API Key** (free at https://huggingface.co)

### Step 1: Clone and Setup

```bash
cd /home/faizancyber/Pictures/Parallel_RAG
```

### Step 2: Configure API Key

Create `backend/.env` file:

```env
HUGGINGFACE_API_KEY=hf_your_token_here
PORT=5000
```

**Note**: This single API key is used by both backend and RAG pipeline.

### Step 3: Install Backend Dependencies

```bash
cd backend
npm install
```

### Step 4: Install RAG Pipeline Dependencies

```bash
cd ../rag_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 5: Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### Step 6: Add Your Documents

Place your PDF files in `rag_pipeline/data/` directory:

```bash
cp /path/to/your/documents/*.pdf rag_pipeline/data/
```

### Step 7: Create Embeddings

```bash
cd ../rag_pipeline
source venv/bin/activate
python create_embeddings.py
```

This will:

- Load all PDFs from `data/` directory
- Generate embeddings using Qwen model
- Store in ChromaDB vector database

### Step 8: Start All Services

**Terminal 1 - RAG Pipeline:**

```bash
cd rag_pipeline
source venv/bin/activate
python api.py
```

Server runs on: `http://localhost:8000`

**Terminal 2 - Backend:**

```bash
cd backend
npm run dev
```

Server runs on: `http://localhost:5000`

**Terminal 3 - Frontend:**

```bash
cd frontend
npm run dev
```

Frontend runs on: `http://localhost:5173`

### Step 9: Use the System

1. Open `http://localhost:5173` in your browser
2. Select an LLM model from the dropdown
3. Ask questions about your documents
4. Get intelligent, context-aware answers!

## 🎯 How It Works

1. **User asks a question** → Frontend sends to Backend
2. **Backend retrieves context** → Queries RAG Pipeline for relevant documents
3. **RAG searches vector DB** → Finds most similar document chunks
4. **Backend calls LLM** → Sends context + question to HuggingFace model
5. **LLM generates answer** → Based only on provided context
6. **Frontend displays response** → Shows answer to user

## 🔧 Configuration

### Embedding Model

Default: `Qwen/Qwen2.5-Coder-7B-Instruct`

To change, edit `rag_pipeline/src/vectorstore.py`:

```python
embedding_model='Your/Model/Name'
```

### Chunk Settings

Edit `rag_pipeline/src/vectorstore.py`:

```python
chunk_size=800        # Characters per chunk
chunk_overlap=200     # Overlap between chunks
```

### Available Models

All models use HuggingFace provider:

- DeepSeek (V3.2, R1, Distill variants)
- Llama (3.3, 3.2, 3.1, 4.x variants)
- Qwen (2.5, 3 VL variants)
- Mistral (7B Instruct)
- GPT-OSS (20B, 120B)

See `backend/config/models.config.js` for complete list.

## 📊 Database Management

### View Current Data

```bash
cd rag_pipeline
source venv/bin/activate
python
```

```python
from src.vectorstore import CHROMAVectorStore
store = CHROMAVectorStore()
store.load()
print(f"Documents: {store.collection.count()}")
```

### Clear Database

```bash
rm -rf rag_pipeline/src/vectorStore/*
```

### Rebuild Database

```bash
cd rag_pipeline
python create_embeddings.py
```

## 🛠️ Troubleshooting

### "HUGGINGFACE_API_KEY not found"

- Ensure `backend/.env` exists with valid API key
- Format: `HUGGINGFACE_API_KEY=hf_...`

### "No PDF files found"

- Add PDF files to `rag_pipeline/data/` directory
- Supported format: `.pdf` only

### "Connection refused" errors

- Ensure all three services are running
- Check ports: 8000 (RAG), 5000 (Backend), 5173 (Frontend)

### "Cannot answer based on available information"

- Question is outside document scope
- Try rephrasing or add relevant documents

### Embeddings taking too long

- Normal for large documents
- HuggingFace API calls take time
- Consider processing smaller batches

## 📝 API Endpoints

### Backend (Port 5000)

```
GET  /api/chat/health        - Health check
GET  /api/chat/models        - List available models
POST /api/chat               - Send chat message
GET  /api/chat/history       - Get chat sessions
POST /api/chat/history       - Create new session
```

### RAG Pipeline (Port 8000)

```
POST /retrieve               - Retrieve relevant documents
```

## 🔐 Security Notes

- Keep your HuggingFace API key secure
- Don't commit `.env` files to version control
- API keys are stored only in `backend/.env`
- No keys needed in frontend or RAG pipeline

### Protected Files (Automatically Ignored by Git)

The `.gitignore` file ensures these sensitive items are **never** pushed to GitHub:

**Secrets & Environment Variables:**

- `backend/.env`, `frontend/.env`, `rag_pipeline/.env`
- All API keys and credentials

**Databases & Data:**

- `data/chats.json` - Chat history
- `rag_pipeline/src/vectorStore/` - Vector database with embeddings
- `*.sqlite3` - All SQLite databases
- `rag_pipeline/data/*.pdf` - Your PDF documents

**Dependencies & Build Files:**

- `node_modules/` - Node.js packages
- `venv/` - Python virtual environment
- `dist/`, `build/` - Build outputs

**Setup Instructions:**

1. Copy `.env.example` to `.env` in backend and rag_pipeline folders
2. Add your HuggingFace API key to `backend/.env`
3. Never commit actual `.env` files - only `.env.example` templates

## 📄 License

MIT License - Feel free to use and modify

## 🤝 Support

For issues or questions:

1. Check this README
2. Verify all services are running
3. Check console logs for errors
4. Ensure API key is valid

---

**Built with**: React, Node.js, FastAPI, ChromaDB, LangChain, HuggingFace
