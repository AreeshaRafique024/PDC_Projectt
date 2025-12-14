#!/bin/bash
# Quick Start Script for Parallel RAG System
# Run this script to check system status and start services

echo "============================================"
echo "   Parallel RAG System - Quick Start"
echo "============================================"
echo ""

# Check if backend/.env exists
if [ ! -f "backend/.env" ]; then
    echo "❌ ERROR: backend/.env not found!"
    echo "   Create backend/.env with:"
    echo "   HUGGINGFACE_API_KEY=hf_your_token_here"
    echo ""
    exit 1
fi

# Check if HuggingFace key is set
if grep -q "HUGGINGFACE_API_KEY=hf_" backend/.env; then
    echo "✅ HuggingFace API key configured"
else
    echo "❌ ERROR: HUGGINGFACE_API_KEY not set in backend/.env"
    exit 1
fi

# Check if embeddings exist
if [ -f "rag_pipeline/src/vectorStore/chroma.sqlite3" ]; then
    echo "✅ Embeddings database exists"
else
    echo "⚠️  WARNING: No embeddings found"
    echo "   Run: cd rag_pipeline && python create_embeddings.py"
fi

# Check if PDFs exist
PDF_COUNT=$(ls -1 rag_pipeline/data/*.pdf 2>/dev/null | wc -l)
if [ $PDF_COUNT -gt 0 ]; then
    echo "✅ $PDF_COUNT PDF files found in data/"
else
    echo "⚠️  WARNING: No PDF files in rag_pipeline/data/"
    echo "   Add PDF files to generate embeddings"
fi

echo ""
echo "============================================"
echo "   To start the system:"
echo "============================================"
echo ""
echo "Terminal 1 - RAG Pipeline:"
echo "  cd rag_pipeline"
echo "  source venv/bin/activate"
echo "  python api.py"
echo ""
echo "Terminal 2 - Backend:"
echo "  cd backend"
echo "  npm run dev"
echo ""
echo "Terminal 3 - Frontend:"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Then open: http://localhost:5173"
echo "============================================"
