"""
================================================================================
RAG Q&A CHATBOT - AI CAPSTONE PROJECT (LOCAL VERSION)
================================================================================
PDF-based question answering using Retrieval-Augmented Generation (RAG).
This version uses Ollama for local LLM inference - no cloud API required.

Skills Demonstrated:
• LLM integration (local models via Ollama)
• RAG architecture and vector databases
• Document processing and chunking
• Semantic embeddings
• Web UI development (Gradio)

SETUP:
1. Install Ollama: https://ollama.ai
2. Run: ollama pull llama3.2
3. Run: ollama pull nomic-embed-text
4. Run this script

Author: Yoav Sela
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Using Ollama local models - no API key needed!
LLM_MODEL = "llama3.2"  # or "mistral", "gemma2", etc.
EMBEDDING_MODEL = "nomic-embed-text"  # or "mxbai-embed-large"

# Text splitting parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80


# ==============================================================================
# LLM SETUP (LOCAL)
# ==============================================================================

def get_llm() -> OllamaLLM:
    """
    Initialize local LLM using Ollama.
    
    Ollama runs models locally - no API keys or cloud access needed.
    """
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.0,
    )


# ==============================================================================
# DOCUMENT PROCESSING PIPELINE
# ==============================================================================

def load_pdf(file_path: str) -> list:
    """Load and parse PDF document."""
    path = file_path if isinstance(file_path, str) else file_path.name
    loader = PyPDFLoader(path)
    return loader.load()


def split_documents(documents: list, 
                   chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


# ==============================================================================
# EMBEDDING & VECTOR DATABASE
# ==============================================================================

def get_embedding_model() -> OllamaEmbeddings:
    """Initialize local embedding model using Ollama."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_vector_database(chunks: list) -> Chroma:
    """Create ChromaDB vector store from document chunks."""
    embedding_model = get_embedding_model()
    
    # Filter empty chunks
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    
    # Create vector store directly from documents
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="rag_collection"
    )
    
    return vectordb


def build_retriever(file_path: str):
    """Build complete retrieval pipeline from PDF file."""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    vectordb = create_vector_database(chunks)
    return vectordb.as_retriever(search_kwargs={"k": 4})


# ==============================================================================
# RAG CHAIN
# ==============================================================================

def answer_question(file_path: str, query: str) -> str:
    """
    Answer a question using RAG over the provided PDF.
    
    RAG Process:
    1. Build retriever from PDF
    2. Retrieve relevant chunks for query
    3. Pass context + query to LLM
    4. Return generated answer
    """
    if not file_path:
        return "Please upload a PDF file first."
    if not query:
        return "Please enter a question."
        
    llm = get_llm()
    retriever = build_retriever(file_path)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa_chain.invoke({"query": query})
    return response["result"]


# ==============================================================================
# GRADIO WEB INTERFACE
# ==============================================================================

def create_gradio_interface() -> gr.Interface:
    """Create Gradio web interface for the RAG chatbot."""
    interface = gr.Interface(
        fn=answer_question,
        allow_flagging="never",
        inputs=[
            gr.File(
                label="📄 Upload PDF Document",
                file_count="single",
                file_types=[".pdf"],
                type="file"
            ),
            gr.Textbox(
                label="❓ Your Question",
                lines=2,
                placeholder="What would you like to know about this document?"
            )
        ],
        outputs=gr.Textbox(label="💬 Answer"),
        title="🤖 RAG Q&A Chatbot (Local)",
        description=(
            f"Upload a PDF document and ask questions. "
            f"Using local Ollama models: {LLM_MODEL} + {EMBEDDING_MODEL}"
        ),
        theme=gr.themes.Soft(),
    )
    return interface


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RAG Q&A CHATBOT - LOCAL VERSION (Ollama)")
    print("="*60)
    print("\nConfiguration:")
    print(f"  • LLM Model: {LLM_MODEL}")
    print(f"  • Embedding Model: {EMBEDDING_MODEL}")
    print(f"  • Chunk Size: {CHUNK_SIZE}")
    
    print("\n⚠️  Make sure Ollama is running with required models:")
    print(f"   ollama pull {LLM_MODEL}")
    print(f"   ollama pull {EMBEDDING_MODEL}")
    
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser\n")
    
    app = create_gradio_interface()
    # Launch with public link to avoid localhost issues
    app.launch(share=True)
