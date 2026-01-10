"""
================================================================================
RAG Q&A CHATBOT - CONVERSATIONAL AI WITH MEMORY
================================================================================
PDF-based question answering using Retrieval-Augmented Generation (RAG).
Features conversation memory for multi-turn dialogue and context retention.
Uses Ollama for local LLM inference - no cloud API required.

Skills Demonstrated:
• LLM integration (local models via Ollama)
• RAG architecture and vector databases
• Conversation memory and context management
• Document processing and semantic chunking
• Semantic embeddings with similarity search
• Production-ready web UI (Gradio)

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

from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
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

# Conversation memory settings
MEMORY_WINDOW_SIZE = 5  # Number of previous exchanges to remember


# ==============================================================================
# CUSTOM PROMPT TEMPLATE
# ==============================================================================

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the 
follow-up question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""")

QA_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on the provided document context.
Use the following pieces of context to answer the question. If you don't know the answer
based on the context, say "I don't have enough information in the document to answer that."

Context from document:
{context}

Question: {question}

Helpful Answer:""")


# ==============================================================================
# LLM SETUP (LOCAL)
# ==============================================================================

def get_llm() -> OllamaLLM:
    """
    Initialize local LLM using Ollama.
    
    Ollama runs models locally - no API keys or cloud access needed.
    Temperature set to 0 for consistent, deterministic responses.
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
    """
    Split documents into overlapping chunks for embedding.
    
    Uses recursive character splitting to maintain semantic coherence
    by preferring splits at paragraph and sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


# ==============================================================================
# EMBEDDING & VECTOR DATABASE
# ==============================================================================

def get_embedding_model() -> OllamaEmbeddings:
    """Initialize local embedding model using Ollama."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_vector_database(chunks: list) -> Chroma:
    """
    Create ChromaDB vector store from document chunks.
    
    ChromaDB provides efficient similarity search for retrieval.
    """
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
# CONVERSATION MEMORY
# ==============================================================================

@dataclass
class ConversationState:
    """
    Manages conversation state including memory and document context.
    
    Maintains:
    - Chat history for context-aware responses
    - Currently loaded document path
    - Retrieval chain for Q&A
    """
    memory: ConversationBufferWindowMemory = field(default=None)
    current_file: Optional[str] = None
    chain: Optional[ConversationalRetrievalChain] = None
    chat_history: List[Tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def reset(self):
        """Reset conversation state for new document."""
        self.memory.clear()
        self.chat_history = []
        self.chain = None
        self.current_file = None


# Global conversation state
conversation_state = ConversationState()


# ==============================================================================
# RAG CHAIN WITH MEMORY
# ==============================================================================

def build_conversational_chain(file_path: str) -> ConversationalRetrievalChain:
    """
    Build RAG chain with conversation memory.
    
    The chain:
    1. Condenses follow-up questions using chat history
    2. Retrieves relevant document chunks
    3. Generates context-aware answers
    """
    llm = get_llm()
    retriever = build_retriever(file_path)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=conversation_state.memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    
    return chain


def answer_question(file_path: str, query: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """
    Answer a question using RAG with conversation memory.
    
    Maintains conversation context across multiple turns, allowing
    follow-up questions like "Can you elaborate?" or "What about X?"
    
    Args:
        file_path: Path to PDF document
        query: User's question
        history: Gradio chat history format
        
    Returns:
        Tuple of (answer, updated_history)
    """
    global conversation_state
    
    if not file_path:
        return "Please upload a PDF file first.", history
    if not query or not query.strip():
        return "Please enter a question.", history
    
    # Check if we need to load a new document
    file_path_str = file_path if isinstance(file_path, str) else file_path.name
    
    if conversation_state.current_file != file_path_str:
        print(f"📄 Loading new document: {file_path_str}")
        conversation_state.reset()
        conversation_state.current_file = file_path_str
        conversation_state.chain = build_conversational_chain(file_path_str)
    
    # Generate response
    try:
        response = conversation_state.chain.invoke({"question": query})
        answer = response["answer"]
        
        # Update chat history for display
        history = history + [[query, answer]]
        conversation_state.chat_history.append((query, answer))
        
        return answer, history
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        return error_msg, history


def clear_conversation():
    """Clear conversation history and memory."""
    global conversation_state
    conversation_state.reset()
    return [], ""


# ==============================================================================
# GRADIO WEB INTERFACE
# ==============================================================================

def create_gradio_interface() -> gr.Blocks:
    """
    Create Gradio web interface with chat functionality.
    
    Features:
    - PDF upload
    - Chat interface with history
    - Clear conversation button
    - Conversation memory indicator
    """
    with gr.Blocks(
        title="RAG Q&A Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .chatbot-container { height: 500px; }
        .memory-indicator { 
            padding: 8px; 
            background: #e8f4ea; 
            border-radius: 8px; 
            margin-bottom: 10px;
            font-size: 14px;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🤖 RAG Q&A Chatbot with Conversation Memory
        
        Upload a PDF document and ask questions. The chatbot remembers your conversation
        context, so you can ask follow-up questions naturally.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📄 Upload PDF Document",
                    file_count="single",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                gr.Markdown(f"""
                <div class="memory-indicator">
                    💾 <strong>Memory:</strong> Remembers last {MEMORY_WINDOW_SIZE} exchanges<br>
                    🔧 <strong>Model:</strong> {LLM_MODEL} + {EMBEDDING_MODEL}
                </div>
                """)
                
                clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the document... (follow-ups welcome!)",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        submit_btn.click(
            fn=answer_question,
            inputs=[file_input, query_input, chatbot],
            outputs=[query_input, chatbot]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )
        
        query_input.submit(
            fn=answer_question,
            inputs=[file_input, query_input, chatbot],
            outputs=[query_input, chatbot]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, query_input]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["What is the main topic of this document?"],
                ["Can you summarize the key points?"],
                ["Tell me more about that."],  # Follow-up example
                ["What are the implications?"],  # Another follow-up
            ],
            inputs=[query_input],
            label="Example Questions"
        )
    
    return interface


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RAG Q&A CHATBOT - WITH CONVERSATION MEMORY")
    print("="*60)
    print("\nConfiguration:")
    print(f"  • LLM Model: {LLM_MODEL}")
    print(f"  • Embedding Model: {EMBEDDING_MODEL}")
    print(f"  • Chunk Size: {CHUNK_SIZE}")
    print(f"  • Memory Window: {MEMORY_WINDOW_SIZE} exchanges")
    
    print("\n⚠️  Make sure Ollama is running with required models:")
    print(f"   ollama pull {LLM_MODEL}")
    print(f"   ollama pull {EMBEDDING_MODEL}")
    
    print("\n🚀 Launching Gradio interface...")
    print("   Open http://localhost:7860 in your browser\n")
    
    app = create_gradio_interface()
    app.launch(share=True)
