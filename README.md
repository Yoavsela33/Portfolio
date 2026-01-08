# Deep Learning & Generative AI Portfolio

A collection of production-ready AI implementations demonstrating proficiency in **Computer Vision**, **Large Language Models (LLMs)**, and **MLOps**. 

These projects highlight advanced architectural patterns, framework interoperability (PyTorch & TensorFlow), and full-stack AI application development.

---

## 🛰️ Project 1: Satellite Imagery Classification Pipeline

**Advanced Deep Learning System for Geospatial Analysis**  
*File: `satellite_land_classifier.py`*

A robust, modular training and evaluation pipeline designed to classify satellite imagery into agricultural and non-agricultural categories. This project demonstrates framework mastery by implementing identical architectures across **PyTorch** and **TensorFlow/Keras** to benchmark performance and training dynamics.

### Architectural Highlights
- **Hybrid Vision Transformers (CNN-ViT)**: Implemented state-of-the-art hybrid models combining the local feature extraction of CNNs with the global attention mechanisms of Transformers.
- **Dual-Framework Implementation**: Complete parity between Keras and PyTorch implementations for:
    - **Custom CNN Architectures**: 6-layer optimized deep convolutional networks.
    - **Vision Transformers**: Manual implementation of Patch Embeddings, Multi-Head Self-Attention (MHSA), and Transformer Encoder blocks.
- **Rigorous Evaluation**: Comprehensive metric suite including ROC-AUC, F1-Score, Precision/Recall, and Confusion Matrices.

### Technical Stack
`PyTorch` `TensorFlow` `Keras` `Vision Transformers` `Scikit-Learn` `NumPy`

### Usage
```bash
# Standard training & evaluation pipeline
python3 satellite_land_classifier.py
```

---

## 🤖 Project 2: RAG Knowledge Retrieval System

**Enterprise-Grade Document Q&A Assistant**  
*File: `rag_qa_chatbot_local.py` (Local) / `rag_qa_chatbot.py` (Cloud)*

A scalable Retrieval-Augmented Generation (RAG) system that transforms static PDF documentation into an interactive knowledge base. Designed with a modular architecture to support swappable LLM backends (Local Ollama vs. Cloud API).

### Engineering Features
- **Flexible Backend Architecture**: 
    - **Privacy-First Local Mode**: Fully offline execution using **Llama 3.2** and **Nomic Embeddings** via Ollama.
    - **Cloud Mode**: Integration with IBM WatsonX Granite models for enterprise scalability.
- **Advanced Retrieval Pipeline**: 
    - Semantic chunking strategies with `RecursiveCharacterTextSplitter`.
    - High-performance vector similarity search using **ChromaDB**.
- **Interactive Interface**: Clean, responsive web UI built with **Gradio**.

### Technical Stack
`LangChain` `Ollama` `Llama 3` `ChromaDB` `IBM WatsonX` `Gradio`

### Quick Start (Local Version)

1. **Prerequisites** (Official Ollama):
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Launch Application**:
   ```bash
   python3 rag_qa_chatbot_local.py
   ```
   *Access the UI at http://127.0.0.1:7860*

---

## 📬 Contact
**Yoav Sela**  
yoavsela@gmail.com
