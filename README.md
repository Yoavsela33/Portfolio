# Deep Learning & Generative AI Portfolio

A collection of production-ready AI implementations demonstrating proficiency in **Large Language Models (LLMs)**, **Computer Vision**, and **Full-Stack AI Application Development**.

These projects highlight advanced architectural patterns, framework interoperability (PyTorch & TensorFlow), and the ability to deploy complex AI systems both locally and in the cloud.

---

## 🤖 Project 1: RAG-Powered Intelligent Chatbot

**Enterprise-Grade Conversational AI with Retrieval-Augmented Generation**  
*File: `rag_qa_chatbot_local.py` (Local) / `rag_qa_chatbot.py` (Cloud)*

A sophisticated, privacy-first chatbot that allows users to converse with their own documents. By implementing a modular **Retrieval-Augmented Generation (RAG)** pipeline, this system bridges the gap between static knowledge (PDFs) and dynamic AI reasoning, eliminating hallucinations and enabling precise, context-aware answers.

### Key Engineering Features
- **Swappable LLM Backends**: Engineered for flexibility, supporting both **Local Inference** (Llama 3.2 via Ollama) for strict data privacy and **Cloud APIs** (IBM WatsonX) for scalable enterprise deployment.
- **Advanced Semantic Retrieval**: 
    - Optimized document processing with semantic chunking strategies for context preservation.
    - High-dimensionality vector search using **ChromaDB** and **Nomic Embeddings** to retrieve exact information.
- **Full-Stack Implementation**: clean, responsive web interface built with **Gradio**, demonstrating end-to-end application delivery.

### Technical Stack
`LangChain` `Ollama` `Llama 3` `ChromaDB` `IBM WatsonX` `Gradio` `Python`

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

## 🛰️ Project 2: Satellite Geospatial Analysis Pipeline

**Hybrid Deep Learning System for Land Classification**  
*File: `satellite_land_classifier.py`*

A comprehensive deep learning pipeline designed to analyze high-resolution satellite imagery for agricultural monitoring. This project goes beyond basic classification by implementing and benchmarking **Hybrid Vision Transformers (CNN-ViT)** against traditional CNN architectures across multiple frameworks.

### Architectural Highlights
- **State-of-the-Art Hybrid Models**: Integrated the spatial feature extraction of CNNs with the global attention capabilities of Vision Transformers (ViT) to achieve superior classification performance.
- **Cross-Framework Proficiency**: Developed identical, rigorous implementations in both **PyTorch** and **TensorFlow/Keras**, demonstrating versatility in the two dominant AI ecosystems.
- **Robust Evaluation Suite**: Implemented a production-grade evaluation pipeline tracking ROC-AUC, F1-Score, Precision/Recall, and Confusion Matrices to ensure model reliability.

### Technical Stack
`PyTorch` `TensorFlow` `Keras` `Vision Transformers` `Scikit-Learn` `NumPy`

### Usage
```bash
# Standard training & evaluation pipeline
python3 satellite_land_classifier.py
```

---

## 📬 Contact
**Yoav Sela**  
Yoavsela@gmail.com
