# Data Science & AI Portfolio

A collection of production-ready implementations demonstrating proficiency across the full ML spectrum: **Deep Learning**, **Generative AI**, **Classical Machine Learning**, and **Feature Engineering**.

These projects showcase advanced architectural patterns, framework interoperability (PyTorch, TensorFlow, scikit-learn), model interpretability, fairness analysis, and the ability to deploy complex AI systems both locally and in the cloud.

---

## 🤖 Project 1: RAG-Powered Intelligent Chatbot

**Enterprise-Grade Conversational AI with Retrieval-Augmented Generation**  
*File: `rag_qa_chatbot_local.py`*

A sophisticated, privacy-first chatbot that allows users to converse with their own documents. By implementing a modular **Retrieval-Augmented Generation (RAG)** pipeline with **conversation memory**, this system bridges the gap between static knowledge (PDFs) and dynamic AI reasoning, enabling precise, context-aware multi-turn dialogue.

### Key Engineering Features
- **Conversation Memory**: Maintains chat history across multiple turns, enabling natural follow-up questions like "Can you elaborate?" or "What about X?"
- **Swappable LLM Backends**: Engineered for flexibility, supporting both **Local Inference** (Llama 3.2 via Ollama) for strict data privacy and **Cloud APIs** for scalable enterprise deployment.
- **Advanced Semantic Retrieval**: 
    - Optimized document processing with recursive character splitting for context preservation
    - High-dimensionality vector search using **ChromaDB** and **Nomic Embeddings**
- **Production-Ready UI**: Chat interface with history display, conversation clearing, and real-time responses built with **Gradio**

### Technical Stack
`LangChain` `Ollama` `Llama 3` `ChromaDB` `Gradio` `ConversationBufferWindowMemory`

### Quick Start
```bash
# Prerequisites
ollama pull llama3.2
ollama pull nomic-embed-text

# Launch
python3 rag_qa_chatbot_local.py
# Access at http://127.0.0.1:7860
```

---

## 🛰️ Project 2: Satellite Geospatial Analysis Pipeline

**Hybrid Deep Learning System for Land Classification**  
*File: `satellite_land_classifier.py`*

A comprehensive deep learning pipeline designed to analyze high-resolution satellite imagery for agricultural monitoring. This project goes beyond basic classification by implementing and benchmarking **Hybrid Vision Transformers (CNN-ViT)** against traditional CNN architectures, with **Grad-CAM visualization** for model interpretability.

### Architectural Highlights
- **State-of-the-Art Hybrid Models**: Integrated the spatial feature extraction of CNNs with the global attention capabilities of Vision Transformers (ViT) to achieve superior classification performance.
- **Grad-CAM Explainability**: Visual explanations showing exactly what regions of satellite imagery the model uses for predictions—critical for trust in geospatial applications.
- **Cross-Framework Proficiency**: Developed identical, rigorous implementations in both **PyTorch** and **TensorFlow/Keras**, demonstrating versatility in the two dominant AI ecosystems.
- **Robust Evaluation Suite**: Production-grade evaluation pipeline tracking ROC-AUC, F1-Score, Precision/Recall, and Confusion Matrices.

### Technical Stack
`PyTorch` `TensorFlow` `Keras` `Vision Transformers` `Grad-CAM` `OpenCV` `Scikit-Learn`

### Usage
```bash
python3 satellite_land_classifier.py
```

### Grad-CAM Visualization
```python
from satellite_land_classifier import generate_gradcam_visualization

# Visualize what the CNN "sees"
generate_gradcam_visualization(model, image, class_names=['agricultural', 'non_agricultural'])
```

---

## 💰 Project 3: Debt Recovery Prediction System

**Two-Stage ML Pipeline for Financial Collections Forecasting**  
*File: `debt_recovery_predictor.py`*

A production-grade machine learning system for predicting post-default debt collection recovery. The pipeline addresses the real-world challenge of **zero-inflated financial data** through a sophisticated two-stage approach, with comprehensive **fairness auditing** for regulatory compliance.

### Key Engineering Features
- **Two-Stage Architecture**: 
    - Stage 1: XGBoost classifier with **SMOTE** for severe class imbalance (11% recovery rate)
    - Stage 2: XGBoost regressor trained only on positive recovery cases
- **Fairness & Bias Analysis**: 
    - Demographic parity ratio computation across protected attributes
    - Equalized odds (TPR/FPR) analysis for regulatory compliance (ECOA, Fair Lending)
    - Automated flagging when metrics fall below 80% threshold
- **Model Interpretability**: Full **SHAP analysis** pipeline for explaining individual predictions and global feature importance
- **Rigorous Uncertainty Quantification**: 
    - Bootstrap confidence intervals for portfolio-level estimates
    - Quantile regression for prediction bounds
- **Model Persistence**: Save/load functionality with joblib for production deployment

### Technical Stack
`XGBoost` `SHAP` `imbalanced-learn` `Scikit-Learn` `Pandas` `SciPy` `joblib`

### Usage
```bash
python3 debt_recovery_predictor.py

# Or import as module
from debt_recovery_predictor import main, run_fairness_audit
pipeline, metrics = main('debts.csv', 'collections.csv')
```

### Fairness Audit Output
```
📊 SEX Analysis:
   Male:   Positive Rate: 11.2%, TPR: 19.5%
   Female: Positive Rate: 10.8%, TPR: 18.9%
   
   Disparity Metrics:
      Demographic Parity Ratio: 0.964 ✅
      Equalized Odds (TPR): 0.969 ✅

📋 OVERALL: All fairness thresholds passed (80% rule)
```

---

## 🎮 Project 4: Competitive Match Outcome Predictor

**Time-Series ML with ELO Rating System for Gaming Analytics**  
*File: `match_outcome_predictor.py`*

A machine learning system for predicting 1v1 match outcomes in competitive gaming, demonstrating advanced **temporal feature engineering**, **ELO rating implementation**, and **pairwise prediction** techniques. The system achieves 0.81 ROC-AUC by capturing player skill evolution, deck synergies, and head-to-head dynamics.

### Key Engineering Features
- **ELO Rating System**: 
    - Full implementation of the classic skill-rating algorithm
    - Tracks player ratings over time with proper temporal handling
    - Expected win probability features for each matchup
- **Temporal Feature Engineering** (No Data Leakage):
    - Rolling window statistics for player performance trends
    - Player-deck interaction features capturing deck mastery
    - General relative score aggregating cross-deck performance
- **Antisymmetric Feature Design**: Delta features (player - opponent) ensure model predictions respect the constraint P(A wins) + P(B wins) = 1
- **Probability Calibration**: Isotonic calibration with **TimeSeriesSplit** produces well-calibrated win probabilities

### Technical Stack
`XGBoost` `Scikit-Learn` `Calibration` `ELO System` `Pandas` `NumPy`

### Usage
```bash
python3 match_outcome_predictor.py

# Or import as module
from match_outcome_predictor import main, ELORatingSystem
output = main('match_data.csv')
```

### Model Comparison Results
| Model | ROC-AUC | LogLoss | F1 | Precision |
|-------|---------|---------|-----|-----------|
| **LogReg** | **0.8123** | 0.5292 | 0.7310 | 0.7309 |
| XGB_Calibrated | 0.8107 | 0.5276 | 0.7269 | 0.7268 |
| RF_Calibrated | 0.8057 | 0.5326 | 0.7190 | 0.7189 |

*Key Insight: Logistic Regression outperforms tree-based models due to the antisymmetric feature design, which is inherently linear.*

---

## 🔧 Skills Demonstrated

| Domain | Technologies |
|--------|-------------|
| **Deep Learning** | PyTorch, TensorFlow/Keras, Vision Transformers, CNNs, Grad-CAM |
| **Generative AI** | LangChain, RAG, Ollama, LLM Integration, Conversation Memory |
| **Classical ML** | XGBoost, Random Forest, Logistic Regression, SMOTE, ELO Ratings |
| **Feature Engineering** | Time-series features, Interaction terms, Antisymmetric design |
| **Model Interpretability** | SHAP, Grad-CAM, Feature importance analysis |
| **Responsible AI** | Fairness auditing, Demographic parity, Equalized odds |
| **MLOps & Deployment** | Gradio, Model persistence, API design |
| **Statistical Analysis** | Hypothesis testing, Confidence intervals, Calibration |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Yoavsela33/Portfolio.git
cd Portfolio

# Install dependencies
pip install -r requirements.txt

# For RAG Chatbot, also install Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## 📬 Contact

**Yoav Sela**  
📧 Yoavsela@gmail.com  
🔗 [GitHub](https://github.com/Yoavsela33)
