# Scalable Academic Policy QA System (Big Data RAG)

## 🎯 Project Objective
To design and implement a scalable retrieval-augmented generation (RAG) system over university handbooks. The system compares traditional exact-match retrieval (TF-IDF) against Big Data approximate techniques (MinHash + LSH + SimHash) to analyze tradeoffs in accuracy, efficiency, and scalability.

## 🏗 System Architecture
The system follows a three-stage sequential pipeline:
1. **Data Agent (Phase 1):** Ingests raw PDFs, performs sliding-window chunking, and builds an $O(N)$ Baseline retrieval engine.
2. **Big Data Agent (Phase 2):** Implements a Hybrid LSH-based method (MinHash signature matrix + SimHash bitwise distance) for $O(1)$ sub-linear retrieval.
3. **Interface Agent (Phase 3):** Integrates LLM answer generation with grounded citations and executes performance benchmarking.

## 🛠 Tech Stack
- **Language:** Python 3.9+
- **Retrieval Math:** Scikit-Learn (TF-IDF), Custom MinHash/LSH Implementation.
- **Data Handling:** PyMuPDF, Pandas, NumPy.
- **LLM Layer:** OpenAI / Gemini API (via LangChain or direct).
- **Frontend:** React