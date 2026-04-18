# 👥 Team Roles & Project Roadmap

This document outlines the division of labor, specific responsibilities, and the sequential handoff points for the **Scalable Academic Policy QA System**.

---

## 🏗 Phase 1: Data Infrastructure & Baseline 
**Assigned to:** Ahmed Ibrahim   
**Focus:** Data Ingestion, System Design, and Ground Truth establishment.

### ✅ Key Responsibilities:
- **PDF Parsing:** Extraction and cleaning of text from UG/PG handbooks.
- **Chunking Strategy:** Implementing a sliding-window overlap (300 words / 50 overlap).
- **Baseline Retrieval:** Implementing TF-IDF + Cosine Similarity ($O(N)$ complexity).
- **Scalability Prep:** Generating the 50x synthetic dataset for stress testing.

### 📦 Handoff Deliverables:
- `data/processed/chunks.json`
- `data/processed/massive_chunks.json`
- `src/baseline_retrieval.py`

**Grading Impact:** System Design (20%), Baseline Method (Required).

---

## ⚡ Phase 2: Big Data Retrieval Engine
**Assigned to:** Zain Amjad
**Focus:** Mathematical implementation of Locality Sensitive Hashing.

### ✅ Key Responsibilities:
- **MinHash + LSH:** Implementing signature matrices and banding logic for candidate retrieval.
- **SimHash:** Implementing 64-bit fingerprinting and bitwise Hamming distance filtering.
- **Hybrid Search:** Merging LSH and SimHash into a single $O(1)$ search function.

### 📦 Handoff Deliverables:
- `src/lsh_engine.py`

**Grading Impact:** Retrieval Implementation via LSH (30%).

---

## 🤖 Phase 3: GenAI Integration & Benchmarking
**Assigned to:** Abdul Rafay 
**Focus:** Interface development, LLM grounding, and performance analysis.

### ✅ Key Responsibilities:
- **LLM Agent:** Connecting the retrieval engine to OpenAI/Gemini with strict grounding prompts.
- **UI Development:** Building the Streamlit interface with source citations.
- **Experimental Analysis:** Generating the comparison graphs (Exact vs. Approximate) and Scalability tests.
- **Final Demo:** Recording the 5–7 minute walkthrough video.

### 📦 Handoff Deliverables:
- `src/app.py`
- `src/evaluation.py`
- `docs/demo_video.mp4`

**Grading Impact:** Experimental Analysis (20%), Project Demo (20%).

---

## 📝 Phase 4: Final Documentation (Collaborative)
**Assigned to:** All Team Members  
**Focus:** 6–8 Page Technical Report.

### ✅ Key Responsibilities:
- **Member 1:** System Architecture & Data Methodology.
- **Member 2:** Algorithm Explanation & LSH Mathematics.
- **Member 3:** Experimental Results & Tradeoff Analysis.

**Grading Impact:** Presentation and Report (10%).

---

## 🔗 Project Workflow (The Chain of Handoff)
1. **Member 1** finishes the Data $\rightarrow$ 2. **Member 2** uses Data to build LSH $\rightarrow$ 3. **Member 3** uses LSH to build UI/Experiments $\rightarrow$ 4. **Team** writes Final Report.