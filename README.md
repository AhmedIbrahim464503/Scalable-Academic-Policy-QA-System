<<<<<<< HEAD
# Scalable Academic Policy QA System

This repository now combines:

- Phase 1 data ingestion and baseline retrieval code from the reference project
- a terminal QA interface for interactive testing
- the project scaffolding for later phases

The system is focused on question answering over university academic policy documents such as undergraduate and postgraduate handbooks.

## Current Status

- Phase 1: integrated
  PDF ingestion, chunking, theme mining, PageRank-style chunk authority, baseline TF-IDF retrieval, verification scripts, and processed corpora are present in `src/`, `data/`, and `docs/`.
- Terminal UI: integrated
  The Rich + prompt-toolkit CLI lives in `app/terminal/` and runs against the baseline retrieval layer.
- Later phases: pending
  MinHash/SimHash retrieval, answer generation, and broader benchmarking can be added on top of this base.

## Project Structure

```text
.
|-- app/
|   `-- terminal/       # Rich terminal QA interface
|-- data/
|   |-- raw/            # Source handbook PDFs
|   |-- processed/      # Phase 1 chunks, themes, and scalability corpus
|   `-- index/          # Search indexes and retrieval artifacts
|-- docs/               # Documentation
|-- experiments/        # Jupyter notebooks and analysis work
|-- src/                # Phase 1 pipeline and retrieval modules
|-- tests/              # Unit tests
|-- requirements.txt    # Core project dependencies
|-- requirements_terminal.txt
|-- .env.example        # Example environment variables
`-- README.md
```

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_terminal.txt
```

3. Run the terminal app:

```bash
python -m app.terminal.cli
```

4. Optional Phase 1 scripts:

```bash
python -m src.data_processor
python -m src.data_augmenter
python -m src.verify_phase1
```

## Notes

- `PyMuPDF` is installed as a package and imported in Python as `fitz`.
- The terminal CLI currently uses the baseline retrieval pipeline.
- `src/baseline.py` exists as a compatibility adapter so the CLI can call the Phase 1 retriever through a stable `search(query, top_k)` interface.
=======
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
>>>>>>> 738a1f51492ec0cbd50a18b61b65db76170a0e0a
