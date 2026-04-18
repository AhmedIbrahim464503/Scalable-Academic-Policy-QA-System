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
