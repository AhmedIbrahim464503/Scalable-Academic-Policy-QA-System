# Configuration parameters for the pipeline
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = str(PROJECT_ROOT / "data" / "raw" / "Undergraduate-Handbook.pdf")
DATA_OUTPUT = str(PROJECT_ROOT / "data" / "processed" / "chunks.json")
MASSIVE_DATA_OUTPUT = str(PROJECT_ROOT / "data" / "processed" / "massive_chunks.json")

# Rubric-compliant chunking settings
CHUNK_SIZE = 300  # Words per chunk
OVERLAP = 50      # Context preservation overlap
SCALABILITY_FACTOR = 50  # Multiply data 50x for stress test