# Configuration parameters for the pipeline
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Paths to raw handbook PDFs (both undergraduate and postgraduate)
PDF_PATHS = {
    "undergraduate": str(RAW_DATA_DIR / "Undergraduate-Handbook.pdf"),
    "postgraduate": str(RAW_DATA_DIR / "Postgraduate-Handbook.pdf"),
}

# Backwards-compatible single path (defaults to undergraduate)
PDF_PATH = PDF_PATHS["undergraduate"]
DATA_OUTPUT = str(PROCESSED_DATA_DIR / "chunks.json")
THEMES_OUTPUT = str(PROCESSED_DATA_DIR / "top_themes.json")
MASSIVE_DATA_OUTPUT = str(PROCESSED_DATA_DIR / "massive_chunks.json")

# Rubric-compliant chunking settings
CHUNK_SIZE = 200  # Words per chunk (project spec: 200-500)
OVERLAP = 75      # Context preservation overlap
SCALABILITY_FACTOR = 50  # Multiply data 50x for stress test

# Frequent itemset + graph scoring controls
MIN_THEME_SUPPORT = 0.03
MAX_THEME_COUNT = 30
PAGERANK_DAMPING = 0.85
PAGERANK_MAX_ITER = 50
PAGERANK_TOL = 1e-6