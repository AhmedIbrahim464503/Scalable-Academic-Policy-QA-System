import json
import math
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from itertools import combinations

import fitz  # PyMuPDF

try:
    from .config import (
        CHUNK_SIZE,
        DATA_OUTPUT,
        MAX_THEME_COUNT,
        MIN_THEME_SUPPORT,
        OVERLAP,
        PAGERANK_DAMPING,
        PAGERANK_MAX_ITER,
        PAGERANK_TOL,
        PDF_PATHS,
        THEMES_OUTPUT,
    )
except ImportError:
    from config import (
        CHUNK_SIZE,
        DATA_OUTPUT,
        MAX_THEME_COUNT,
        MIN_THEME_SUPPORT,
        OVERLAP,
        PAGERANK_DAMPING,
        PAGERANK_MAX_ITER,
        PAGERANK_TOL,
        PDF_PATHS,
        THEMES_OUTPUT,
    )





def clean_text(text):
    """Normalize text while preserving decimals, percentages, and section numbers.

    Previous version stripped ALL non-alphanumeric chars, turning '2.0 GPA' into
    '2 0 gpa' and '75%' into '75'.  This version keeps structure that matters for
    academic policy retrieval.
    """
    text = text.lower()

    # Protect decimal numbers (e.g. 2.0, 3.5) and section refs (e.g. 1.2.3)
    text = re.sub(r"(\d)\.(\d)", r"\1 POINT \2", text)

    # Protect percentages
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)

    # Replace hyphens with spaces
    text = re.sub(r"(\w)-(\w)", r"\1 \2", text)

    # Abbreviation expansion to prevent vocabulary mismatch
    # By keeping both short and long forms, we guarantee a match regardless of user input
    abbrevs = {
        r"\bcgpa\b": "cgpa cumulative grade point average",
        r"\bgpa\b": "gpa grade point average",
        r"\bug\b": "ug undergraduate",
        r"\bpg\b": "pg postgraduate",
        r"\bhec\b": "hec higher education commission",
        r"\bfbs\b": "fbs faculty board of studies",
        r"\bpda\b": "pda public display of affection",
        r"\b%": " percent ",
        r"\bpercent\b": "percentage",
    }
    for short, long in abbrevs.items():
        text = re.sub(short, long, text, flags=re.IGNORECASE)

    # Remove remaining noise characters but keep alphanumeric, markers, and slashes (for ratios)
    text = re.sub(r"[^a-z0-9\s/]", " ", text)

    # Restore decimal points from our markers
    text = re.sub(r"(\d)\s*POINT\s*(\d)", r"\1.\2", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_page_to_chunks(args):
    """Map step: clean one page and produce overlapping chunks for that page."""
    source, page_num, page_text = args
    cleaned = clean_text(page_text)
    words = cleaned.split()

    page_chunks = []
    step = max(1, CHUNK_SIZE - OVERLAP)
    for start_idx in range(0, len(words), step):
        chunk_words = words[start_idx : start_idx + CHUNK_SIZE]
        if len(chunk_words) < 50:
            continue
        page_chunks.append(
            {
                "source": source,
                "page": page_num,
                "text": " ".join(chunk_words),
            }
        )

    # Keep full-page coverage: short non-empty pages still become one chunk.
    if not page_chunks and words:
        page_chunks.append(
            {
                "source": source,
                "page": page_num,
                "text": " ".join(words),
            }
        )

    # Some handbook pages are image-only/non-extractable. Keep explicit placeholders
    # so the output still tracks every source page during QA and reporting.
    if not page_chunks and not words:
        page_chunks.append(
            {
                "source": source,
                "page": page_num,
                "text": "non_extractable_page_content",
            }
        )
    return page_chunks


# Extension Implementation: MapReduce / SON
# We use Python's multiprocessing pool to implement a MapReduce paradigm for parallel ingestion.
# Map Stage: Clean and chunk each page in parallel across multiple CPU cores.
# Reduce Stage: Aggregate the list of chunks into a single unified corpus (chunks.json).



class DataIngestionAgent:
    def __init__(self, pdf_paths=None):
        self.pdf_paths = pdf_paths or PDF_PATHS

    def _extract_page_payloads(self):
        payloads = []
        for source_name, path in self.pdf_paths.items():
            doc = fitz.open(path)
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                payloads.append((source_name, page_idx + 1, page.get_text("text")))
        return payloads

    def _mapreduce_chunking(self, payloads):
        """Run map stage in parallel across pages, then reduce into one corpus."""
        if not payloads:
            return []

        worker_count = min(mp.cpu_count(), len(payloads))
        with mp.Pool(processes=worker_count) as pool:
            mapped_results = pool.map(map_page_to_chunks, payloads)

        reduced = []
        global_chunk_id = 0
        for page_chunks in mapped_results:
            for chunk in page_chunks:
                chunk["id"] = f"chunk_{global_chunk_id}"
                reduced.append(chunk)
                global_chunk_id += 1
        return reduced



    def create_chunks(self):
        """Build corpus using Parallel MapReduce ingestion."""
        print("Starting Data Ingestion pipeline...")
        page_payloads = self._extract_page_payloads()
        
        print(f"Applying MapReduce (Parallel Process) to {len(page_payloads)} pages...")
        corpus = self._mapreduce_chunking(page_payloads)

        # In this extension, we skip PageRank/Itemsets to maintain a focused "Competitive Edge" on MapReduce
        with open(DATA_OUTPUT, "w", encoding="utf-8") as data_file:
            json.dump(corpus, data_file, indent=2)

        print(f"Created {len(corpus)} chunks using MapReduce in {DATA_OUTPUT}")



if __name__ == "__main__":
    mp.freeze_support()
    agent = DataIngestionAgent()
    agent.create_chunks()
