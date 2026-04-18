# System Architecture Specification

## 1. Retrieval Strategy: Hybrid Approximate vs. Exact
The system is designed to handle "Big Data" scale by moving away from linear scans.
- **Exact Match (Baseline):** Uses **TF-IDF**. Good for small datasets, but fails as $N \rightarrow \infty$ due to $O(N)$ search complexity.
- **Approximate Match (LSH):** 
    - **MinHash:** Reduces high-dimensional text sets into small signatures.
    - **LSH Banding:** Maps similar signatures into the same "bucket" to find candidates without searching the whole database.
    - **SimHash:** Provides a 64-bit fingerprint to filter candidates via Hamming Distance.

## 2. Pipeline Flow
`PDF` → `Normalization` → `Sliding Window Chunking` → `MinHash Signature Matrix` → `LSH Buckets` → `Candidate Selection` → `LLM Grounding` → `Final Answer`.

## 3. Key Design Tradeoffs
- **Precision vs. Recall:** LSH may miss some chunks (False Negatives) but significantly reduces retrieval time.
- **Memory vs. Latency:** Storing hash tables uses more RAM but reduces query latency from seconds to milliseconds.