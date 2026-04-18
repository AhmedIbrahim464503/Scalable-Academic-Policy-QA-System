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


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "or", "this",
    "these", "those", "you", "your", "we", "our", "they", "their", "if", "can", "not", "must",
}


def clean_text(text):
    """Normalize text and remove noise while preserving alphanumeric tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
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


def tokenize_for_itemsets(text):
    tokens = [t for t in text.split() if len(t) > 2 and not t.isdigit() and t not in STOPWORDS]
    return set(tokens)


def mine_frequent_itemsets(corpus, min_support=MIN_THEME_SUPPORT, top_k=MAX_THEME_COUNT):
    """Apriori-style mining over token pairs to discover common policy themes."""
    if not corpus:
        return []

    token_sets = [tokenize_for_itemsets(item["text"]) for item in corpus]
    chunk_count = len(token_sets)

    token_freq = Counter()
    for token_set in token_sets:
        token_freq.update(token_set)

    min_df = max(2, int(math.ceil(min_support * chunk_count)))
    frequent_tokens = {tok for tok, freq in token_freq.items() if freq >= min_df}
    pair_freq = Counter()

    for token_set in token_sets:
        candidate_tokens = sorted(token_set.intersection(frequent_tokens))
        for pair in combinations(candidate_tokens, 2):
            pair_freq[pair] += 1

    themes = []
    for items, freq in pair_freq.items():
        support = freq / chunk_count
        if support >= min_support:
            themes.append(
                {
                    "items": list(items),
                    "support": round(support, 6),
                    "frequency": freq,
                }
            )

    themes.sort(key=lambda x: (x["support"], x["frequency"]), reverse=True)
    return themes[:top_k]


def build_chunk_graph(corpus):
    """Connect chunks that share at least one frequent-theme tag."""
    adjacency = [set() for _ in corpus]
    tag_to_nodes = defaultdict(list)

    for idx, chunk in enumerate(corpus):
        for tag in chunk.get("tags", []):
            tag_to_nodes[tag].append(idx)

    for nodes in tag_to_nodes.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                adjacency[u].add(v)
                adjacency[v].add(u)

    return adjacency


def compute_pagerank(adjacency):
    """Power-iteration PageRank over the chunk similarity graph."""
    n = len(adjacency)
    if n == 0:
        return []

    damping = PAGERANK_DAMPING
    ranks = [1.0 / n] * n

    for _ in range(PAGERANK_MAX_ITER):
        new_ranks = [(1.0 - damping) / n] * n
        sink_rank = sum(ranks[i] for i in range(n) if not adjacency[i])

        for i in range(n):
            neighbors = adjacency[i]
            if not neighbors:
                continue
            contribution = ranks[i] / len(neighbors)
            for neighbor in neighbors:
                new_ranks[neighbor] += damping * contribution

        sink_contribution = damping * sink_rank / n
        new_ranks = [value + sink_contribution for value in new_ranks]

        delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if delta < PAGERANK_TOL:
            break

    rank_sum = sum(ranks)
    if rank_sum > 0:
        ranks = [r / rank_sum for r in ranks]
    return ranks


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

    def _attach_theme_tags(self, corpus, themes):
        theme_pairs = [tuple(theme["items"]) for theme in themes]
        for chunk in corpus:
            token_set = tokenize_for_itemsets(chunk["text"])
            tags = []
            for left, right in theme_pairs:
                if left in token_set and right in token_set:
                    tags.append(f"{left}_{right}")
            chunk["tags"] = tags

    def create_chunks(self):
        """Build enriched corpus with MapReduce ingestion, themes, and PageRank."""
        page_payloads = self._extract_page_payloads()
        corpus = self._mapreduce_chunking(page_payloads)

        themes = mine_frequent_itemsets(corpus)
        self._attach_theme_tags(corpus, themes)

        graph = build_chunk_graph(corpus)
        scores = compute_pagerank(graph)
        for idx, chunk in enumerate(corpus):
            chunk["pagerank_score"] = round(scores[idx], 10) if idx < len(scores) else 0.0

        with open(THEMES_OUTPUT, "w", encoding="utf-8") as theme_file:
            json.dump(themes, theme_file, indent=2)

        with open(DATA_OUTPUT, "w", encoding="utf-8") as data_file:
            json.dump(corpus, data_file, indent=2)

        print(f"Created {len(corpus)} chunks in {DATA_OUTPUT}")
        print(f"Saved {len(themes)} themes in {THEMES_OUTPUT}")


if __name__ == "__main__":
    mp.freeze_support()
    agent = DataIngestionAgent()
    agent.create_chunks()
