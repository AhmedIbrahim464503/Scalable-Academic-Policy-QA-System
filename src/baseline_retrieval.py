import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import DATA_OUTPUT
except ImportError:
    from config import DATA_OUTPUT


class BaselineRetrievalAgent:
    def __init__(self, data_path=DATA_OUTPUT):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.corpus = [item["text"] for item in self.data]
        self.authority = np.array([float(item.get("pagerank_score", 0.0)) for item in self.data])
        if len(self.authority) > 0:
            max_authority = float(np.max(self.authority))
            if max_authority > 0:
                self.authority = self.authority / max_authority

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        custom_stops = {
            "nust", "university", "student", "handbook", "policy", "chapter", "page", "section",
            "cumulative", "grade", "point", "average", "requirement", "rule", "regulation"
        }
        all_stops = list(ENGLISH_STOP_WORDS.union(custom_stops))

        self.vectorizer = TfidfVectorizer(
            stop_words=all_stops,
            ngram_range=(1, 2),      # capture phrases like "credit hours", "minimum gpa"
            sublinear_tf=True,       # dampen dominant terms in long chunks
            max_df=0.75,             # ignore terms in >75% of chunks to heavily filter noise
            min_df=2,                # ignore terms appearing only once
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query, top_k=3):
        """Exact lexical retrieval (TF-IDF) weighted by PageRank authority."""
        start_time = time.time()

        query_vec = self.vectorizer.transform([query.lower()])
        relevance_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # Use strictly reduced authority weight so specific rules aren't drowned by "important" pages
        final_scores = relevance_scores * (1.0 + 0.05 * self.authority)

        top_indices = np.argsort(final_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.data[idx].get("id"),
                "source": self.data[idx].get("source"),
                "page": self.data[idx]["page"],
                "text": self.data[idx]["text"],
                "relevance_score": float(relevance_scores[idx]),
                "pagerank_score": float(self.data[idx].get("pagerank_score", 0.0)),
                "score": float(final_scores[idx]),
            })

        latency = time.time() - start_time
        return results, latency


if __name__ == "__main__":
    agent = BaselineRetrievalAgent()
    query = "What is the minimum CGPA requirement?"
    res, t = agent.retrieve(query)
    top = res[0]
    print(f"Top Result [{top['source']} p.{top['page']}]: {top['text'][:120]}...")
    print(
        f"Scores -> relevance={top['relevance_score']:.4f}, "
        f"pagerank={top['pagerank_score']:.6f}, final={top['score']:.4f}"
    )
    print(f"Baseline Latency: {t:.4f} seconds")
