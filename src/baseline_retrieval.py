import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import DATA_OUTPUT

class BaselineRetrievalAgent:
    def __init__(self, data_path=DATA_OUTPUT):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.corpus = [item['text'] for item in self.data]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query, top_k=3):
        """O(N) search complexity using exact cosine similarity."""
        start_time = time.time()
        
        query_vec = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.data[idx]['text'],
                "page": self.data[idx]['page'],
                "score": float(similarities[idx])
            })
            
        latency = time.time() - start_time
        return results, latency

if __name__ == "__main__":
    agent = BaselineRetrievalAgent()
    query = "What is the minimum CGPA requirement?"
    res, t = agent.retrieve(query)
    print(f"🔍 Top Result (Page {res[0]['page']}): {res[0]['text'][:100]}...")
    print(f"⏱ Baseline Latency: {t:.4f} seconds")