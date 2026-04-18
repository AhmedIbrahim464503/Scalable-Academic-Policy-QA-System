import fitz  # PyMuPDF
import json
import re
from config import PDF_PATH, DATA_OUTPUT, CHUNK_SIZE, OVERLAP

class DataIngestionAgent:
    def __init__(self):
        self.pdf_path = PDF_PATH

    def clean_text(self, text):
        """Removes noise, standardizes whitespace and case."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def create_chunks(self):
        """Implements Sliding Window Chunking for Semantic Continuity."""
        doc = fitz.open(self.pdf_path)
        corpus = []
        global_chunk_id = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = self.clean_text(page.get_text("text"))
            words = text.split()

            # Sliding window: step = size - overlap
            for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                chunk_words = words[i : i + CHUNK_SIZE]
                if len(chunk_words) < 50: continue # Skip fragments

                corpus.append({
                    "id": f"chunk_{global_chunk_id}",
                    "page": page_num + 1,
                    "text": " ".join(chunk_words)
                })
                global_chunk_id += 1

        with open(DATA_OUTPUT, "w") as f:
            json.dump(corpus, f, indent=4)
        print(f"✅ Created {len(corpus)} chunks in {DATA_OUTPUT}")

if __name__ == "__main__":
    agent = DataIngestionAgent()
    agent.create_chunks()