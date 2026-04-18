import json
import uuid
from config import DATA_OUTPUT, MASSIVE_DATA_OUTPUT, SCALABILITY_FACTOR

def generate_massive_dataset():
    """Simulates Big Data by duplicating the corpus for stress testing."""
    with open(DATA_OUTPUT, "r") as f:
        original_data = json.load(f)

    massive_corpus = []
    print(f"🚀 Augmenting data {SCALABILITY_FACTOR}x...")

    for i in range(SCALABILITY_FACTOR):
        for item in original_data:
            # Deep copy and give new unique ID for hashing
            new_item = item.copy()
            new_item["id"] = str(uuid.uuid4()) 
            massive_corpus.append(new_item)

    with open(MASSIVE_DATA_OUTPUT, "w") as f:
        json.dump(massive_corpus, f, indent=4)
    
    print(f"✅ Scalability dataset created: {len(massive_corpus)} chunks.")

if __name__ == "__main__":
    generate_massive_dataset()