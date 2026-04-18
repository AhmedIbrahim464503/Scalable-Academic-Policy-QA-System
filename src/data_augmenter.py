import json
import uuid

from config import DATA_OUTPUT, MASSIVE_DATA_OUTPUT, SCALABILITY_FACTOR


def generate_massive_dataset():
    """Simulate larger corpora by duplicating chunks while assigning new unique IDs."""
    with open(DATA_OUTPUT, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    massive_corpus = []
    print(f"Augmenting data {SCALABILITY_FACTOR}x...")

    for copy_index in range(SCALABILITY_FACTOR):
        for item in original_data:
            new_item = item.copy()
            new_item["id"] = f"{item.get('id', 'chunk')}_copy{copy_index}_{uuid.uuid4().hex[:8]}"
            massive_corpus.append(new_item)

    with open(MASSIVE_DATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(massive_corpus, f, indent=2)

    print(f"Scalability dataset created: {len(massive_corpus)} chunks")
    print(f"Output path: {MASSIVE_DATA_OUTPUT}")


if __name__ == "__main__":
    generate_massive_dataset()