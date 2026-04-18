import json
from collections import defaultdict

import fitz

from baseline_retrieval import BaselineRetrievalAgent
from config import DATA_OUTPUT, MASSIVE_DATA_OUTPUT, PDF_PATHS, THEMES_OUTPUT


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_schema(chunks):
    required = {"id", "source", "page", "text", "tags", "pagerank_score"}
    missing = []
    for idx, chunk in enumerate(chunks):
        missing_fields = required.difference(chunk.keys())
        if missing_fields:
            missing.append((idx, sorted(missing_fields)))
    return missing


def verify_page_coverage(chunks):
    pages_in_pdf = {}
    for source, path in PDF_PATHS.items():
        doc = fitz.open(path)
        pages_in_pdf[source] = len(doc)

    pages_with_chunks = defaultdict(set)
    for chunk in chunks:
        pages_with_chunks[chunk["source"]].add(int(chunk["page"]))

    report = {}
    for source, total_pages in pages_in_pdf.items():
        missing = [p for p in range(1, total_pages + 1) if p not in pages_with_chunks[source]]
        report[source] = {
            "total_pages": total_pages,
            "chunked_pages": len(pages_with_chunks[source]),
            "missing_pages": missing,
        }
    return report


def verify_outputs_exist():
    load_json(DATA_OUTPUT)
    load_json(THEMES_OUTPUT)
    try:
        load_json(MASSIVE_DATA_OUTPUT)
        massive_exists = True
    except FileNotFoundError:
        massive_exists = False
    return massive_exists


def verify_baseline_query():
    agent = BaselineRetrievalAgent()
    results, latency = agent.retrieve("what is the minimum gpa requirement", top_k=3)
    return len(results), latency


def main():
    chunks = load_json(DATA_OUTPUT)
    themes = load_json(THEMES_OUTPUT)

    schema_issues = verify_schema(chunks)
    coverage = verify_page_coverage(chunks)
    massive_exists = verify_outputs_exist()
    result_count, latency = verify_baseline_query()

    print("=== Phase 1 Verification Report ===")
    print(f"chunks.json records: {len(chunks)}")
    print(f"top_themes.json records: {len(themes)}")
    print(f"massive_chunks.json present: {massive_exists}")
    print(f"schema issues: {len(schema_issues)}")
    if schema_issues:
        print("schema issue sample:", schema_issues[:3])

    for source, data in coverage.items():
        print(
            f"{source}: pages={data['total_pages']}, chunked_pages={data['chunked_pages']}, "
            f"missing_pages={len(data['missing_pages'])}"
        )
        if data["missing_pages"]:
            print("missing page numbers:", data["missing_pages"]) 

    print(f"baseline retrieval check: results={result_count}, latency={latency:.4f}s")

    is_ok = (
        len(schema_issues) == 0
        and all(len(v["missing_pages"]) == 0 for v in coverage.values())
        and result_count > 0
    )
    print(f"overall_status: {'PASS' if is_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
