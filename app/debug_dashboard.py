"""Streamlit dashboard for debugging retrieval behavior across methods."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever

st.set_page_config(page_title="NUST QA Debug Dashboard", layout="wide")

CHUNKS_PATH = Path("data/processed/chunks.json")
TEST_QUERIES_PATH = Path("data/processed/test_queries.json")


def chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "n/a")


def chunk_page(chunk: dict[str, Any]) -> str:
    return str(chunk.get("page") or chunk.get("page_number") or "n/a")


def chunk_score(chunk: dict[str, Any]) -> float:
    return float(chunk.get("score", 0.0))


@st.cache_data
def load_chunks() -> list[dict[str, Any]]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_queries() -> list[dict[str, Any]]:
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_baseline(_chunks_path: str) -> TFIDFBaseline:
    return TFIDFBaseline(_chunks_path)


@st.cache_resource
def get_minhash(_chunks: tuple[str, ...], num_perm: int, threshold: float, shingle_size: int) -> MinHashRetriever:
    chunks = [json.loads(item) for item in _chunks]
    retriever = MinHashRetriever(num_perm=num_perm, threshold=threshold, shingle_size=shingle_size)
    retriever.create_index(chunks)
    return retriever


@st.cache_resource
def get_simhash(_chunks: tuple[str, ...], fingerprint_size: int, hamming_threshold: int) -> SimHashRetriever:
    chunks = [json.loads(item) for item in _chunks]
    retriever = SimHashRetriever(fingerprint_size=fingerprint_size, hamming_threshold=hamming_threshold)
    retriever.create_index(chunks)
    return retriever


def search_with_latency(retriever: Any, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    results = retriever.search(query, top_k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    return results, latency_ms


def overlap_percent(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_ids = {chunk_id(chunk) for chunk in left}
    right_ids = {chunk_id(chunk) for chunk in right}
    if not left_ids:
        return 0.0
    return (len(left_ids & right_ids) / len(left_ids)) * 100


st.title("NUST QA System - Debug Dashboard")
st.caption("Inspect whether related chunks are being picked accurately by each retrieval method.")

chunks = load_chunks()
chunk_payload = tuple(json.dumps(chunk, sort_keys=True) for chunk in chunks)

st.sidebar.title("Configuration")
st.sidebar.metric("Total Chunks", len(chunks))

st.sidebar.subheader("MinHash Parameters")
minhash_num_perm = st.sidebar.slider("num_perm", 32, 256, 64, 32)
minhash_threshold = st.sidebar.slider("threshold", 0.1, 0.9, 0.3, 0.1)
minhash_shingle = st.sidebar.slider("shingle_size", 1, 5, 3, 1)

st.sidebar.subheader("SimHash Parameters")
simhash_bits = st.sidebar.select_slider("fingerprint_size", options=[64, 128], value=64)
simhash_hamming = st.sidebar.slider("hamming_threshold", 1, 32, 24, 1)

baseline = get_baseline(str(CHUNKS_PATH))
minhash = get_minhash(chunk_payload, minhash_num_perm, minhash_threshold, minhash_shingle)
simhash = get_simhash(chunk_payload, simhash_bits, simhash_hamming)
retrievers = {"baseline": baseline, "minhash": minhash, "simhash": simhash}

tab1, tab2, tab3, tab4 = st.tabs(
    ["Query Analysis", "Method Comparison", "Parameter Tuning", "Batch Testing"]
)

with tab1:
    query = st.text_input("Enter your query:", "What is the attendance policy?")
    if st.button("Search All Methods", type="primary"):
        cols = st.columns(3)
        results_all: dict[str, list[dict[str, Any]]] = {}

        for index, (method_name, retriever) in enumerate(retrievers.items()):
            with cols[index]:
                st.subheader(method_name.capitalize())
                results, latency = search_with_latency(retriever, query, top_k=5)
                results_all[method_name] = results

                st.metric("Latency", f"{latency:.2f} ms")
                st.metric("Results", len(results))
                st.metric("Top Score", f"{chunk_score(results[0]):.4f}" if results else "0.0000")
                st.metric("Top Page", chunk_page(results[0]) if results else "N/A")

                for rank, result in enumerate(results, start=1):
                    with st.expander(
                        f"Rank {rank} - Page {chunk_page(result)} (Score: {chunk_score(result):.4f})"
                    ):
                        st.write(f"**Chunk ID:** {chunk_id(result)}")
                        st.write(result.get("text", ""))

        st.subheader("Overlap Analysis")
        overlap_data = pd.DataFrame(
            {
                "Comparison": [
                    "Baseline ∩ MinHash",
                    "Baseline ∩ SimHash",
                    "MinHash ∩ SimHash",
                ],
                "Overlap": [
                    overlap_percent(results_all.get("baseline", []), results_all.get("minhash", [])),
                    overlap_percent(results_all.get("baseline", []), results_all.get("simhash", [])),
                    overlap_percent(results_all.get("minhash", []), results_all.get("simhash", [])),
                ],
            }
        )
        fig = px.bar(
            overlap_data,
            x="Comparison",
            y="Overlap",
            title="Chunk Overlap Percentage",
            labels={"Overlap": "Overlap %"},
        )
        st.plotly_chart(fig, use_container_width=False)

with tab2:
    st.subheader("Performance Comparison")
    if st.button("Run Benchmark"):
        test_queries = load_test_queries()
        benchmark_rows: list[dict[str, Any]] = []
        progress = st.progress(0)

        sample_queries = test_queries[:10]
        for index, query_obj in enumerate(sample_queries):
            for method_name, retriever in retrievers.items():
                results, latency = search_with_latency(retriever, query_obj["query"], top_k=5)
                benchmark_rows.append(
                    {
                        "Query": query_obj["query"][:40] + ("..." if len(query_obj["query"]) > 40 else ""),
                        "Method": method_name,
                        "Latency (ms)": latency,
                        "Results": len(results),
                        "Top Score": chunk_score(results[0]) if results else 0.0,
                    }
                )
            progress.progress((index + 1) / len(sample_queries))

        df = pd.DataFrame(benchmark_rows)
        fig_latency = px.box(df, x="Method", y="Latency (ms)", title="Latency Distribution by Method", color="Method")
        fig_score = px.box(df, x="Method", y="Top Score", title="Top Score Distribution by Method", color="Method")
        st.plotly_chart(fig_latency, use_container_width=False)
        st.plotly_chart(fig_score, use_container_width=False)
        st.dataframe(df, use_container_width=False)

with tab3:
    st.subheader("Live Parameter Tuning")
    st.info("Adjust parameters in the sidebar and test how retrieval behavior changes.")

    tune_query = st.text_input("Test query:", "What is the minimum GPA requirement?")
    if st.button("Test Current Parameters"):
        cols = st.columns(2)

        with cols[0]:
            st.subheader("MinHash Results")
            results, latency = search_with_latency(minhash, tune_query, top_k=5)
            st.write(
                f"**Parameters:** num_perm={minhash_num_perm}, threshold={minhash_threshold}, shingle_size={minhash_shingle}"
            )
            st.write(f"**Latency:** {latency:.2f} ms")
            st.write(f"**Results:** {len(results)}")
            for rank, result in enumerate(results, start=1):
                st.write(f"{rank}. Page {chunk_page(result)} | Score {chunk_score(result):.4f} | ID {chunk_id(result)}")

        with cols[1]:
            st.subheader("SimHash Results")
            results, latency = search_with_latency(simhash, tune_query, top_k=5)
            st.write(f"**Parameters:** bits={simhash_bits}, hamming_threshold={simhash_hamming}")
            st.write(f"**Latency:** {latency:.2f} ms")
            st.write(f"**Results:** {len(results)}")
            for rank, result in enumerate(results, start=1):
                st.write(f"{rank}. Page {chunk_page(result)} | Score {chunk_score(result):.4f} | ID {chunk_id(result)}")

with tab4:
    st.subheader("Batch Query Testing")
    uploaded_file = st.file_uploader("Upload query file (JSON)", type=["json"])

    if uploaded_file:
        test_data = json.load(uploaded_file)
        if st.button("Run Batch Test"):
            rows: list[dict[str, Any]] = []
            for query_obj in test_data:
                query = query_obj["query"]
                row: dict[str, Any] = {"query": query}
                for method_name, retriever in retrievers.items():
                    results, _latency = search_with_latency(retriever, query, top_k=5)
                    row[f"{method_name}_count"] = len(results)
                    row[f"{method_name}_top_score"] = chunk_score(results[0]) if results else 0.0
                    row[f"{method_name}_top_page"] = chunk_page(results[0]) if results else "N/A"
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=False)
            st.download_button(
                "Download Results CSV",
                df.to_csv(index=False),
                "batch_results.csv",
                "text/csv",
            )

