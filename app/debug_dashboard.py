"""Streamlit dashboard for debugging retrieval behavior and fulfilling project rubric requirements."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever
from src.config import DATA_OUTPUT, MASSIVE_DATA_OUTPUT, TEST_QUERIES_PATH

st.set_page_config(page_title="NUST QA Experiments Dashboard", layout="wide")

def chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "n/a")

def chunk_page(chunk: dict[str, Any]) -> int:
    try:
        return int(chunk.get("page") or chunk.get("page_number") or 0)
    except:
        return 0

def chunk_score(chunk: dict[str, Any]) -> float:
    return float(chunk.get("score", 0.0))

@st.cache_data
def load_data(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_test_queries() -> list[dict[str, Any]]:
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def get_baseline(path: str) -> TFIDFBaseline:
    return TFIDFBaseline(path)

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

def calculate_recall(results: list[dict[str, Any]], expected_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    found_pages = {chunk_page(r) for r in results}
    hits = len(found_pages.intersection(set(expected_pages)))
    return (hits / len(expected_pages)) * 100

# Sidebar Configuration
st.sidebar.title("Global Configuration")

# Dataset Selection
use_massive = st.sidebar.checkbox("Use Massive Dataset (Scalability Mode)", value=False)
data_path = MASSIVE_DATA_OUTPUT if use_massive else DATA_OUTPUT
chunks = load_data(data_path)
chunk_payload = tuple(json.dumps(chunk, sort_keys=True) for chunk in chunks)

st.sidebar.metric("Active Dataset Size", f"{len(chunks)} chunks")

st.sidebar.divider()
st.sidebar.subheader("MinHash LSH Params")
mh_perm = st.sidebar.slider("Permutations", 32, 256, 128, 32)
mh_thresh = st.sidebar.slider("Jaccard Threshold", 0.05, 0.5, 0.1, 0.05)
mh_shingle = st.sidebar.slider("Shingle Size", 1, 5, 2, 1)

st.sidebar.divider()
st.sidebar.subheader("SimHash Params")
sh_bits = st.sidebar.selectbox("Fingerprint Bits", [64, 128], index=0)
sh_hamming = st.sidebar.slider("Hamming Threshold", 1, 32, 24)

# Initialize Retrievers
with st.spinner("Indexing dataset..."):
    baseline = get_baseline(data_path)
    minhash = get_minhash(chunk_payload, mh_perm, mh_thresh, mh_shingle)
    simhash = get_simhash(chunk_payload, sh_bits, sh_hamming)

retrievers = {"baseline": baseline, "minhash": minhash, "simhash": simhash}

# Main UI
st.title("NUST QA: Rubric Evaluation Dashboard")
st.markdown("""
This dashboard fulfills the project requirements for **Exact vs Approximate Retrieval**, **Parameter Sensitivity**, and **Scalability Testing**.
""")

tabs = st.tabs(["Live Query", "Quantitative Metrics", "Sensitivity Analysis", "Scalability Test"])

# Tab 1: Live Query
with tabs[0]:
    st.header("1. Qualitative Analysis & Live Testing")
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query = st.text_input("Enter query:", "What is the minimum GPA requirement?")
    with col_q2:
        top_k = st.number_input("Top-K", 1, 20, 5)

    if st.button("Run Evaluation", type="primary"):
        cols = st.columns(3)
        for i, (name, retriever) in enumerate(retrievers.items()):
            with cols[i]:
                st.subheader(name.upper())
                results, latency = search_with_latency(retriever, query, top_k=top_k)
                st.write(f"**Latency:** `{latency:.2f}ms`")
                for r in results:
                    with st.expander(f"Page {chunk_page(r)} (Score: {chunk_score(r):.3f})"):
                        st.write(r.get("text", ""))

# Tab 2: Quantitative Metrics
with tabs[1]:
    st.header("2. Accuracy & Latency (Precision/Recall)")
    if st.button("Run Batch Evaluation (15 Queries)"):
        test_queries = load_test_queries()
        stats = []
        progress = st.progress(0)
        
        for idx, tq in enumerate(test_queries):
            q_text = tq["query"]
            expected = tq.get("expected_pages", [])
            
            # Baseline is ground truth for precision (overlap)
            base_res, _ = search_with_latency(baseline, q_text, top_k=5)
            base_ids = {chunk_id(r) for r in base_res}
            
            for name, retriever in retrievers.items():
                res, lat = search_with_latency(retriever, q_text, top_k=5)
                
                # Recall against expected pages
                recall = calculate_recall(res, expected)
                
                # Precision (Overlap with Baseline)
                res_ids = {chunk_id(r) for r in res}
                precision = (len(res_ids & base_ids) / len(base_ids) * 100) if base_ids else 0
                
                stats.append({
                    "Query": q_text[:30] + "...",
                    "Method": name,
                    "Latency (ms)": lat,
                    "Recall (%)": recall,
                    "Precision (Overlap %)": precision
                })
            progress.progress((idx + 1) / len(test_queries))
            
        df = pd.DataFrame(stats)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(df, x="Method", y="Latency (ms)", color="Method", title="Latency Distribution"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(df.groupby("Method")[["Recall (%)", "Precision (Overlap %)"]].mean().reset_index(), 
                                  x="Method", y=["Recall (%)", "Precision (Overlap %)"], barmode="group", title="Average Accuracy Metrics"), use_container_width=True)
        
        st.dataframe(df)

# Tab 3: Sensitivity Analysis
with tabs[2]:
    st.header("3. Parameter Sensitivity Analysis")
    st.write("Evaluate how changing internal LSH parameters impacts retrieval.")
    
    sens_query = st.text_input("Sensitivity Test Query:", "What is the attendance policy?")
    
    if st.button("Run Sensitivity Scan"):
        sens_data = []
        # Test MinHash permutations
        for p in [32, 64, 128, 256]:
            tmp_mh = get_minhash(chunk_payload, p, mh_thresh, mh_shingle)
            res, lat = search_with_latency(tmp_mh, sens_query)
            sens_data.append({"Param": "Permutations", "Value": p, "Results": len(res), "Method": "MinHash"})
            
        # Test SimHash Hamming
        for h in [8, 16, 24, 32]:
            tmp_sh = get_simhash(chunk_payload, sh_bits, h)
            res, lat = search_with_latency(tmp_sh, sens_query)
            sens_data.append({"Param": "Hamming Threshold", "Value": h, "Results": len(res), "Method": "SimHash"})
            
        df_sens = pd.DataFrame(sens_data)
        st.plotly_chart(px.line(df_sens, x="Value", y="Results", color="Param", markers=True, title="Impact of Parameters on Result Count"), use_container_width=True)

# Tab 4: Scalability Test
with tabs[3]:
    st.header("4. Scalability: Normal vs Massive")
    st.info("This test compares the performance of the system as the data size increases 50x.")
    
    if st.button("Execute Scalability Benchmark"):
        # Load small dataset
        small_chunks = load_data(DATA_OUTPUT)
        small_payload = tuple(json.dumps(c, sort_keys=True) for c in small_chunks)
        
        # Load massive dataset
        massive_chunks = load_data(MASSIVE_DATA_OUTPUT)
        massive_payload = tuple(json.dumps(c, sort_keys=True) for c in massive_chunks)
        
        scalability_results = []
        sample_q = "What are the rules for failing a course?"
        
        for size_label, payload, path in [("Normal (500)", small_payload, DATA_OUTPUT), ("Massive (25k)", massive_payload, MASSIVE_DATA_OUTPUT)]:
            # We must re-index for the massive one
            with st.status(f"Indexing {size_label} dataset..."):
                b = TFIDFBaseline(path)
                m = get_minhash(payload, mh_perm, mh_thresh, mh_shingle)
                s = get_simhash(payload, sh_bits, sh_hamming)
                
                for name, ret in [("Baseline", b), ("MinHash", m), ("SimHash", s)]:
                    _, lat = search_with_latency(ret, sample_q)
                    scalability_results.append({"Scale": size_label, "Method": name, "Latency (ms)": lat})
        
        df_scale = pd.DataFrame(scalability_results)
        st.plotly_chart(px.bar(df_scale, x="Method", y="Latency (ms)", color="Scale", barmode="group", title="Latency Scaling: Normal vs Massive"), use_container_width=True)
        st.table(df_scale)
