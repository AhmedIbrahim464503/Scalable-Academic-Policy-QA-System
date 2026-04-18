# Agent-Based Team Roles

### 🤖 Agent 1: Data & Baseline (Member 1)
- **Input:** Raw NUST Handbooks.
- **Output:** Cleaned `chunks.json` & `baseline_engine.py`.
- **Primary Goal:** Establish the "Ground Truth" and provide the data infrastructure for the team.

### 🤖 Agent 2: Big Data Engine (Member 2)
- **Input:** `chunks.json` (from Agent 1).
- **Output:** `lsh_engine.py`.
- **Primary Goal:** Implement the Locality Sensitive Hashing math to solve the scalability bottleneck.

### 🤖 Agent 3: GenAI & Evaluator (Member 3)
- **Input:** Search functions from Agent 1 & 2.
- **Output:** `app.py` & Evaluation Graphs.
- **Primary Goal:** Transform retrieved data into human-readable answers and prove performance gains via metrics.