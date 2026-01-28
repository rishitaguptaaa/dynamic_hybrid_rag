# Dynamic Hybrid RAG Document Q&A System
A hybrid Retrieval-Augmented Generation system that dynamically routes queries between vector search and knowledge-graph reasoning. Designed to run locally with minimal dependencies, making it safe to use easy to experiment, extend, or integrate into larger systems.

A project that combines **vector search** and **knowledge-graph reasoning** to answer questions from large PDFs.  
Supports four query modes: `factual`, `relationship`, `complex`, and `summary`.

---

## Project Overview

This repository implements a **hybrid Retrieval-Augmented Generation (RAG)** system that:

- Builds a vector index (semantic chunks) and a chunk-level knowledge graph from a PDF.  
- Uses an LLM-based classifier to route queries to the best retrieval strategy.  
- Synthesizes results when multiple retrieval paths are required.  
- Includes a lightweight Tkinter-based interface (`Dynamic_Hybrid_RAG.py`) for document upload and Q&A.

---

## Key Features

- Dynamic query routing: `factual`, `relationship`, `complex`, `summary`  
- Dual retrieval architecture: Vector RAG (Chroma + MMR) + Graph RAG (NetworkX)  
- Hybrid synthesis for multi-step reasoning  
- PDF ingestion with configurable chunking and overlap  
- Knowledge graph construction and visualization  
- Lightweight Tkinter-based interface for interactive use

---

## How It Works

1. Load a PDF and split it into overlapping text chunks.  
2. Build two representations:
   - A vector store for semantic search.  
   - A knowledge graph of entities and relationships.  
3. When a user asks a question:
   - An LLM-based classifier determines the query type.  
   - The system routes it to the appropriate retrieval method.  
4. For complex queries, results from both methods are synthesized into a final answer.

```
PDF
 ↓
Chunking
 ↓
 ├── Vector Store (MMR)
 ├── Knowledge Graph
 ↓
Query Classifier
 ↓
Routing / Hybrid Synthesis
```

---

## Architecture Decisions & Design Trade-Offs

**Vector + Graph instead of just Vector**  
• *Decision:* Use both semantic search and a knowledge graph.  
• *Trade-off:* More preprocessing time and complexity, but improved handling of relationship and multi-hop queries.

**Query Routing via LLM instead of fixed rules**  
• *Decision:* Let the model classify each query dynamically.  
• *Trade-off:* Adds an extra LLM call, but avoids brittle keyword rules and improves flexibility across domains.

**Chunk-level Graph Construction**  
• *Decision:* Build the graph from chunks, not full documents.  
• *Trade-off:* More nodes and edges, but better local context and more precise relationships.

**MMR Retrieval over Pure Similarity**  
• *Decision:* Use Max Marginal Relevance for vector search.  
• *Trade-off:* Slightly more computation, but reduces redundancy and improves coverage.

---

## Requirements / Prerequisites

- Python 3.8+  
- Core libraries used in this project:
  - `langchain` / `langchain-community`  
  - `chromadb`  
  - `networkx`  
  - `pdfplumber`  
  - `matplotlib`  
  - `Pillow`  
  - `tkinter` (bundled with standard Python on most systems)  
- OpenRouter-compatible LLM + embedding model access

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python Dynamic_Hybrid_RAG.py
```

3. In the GUI:

* Enter your API key.
* Upload a PDF and click **Process PDF & Create Enhanced RAGs**.
* Ask questions using `auto`, `factual`, `relationship`, `summary`, or `complex`.

---

## Example Queries

- **Factual:** “What is the definition of X?”
- **Relationship:** “How does A relate to B?”
- **Summary:** “Summarize chapter 4”
- **Complex:** “Compare X and Y and explain their impact”
