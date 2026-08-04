# Triplet RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/RAG-Triplet--Driven-orange.svg)](#workflow--pipeline)

**Triplet RAG System** is a Retrieval-Augmented Generation (RAG) framework designed to improve factual precision, multi-hop reasoning, and retrieval efficiency. Unlike traditional chunk-based RAG pipelines that retrieve large blocks of unstructured text, this system extracts, indexes, and retrieves structured knowledge in the form of **atomic triplets (`Subject` → `Predicate` → `Object`)**.

---

# 📖 Table of Contents

- [Overview](#triplet-rag-system)
- [Workflow & Pipeline](#-workflow--pipeline)
- [Tech Stack](#-tech-stack--dependencies)
- [Supported Models](#-supported-models)
- [Environment Variables](#-environment-variables--api-requirements)
- [Installation](#-how-to-use)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

# 🔄 Workflow & Pipeline

```text
┌─────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│ Raw Documents   │ ──► │  Triplet Extractor    │ ──► │ Vector & Graph Stores  │
│ (PDF/TXT/Doc)   │     │ (Subject-Rel-Object)  │     │ (Embeddings & Graph)   │
└─────────────────┘     └───────────────────────┘     └────────────────────────┘
                                                                   │
                                                                   ▼
┌─────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│ LLM Generation  │ ◄── │ Hybrid Retriever &    │ ◄── │ User Query &           │
│ (with Evidence) │     │ Re-Ranker             │     │ Placeholder Expansion  │
└─────────────────┘     └───────────────────────┘     └────────────────────────┘
```

## 1. Document Ingestion & Parsing

Raw documents (PDF, DOCX, TXT, Markdown, etc.) are parsed and segmented into clean text passages suitable for downstream processing.

## 2. Triplet Extraction

Each document segment is transformed into structured factual knowledge using either:

- Prompt-engineered LLMs
- Relation extraction models
- Named Entity Recognition (NER)

Output format:

```text
(Subject, Predicate, Object)
```

Example:

```text
(Telkom University, located_in, Bandung)
(Python, supports, Object-Oriented Programming)
```

---

## 3. Indexing & Vectorization

Every extracted triplet is embedded into a semantic vector representation using embedding models.

Each record stores:

- Subject
- Predicate
- Object
- Source document
- Metadata
- Embedding vector

The embeddings are stored inside a Vector Database such as:

- Qdrant
- ChromaDB
- FAISS

Optionally, triplets are also inserted into a graph database (Neo4j or NetworkX) to enable graph traversal and multi-hop reasoning.

---

## 4. Query Decomposition & Retrieval

Instead of searching long chunks of text, the user query is decomposed into triplet patterns.

Example:

```text
Question:
Where is Telkom University located?

↓

Triplet Pattern:
(Telkom University, located_in, ?)
```

Hybrid retrieval combines:

- Dense vector similarity
- Sparse keyword matching
- Graph neighborhood expansion

---

## 5. Re-Ranking & Evidence Assembly

Candidate triplets are ranked according to:

- Semantic similarity
- Predicate alignment
- Entity overlap
- Graph connectivity
- Source confidence

Only the highest-quality evidence is passed to the language model.

---

## 6. Grounded LLM Generation

The LLM receives structured evidence rather than raw documents, enabling:

- Higher factual accuracy
- Lower hallucination rate
- Better explainability
- Source-backed answers

---

# 🛠️ Tech Stack & Dependencies

| Component | Technology | Version |
|------------|------------|----------|
| Language | Python | >=3.10 |
| LLM Framework | LangChain / LlamaIndex | >=0.1 |
| Vector Database | Qdrant / ChromaDB / FAISS | Latest |
| Graph Store | NetworkX / Neo4j | Optional |
| Embeddings | HuggingFace / OpenAI | sentence-transformers >=2.5 |
| API | FastAPI | >=0.110 |
| UI | Streamlit | Latest |

---

# 🤖 Supported Models

## Large Language Models

### OpenAI

- GPT-4o
- GPT-4 Turbo
- GPT-3.5 Turbo

### Google Gemini

- Gemini 1.5 Pro
- Gemini 1.5 Flash

### Local Models

- Llama 3 (8B)
- Mistral 7B
- Any Ollama-compatible model
- vLLM deployments

---

## Embedding Models

### OpenAI

- text-embedding-3-small
- text-embedding-3-large

### Hugging Face

- BAAI/bge-m3
- sentence-transformers/all-MiniLM-L6-v2
- Other SentenceTransformer models

---

# 🔑 Environment Variables & API Requirements

Create a `.env` file:

```env
# -------------------------
# LLM API Keys
# -------------------------

OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# -------------------------
# Model Configuration
# -------------------------

LLM_MODEL_NAME=gpt-4o
EMBEDDING_MODEL_NAME=text-embedding-3-small

# -------------------------
# Vector Database
# -------------------------

VECTOR_DB_TYPE=qdrant

QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional
# QDRANT_API_KEY=your_api_key

# -------------------------
# Neo4j (Optional)
# -------------------------

# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password
```

---

# 🚀 How to Use

## 1. Clone Repository

```bash
git clone https://github.com/farhanahmadn/Triplet_RAG_System.git

cd Triplet_RAG_System
```

---

## 2. Create Virtual Environment

Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

Windows

```powershell
python -m venv venv

venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Build Knowledge Index

Extract triplets from documents:

```bash
python main.py \
    --mode ingest \
    --data-path ./data/sample.pdf
```

---

## 5. Query the RAG System

Example:

```bash
python main.py \
    --mode query \
    --prompt "What is the relation between Component A and Component B?"
```

---

## 6. Run FastAPI Server

```bash
uvicorn app.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

---

## 7. Run Streamlit UI

```bash
streamlit run app/ui.py
```

---

# 📁 Project Structure

```text
Triplet_RAG_System/
│
├── app/
│   ├── api.py
│   └── ui.py
│
├── data/
│
├── src/
│   ├── extraction/
│   │   ├── parser.py
│   │   └── triplet_extractor.py
│   │
│   ├── indexing/
│   │   ├── vector_store.py
│   │   └── graph_store.py
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   │
│   └── pipeline.py
│
├── main.py
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

# ⭐ Features

- 📄 Multi-format document ingestion
- 🧠 Automatic Subject–Predicate–Object extraction
- 🔎 Hybrid retrieval (Vector + Graph)
- 📚 Dense semantic embeddings
- 🌐 Optional Knowledge Graph
- 🎯 Re-ranking for better evidence quality
- 🤖 LLM-grounded answer generation
- 📖 Source-aware responses
- ⚡ FastAPI REST API
- 🖥️ Streamlit Web UI
- 🔌 Supports OpenAI, Gemini, Ollama, and local models

---

# 📊 Advantages Over Traditional RAG

| Traditional RAG | Triplet RAG |
|-----------------|------------|
| Retrieves long chunks | Retrieves atomic facts |
| More hallucination | Better factual grounding |
| Hard multi-hop reasoning | Native graph reasoning |
| Large context window | Compact structured evidence |
| Weak explainability | Explainable triplets |

---

# 🤝 Contributing

Contributions, feature requests, and pull requests are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/new-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push to your branch.

```bash
git push origin feature/new-feature
```

5. Open a Pull Request.

---

# 📜 License

This project is distributed under the **MIT License**.

See the **LICENSE** file for more information.

---

# 👨‍💻 Author

**Farhan Ahmad**

- GitHub: https://github.com/farhanahmadn

---

## ⭐ If you find this project useful, don't forget to give it a Star!