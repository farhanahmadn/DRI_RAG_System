# 🏛️ DRI RAG System (Digital Triplet - Spatial Governance RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/RAG-Hierarchical%20Legal%20Reasoning-orange.svg)

**DRI RAG System (Digital Triplet - Spatial Governance RAG)** adalah sistem **Retrieval-Augmented Generation (RAG)** yang dirancang untuk mendukung proses **tata kelola ruang** serta **monitoring perizinan KKPR dan PBG** di Kabupaten Sleman.

Sistem ini mengintegrasikan **Digital Twin berbasis GIS** dengan modul **Intelligent Activity World** sehingga mampu melakukan analisis terhadap dokumen hukum tata ruang, menghubungkannya dengan fakta spasial dari backend, kemudian menghasilkan **legal reasoning**, **sitasi pasal**, dan **rekomendasi penyesuaian desain** secara otomatis.

Pipeline dibangun menggunakan **LlamaIndex**, **Groq AI (Llama 3.3 70B)**, **BAAI/bge-m3 Embedding**, serta **Supabase PGVector** sebagai vector database.

---

# 📌 System Overview

DRI RAG System memiliki tiga modul utama:

1. **Legal Document Processing**
   - Parsing PDF regulasi menggunakan LlamaParse
   - Caching Markdown lokal
   - Hierarchical Parent-Child Chunking

2. **Knowledge Base Construction**
   - Embedding menggunakan BAAI/bge-m3
   - Penyimpanan vector pada Supabase PGVector
   - Penyimpanan struktur parent-child pada SimpleDocumentStore

3. **Legal Reasoning Engine**
   - Query Expansion
   - AutoMerging Retrieval
   - Fact Injection
   - Deterministic Recommendation
   - Groq LLM Reasoning
   - Structured JSON Output

---

# 🔄 Workflow & Architecture

```text
                    PDF Regulasi Hukum
                            │
                            ▼
               LlamaParse Document Parsing
                            │
                            ▼
               Local Markdown Cache System
                            │
                            ▼
          Parent-Child Hierarchical Chunking
      ┌──────────────────────────────────────┐
      │ Parent Node (2048 Token)             │
      │ Middle Node (512 Token)              │
      │ Leaf Node (128 Token)                │
      └──────────────────────────────────────┘
                            │
                            ▼
          Embedding (BAAI/bge-m3 Local Model)
                            │
                            ▼
            Supabase PostgreSQL (PGVector)
                            │
                            ▼
       Backend Payload (JejakAturan + FaktaSpasial)
                            │
                            ▼
      Query Expansion & Guardrail Bypass Logic
                            │
                            ▼
            AutoMergingRetriever (Top-k Search)
                            │
                            ▼
             Fact Injection & Target Calculator
                            │
                            ▼
          Groq Llama-3.3-70B-Versatile LLM
                            │
                            ▼
         Structured JSON (Pydantic Validation)
                            │
                            ▼
             Interactive Gradio Dashboard
```

---

# ⚙️ Pipeline Detail

## 1. Document Parsing & Local Caching

**Notebook**

```
RAG.ipynb
RAG_chunk.ipynb
```

Tahapan pertama membaca dokumen hukum berbentuk PDF menggunakan **LlamaParse API**.

Dokumen dikonversi menjadi Markdown yang lebih mudah diproses.

Hasil parsing kemudian disimpan pada folder lokal:

```
cache_markdown/
```

Keuntungan caching:

- Menghemat kuota LlamaCloud API
- Mempercepat ingestion berikutnya
- Menghindari parsing ulang dokumen yang sama

---

## 2. Parent-Child Hierarchical Legal Chunking

**Notebook**

```
RAG_chunk.ipynb
```

Dokumen hukum diproses menggunakan parser khusus **LegalDocumentChunker**.

Regex digunakan untuk mengenali struktur hukum seperti:

- BAB
- Bagian
- Paragraf
- Pasal
- Ayat
- Penjelasan

Selain itu dilakukan pembersihan:

- Header OCR
- Footer OCR
- Nomor halaman
- Noise parsing

Selanjutnya digunakan **HierarchicalNodeParser** dari LlamaIndex.

Struktur node yang dibentuk:

| Node | Ukuran | Fungsi |
|-------|---------|--------|
| Parent | 2048 Token | Konteks BAB / Bagian |
| Middle | 512 Token | Konteks Pasal |
| Leaf | 128 Token | Konteks Ayat / Huruf |

Seluruh struktur parent-child kemudian disimpan pada:

```
SimpleDocumentStore
```

---

## 3. Embedding & Vector Storage

**Notebook**

```
RAG.ipynb
```

Hanya **Leaf Nodes** yang dibuat embedding.

Model embedding:

```
BAAI/bge-m3
```

Karakteristik:

- Multilingual
- Embedding Dimension: **1024**
- Sangat baik untuk dokumen Bahasa Indonesia

Vector kemudian disimpan pada:

```
Supabase PostgreSQL
PGVector
```

Table:

```
regulasi_sleman_hierarchical
```

Menggunakan:

```
Connection Pooler Port 6543
```

---

## 4. Query Expansion & Guardrail Bypass

**Notebook**

```
LLM_as_a_Reasoner.ipynb
```

Modul menerima payload backend berupa:

- JejakAturan
- FaktaSpasial

### Guardrail Bypass

Apabila:

```
indikator == 0
```

maka proses LLM dilewati sehingga:

- lebih cepat
- hemat biaya API
- tidak melakukan reasoning yang tidak diperlukan

### Query Expansion

Singkatan teknis diubah menjadi kalimat natural.

Contoh:

| Singkatan | Expanded Query |
|------------|----------------|
| KDB | Koefisien Dasar Bangunan |
| KLB | Koefisien Lantai Bangunan |
| KDH | Koefisien Dasar Hijau |
| KTB | Koefisien Tapak Basement |
| SMP | Sempadan |
| LP2B | Lahan Pertanian Pangan Berkelanjutan |

---

## 5. Hierarchical Auto-Merging Retrieval

**Notebook**

```
LLM_as_a_Reasoner.ipynb
```

Tahapan retrieval menggunakan dua level:

### Base Retriever

Mengambil:

```
Top-6 Child Nodes
```

berdasarkan similarity embedding.

### AutoMergingRetriever

Jika beberapa child berasal dari parent yang sama maka otomatis digabung kembali menjadi parent.

Keuntungan:

- konteks hukum tetap utuh
- reasoning lebih akurat
- mengurangi kehilangan konteks

---

## 6. Fact Injection & Legal Reasoning

**Notebook**

```
LLM_as_a_Reasoner.ipynb
```

Sebelum memanggil LLM dilakukan:

### Fact Injection

Mengunci fakta hasil backend melalui:

```
susun_fakta_pelanggaran()
```

LLM tidak diperbolehkan mengubah fakta tersebut.

---

### Deterministic Target Calculator

Dilakukan menggunakan:

```
hitung_target_rekomendasi()
```

Contoh:

- Selisih KDB
- Selisih KDH
- Kekurangan sempadan
- Target penyesuaian numerik

Perhitungan dilakukan di luar LLM sehingga hasil selalu deterministik.

---

### Groq LLM Reasoning

Model:

```
llama-3.3-70b-versatile
```

Output dipaksa menggunakan:

```python
response_format={
    "type":"json_object"
}
```

Sehingga selalu menghasilkan JSON valid.

---

### Structured Output

Output mengikuti schema Pydantic:

- Ringkasan
- Reasoning Pendek
- Reasoning Panjang
- Sitasi Pasal
- Rekomendasi

---

## 7. Interactive Gradio Dashboard

Notebook:

```
LLM_as_a_Reasoner.ipynb
```

Dashboard digunakan untuk:

- Simulasi data backend
- Menampilkan reasoning
- Menampilkan HTML Executive Summary
- Menampilkan JSON Output
- Debug hasil retrieval

---

# 📂 Project Structure

```text
DRI_RAG_System/
│
├── RAG_chunk.ipynb
│     Module 1
│     Hierarchical Legal Chunking
│
├── RAG.ipynb
│     Module 2
│     LlamaParse
│     Embedding
│     PGVector Upload
│
├── LLM_as_a_Reasoner.ipynb
│     Module 3
│     AutoMergingRetriever
│     Fact Injection
│     Groq Reasoning
│     Gradio Dashboard
│
├── data_hukum_sleman/
│
│   ├── cache_markdown/
│   │     Markdown cache hasil parsing
│   │
│   └── docstore_hierarchical/
│         Parent-child document store
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Features

## 📜 Parent-Child Hierarchical Legal Chunking

Mempertahankan hubungan antar BAB, Pasal, Ayat, dan Huruf sehingga konteks hukum tetap utuh.

---

## ⚡ Local Markdown Cache

Menyimpan hasil parsing PDF sehingga tidak perlu memanggil API berulang kali.

---

## 🔗 Auto-Merging Retrieval

Menggabungkan child node menjadi parent node secara otomatis ketika berasal dari konteks yang sama.

---

## 🛡️ Fact Injection Guardrail

Mencegah hallucination dengan mengunci fakta hasil backend.

---

## 🧮 Deterministic Target Calculator

Menghitung target numerik secara deterministic di luar LLM.

---

## 🎯 Intelligent Guardrail Bypass

Apabila seluruh indikator memenuhi aturan maka proses LLM dilewati untuk menghemat biaya API.

---

## 📄 Structured JSON Output

Seluruh output tervalidasi menggunakan Pydantic sehingga siap dikonsumsi Front-End maupun Backend.

---

## 📊 Interactive Gradio Dashboard

Dashboard interaktif untuk simulasi berbagai skenario tata ruang.

---

# 💻 Tech Stack

## Programming Language

- Python 3.10+

---

## Core Framework

- LlamaIndex
- StorageContext
- VectorStoreIndex
- HierarchicalNodeParser
- AutoMergingRetriever
- SimpleDocumentStore

---

## Document Parsing

- LlamaParse
- LlamaCloud

---

## Embedding

- llama-index-embeddings-huggingface

---

## Large Language Model

- Groq API
- OpenAI SDK

---

## Database

- PostgreSQL
- Supabase PGVector
- SQLAlchemy
- asyncpg
- psycopg2

---

## Validation

- Pydantic V2

---

## User Interface

- Gradio

---

## Utilities

- pandas
- requests
- python-dotenv
- nest-asyncio
- urllib.parse
- re

---

# 🤖 Models

| Component | Model | Description |
|------------|-------|-------------|
| Embedding | **BAAI/bge-m3** | Multilingual embedding model (1024 dimensions) optimized for Indonesian legal documents |
| LLM | **llama-3.3-70b-versatile** | Groq-hosted LLM used for legal reasoning and structured response generation |
| Document Parser | **LlamaParse** | AI-powered PDF parser from LlamaCloud for extracting structured Markdown from legal documents |

---

# 🔑 Environment Variables

```env
# LlamaCloud API Key
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key

# Groq API Key
LLM_API_KEY=your_groq_api_key

# OpenAI API Key (Optional)
OPENAI_API_KEY=your_openai_api_key

# Supabase PostgreSQL Connection
SUPABASE_DB_URL=postgresql://postgres.<project_ref>:<password>@<region>.pooler.supabase.com:6543/postgres
```

> **Note**
>
> Gunakan **Supabase Connection Pooler Port 6543 (Session Mode)** agar koneksi dari Google Colab maupun server IPv4 berjalan stabil.

---

# 📦 Installation

Clone repository:

```bash
git clone https://github.com/yourusername/DRI_RAG_System.git
cd DRI_RAG_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Atur environment variables sesuai kebutuhan.

Jalankan notebook sesuai urutan:

1. `RAG_chunk.ipynb`
2. `RAG.ipynb`
3. `LLM_as_a_Reasoner.ipynb`

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👤 Author

**Farhan Ahmad Naufal**

**Aspiring AI/ML Developer | Computer Vision Enthusiast | Informatics Researcher**

- GitHub: **@farhanahmadn**