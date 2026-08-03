# Integrasi RAG (Retriever) — Kontrak SEAM

Dokumen ini untuk yang mengerjakan RAG asli (parsing, chunking, embedding, retrieval nyata via
pgvector/hybrid search). Kontrak di bawah ada di `app/retrieval/base.py` — file itu yang
"jarang diubah" (CLAUDE.md § SEAM), disepakati bersama, dan HANYA file itu yang perlu diikuti
persis. Kode reasoning (`app/reasoning/*`) memanggil lewat `Protocol` ini, tidak pernah tahu
implementasi konkretnya.

## Protocol `Retriever` (persis dari `app/retrieval/base.py`)

```python
from datetime import date
from typing import Literal, Protocol, runtime_checkable
from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    level: Literal["pasal", "ayat", "tabel"]
    parent_id: str | None = None
    teks: str
    dokumen: str
    pasal: str | None = None
    ayat: str | None = None
    halaman: int | None = None
    skor: float = 0.0
    tanggal_berlaku: date | None = None
    istilah_kode: str | None = None
    zona: str | None = None
    jenis: str | None = None
    tanggal_dicabut: date | None = None


class RetrievalFilters(BaseModel):
    as_of: date | None = None
    zona: str | None = None
    dokumen: str | None = None
    jenis: str | None = None


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, filters: RetrievalFilters, top_k: int = 5) -> list[Chunk]: ...
    def get_by_reference(self, referensi: list[str]) -> list[Chunk]: ...
    def get_parent(self, chunk_id: str) -> Chunk | None: ...
```

Implementasi RAG asli tinggal punya class yang punya ketiga method itu dengan signature persis —
tidak perlu inherit dari apa pun (Python `Protocol` = *structural typing*, cukup cocok bentuknya).

## Field `Chunk` — arti tiap field

| Field | Arti | Wajib? |
|---|---|---|
| `id` | ID unik chunk (dipakai sebagai `citation_id` di output akhir) | Wajib |
| `level` | `"pasal"` (induk), `"ayat"` (anak dari pasal), atau `"tabel"` (matriks/lampiran, bukan pasal bernomor) | Wajib |
| `parent_id` | ID chunk induk kalau `level="ayat"` (pola parent-child, CLAUDE.md § Chunking); `None` kalau chunk ini sendiri induk/tabel | — |
| `teks` | Isi teks chunk — INI yang dikutip LLM sebagai `kutipan` di sitasi output | Wajib |
| `dokumen` | Nama dokumen sumber (mis. `"UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan"`) | Wajib |
| `pasal` | Nomor pasal (string, mis. `"44"`) — `None` kalau chunk tabel/lampiran tanpa nomor pasal | — |
| `ayat` | Nomor ayat (string, mis. `"1"`) — hanya diisi kalau `level="ayat"` | — |
| `halaman` | Nomor halaman di dokumen sumber | — |
| `skor` | Skor relevansi RETRIEVAL (bukan skor risiko bangunan!) — reasoning tidak memakai nilai ini utk keputusan, cuma metadata | — |
| `tanggal_berlaku` | Tanggal mulai berlaku — dipakai `RetrievalFilters.as_of` utk filter dokumen kedaluwarsa | — |
| `istilah_kode` | Kode/istilah non-pasal (mis. `"Lampiran VI"`) — ditampilkan di prompt LLM sebagai penanda lokasi kutipan | — |
| `zona` | Kode zona kalau chunk ini spesifik satu zona (mis. `"C-1"`) | — |
| `jenis` | Jenis dokumen (mis. `"UU"`, `"RDTR"`, `"Permen"`, `"RTRW"`, `"Metodologi"` — **bebas string**, tidak di-enum-kan) | — |
| `tanggal_dicabut` | Tanggal dicabut/tidak berlaku lagi — dipakai filter `as_of` juga | — |

## Contoh `Chunk` NYATA (persis dari `app/retrieval/mock.py`, bukan karangan)

**Chunk level `"pasal"` (induk):**
```python
Chunk(
    id="uu41-2009-p44",
    level="pasal",
    parent_id=None,
    teks="Pasal 44: Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilindungi dan "
         "dilarang dialihfungsikan, kecuali untuk kepentingan umum dengan syarat tertentu "
         "(kajian kelayakan strategis, penyusunan rencana alih fungsi, dan penyediaan lahan "
         "pengganti).",
    dokumen="UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan",
    pasal="44",
    halaman=21,
    tanggal_berlaku=date(2009, 8, 14),
    zona="LP2B",
    jenis="UU",
)
```

**Chunk level `"ayat"` (anak dari pasal di atas, pola parent-child):**
```python
Chunk(
    id="uu41-2009-p44-a1",
    level="ayat",
    parent_id="uu41-2009-p44",   # merujuk id chunk pasal induk
    teks="(1) Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilarang dialihfungsikan.",
    dokumen="UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan",
    pasal="44",
    ayat="1",
    halaman=21,
    zona="LP2B",
    jenis="UU",
)
```

**Chunk level `"tabel"` (matriks/lampiran, TANPA nomor pasal — CLAUDE.md § Data: "matriks/tabel ke
tabel terstruktur, bukan embedding"):**
```python
Chunk(
    id="rdtr-lampiran-vi-c1",
    level="tabel",
    parent_id=None,
    teks="Lampiran VI: Matriks Intensitas Pemanfaatan Ruang. Zona C-1 (Perdagangan dan Jasa): "
         "KDB maksimum 80%, KLB maksimum 2.4, KDH minimum 20%.",
    dokumen="Peraturan Daerah Kabupaten Sleman tentang Rencana Detail Tata Ruang (RDTR)",
    pasal=None,
    istilah_kode="Lampiran VI",
    halaman=112,
    zona="C-1",
    jenis="RDTR",
)
```

## Semantik tiap method

### `search(query: str, filters: RetrievalFilters, top_k: int = 5) -> list[Chunk]`
Pencarian bebas teks (dipanggil reasoning sebagai **fallback** kalau `get_by_reference` kosong —
lihat `app/reasoning/generator.py::ambil_chunks_pendukung`). Query biasanya `kategori` indikator
(mis. `"KDB"`, `"Lokasional Banjir"`). Implementasi asli: embed query (bge-m3) → similarity
pgvector (+ BM25 hybrid) → rerank (bge-reranker-v2-m3) → terapkan `filters` → potong `top_k`.
`MockRetriever` (dev) memakai keyword-overlap sederhana sebagai stand-in — TIDAK perlu ditiru
persis, cuma perlu mengembalikan hasil paling relevan diurutkan `skor` menurun.

### `get_by_reference(referensi: list[str]) -> list[Chunk]`
Lookup LANGSUNG dari string rujukan hukum (mis. `"UU No. 41 Tahun 2009 Pasal 44"`,
`"RDTR Lampiran VI"`) — ini `IndikatorJejak.referensi_hukum` yang dikirim back-end (lihat
`docs/INTEGRASI_BACKEND.md`). Ini jalur UTAMA (dicoba lebih dulu sebelum `search`) karena paling
presisi — back-end sudah tahu pasal mana yang relevan untuk indikator itu. Idealnya: parse nomor
dokumen + pasal dari string, lookup persis; kalau tak ketemu match presisi, boleh fallback ke
pencarian dokumen-level (seperti yang dilakukan `MockRetriever` — lihat kode di sana untuk contoh
pendekatan token-overlap sederhana).

### `get_parent(chunk_id: str) -> Chunk | None`
Ambil chunk INDUK dari sebuah chunk anak (pola "small-to-big": retrieval di level ayat yang presisi,
tapi kadang reasoning butuh konteks pasal penuh). `None` kalau `chunk_id` tidak ditemukan ATAU
chunk itu sendiri sudah level induk (`parent_id=None`).

## Cara Swap `MockRetriever` → RAG Asli

Reasoning **TIDAK PERNAH** mengimpor `MockRetriever` secara langsung — satu-satunya titik
sambung ada di `app/api/dependencies.py`:

```python
# app/api/dependencies.py — SEBELUM (sekarang)
from app.retrieval.mock import MockRetriever

@lru_cache
def get_retriever() -> Retriever:
    return MockRetriever()
```

```python
# app/api/dependencies.py — SESUDAH (ganti baris ini saja)
from app.retrieval.retriever import RetrieverAsli   # nama kelas implementasi Anda

@lru_cache
def get_retriever() -> Retriever:
    return RetrieverAsli(...)   # koneksi DB/config Anda
```

Itu SATU-SATUNYA perubahan yang dibutuhkan di sisi reasoning. Tidak ada file lain di
`app/reasoning/*` yang perlu disentuh — semuanya sudah menerima `retriever: Retriever` sebagai
parameter/dependency, bukan hardcode. Implementasi asli boleh ditaruh di
`app/retrieval/retriever.py` (nama file yang sudah disepakati di CLAUDE.md § SEAM).

## Yang BUKAN kontrak ini

- `app/ingest/` (parsing PDF, chunking, embedding, isi DB) — sepenuhnya wilayah RAG, tidak
  disentuh reasoning sama sekali.
- Skor retrieval (`Chunk.skor`) tidak dipakai reasoning untuk keputusan apa pun — murni informasi
  ranking internal Anda.
