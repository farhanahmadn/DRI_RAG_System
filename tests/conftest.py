"""conftest.py — dimuat pytest paling awal (sebelum collection).

Paksa RETRIEVER=mock supaya suite offline (`pytest -q -m "not live"`) TIDAK PERNAH mencoba
konek Postgres sungguhan, terlepas dari default "asli" di app/api/dependencies.py::get_retriever
(env dibaca lazy di dalam fungsi itu, jadi harus di-set sebelum test pertama memanggilnya).

Juga kosongkan DATABASE_URL di env proses pytest ini (str kosong, BUKAN pop — python-dotenv's
load_dotenv() defaultnya tidak menimpa var yang sudah ADA di os.environ sekalipun kosong, jadi ini
mencegah .env mengisinya ulang belakangan). tests/test_retrieval.py (milik tim RAG) skip otomatis
berdasar `bool(os.getenv("DATABASE_URL"))` — sejak .env WAJIB mengisi DATABASE_URL (dipakai
RetrieverAsli produksi) & app/retrieval/db.py memanggil load_dotenv() saat modul lain di-import
lebih dulu (mis. app.api.main via test_api.py), guard skip itu bisa keliru jadi "DB tersedia" lalu
mencoba konek Postgres sungguhan & hang/timeout — bukan cuma gagal cepat seperti sebelum psycopg
terpasang. Untuk uji retrieval asli terhadap DB sungguhan, jalankan `pytest tests/test_retrieval.py
-v` langsung (lihat docstring file itu) dengan DATABASE_URL di-set manual di shell, bukan lewat
suite utama.
"""

import os

os.environ["RETRIEVER"] = "mock"
os.environ["DATABASE_URL"] = ""
