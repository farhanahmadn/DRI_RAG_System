"""Uji integrasi SEAM: RetrieverAsli (RAG teman) lewat jalur reasoning model L2 yang SEBENARNYA.

Membuktikan swap retrieval berfungsi di boundary reasoning (app/adapter.py -> app/reasoning/*)
tanpa mengubah app/api/dependencies.py permanen. Poin diambil dari fixture L2Assessment NYATA lewat
app.adapter.adaptasi() (BUKAN PoinKonteks buatan tangan) — supaya bentuk fakta{} dijamin sama
persis dengan yang benar-benar dihasilkan pipeline, bukan tebakan.

Bagian 1 (retrieval, tanpa Groq): ambil_chunks_pendukung per poin lewat RetrieverAsli — bukti seam
retrieval jalan. Bagian 2 (generate_poin penuh, butuh GROQ_API_KEY) HANYA jalan utk poin NON-aman
(intensitas MELAMPAUI_BATAS di fixture ini) — poin "aman" short-circuit ke template tanpa LLM/RAG,
jadi tidak relevan sbg bukti integrasi RAG+LLM.

CATATAN JUJUR (belum diverifikasi dari sisi saya): `dasar_hukum` di fixture (dokumen="RDTR Sleman",
pasal="Matriks ITBX") BELUM dikonfirmasi benar-benar ter-ingest di DB RetriverAsli utk wilayah yang
dipakai — get_by_reference bisa saja kosong kalau dokumen itu belum di-ingest, itu bukan bug script
ini. Fallback search (poin.kategori/kata kunci pendek) tetap jalan sbg jaring pengaman.

Jalankan:  python -m scripts.test_integrasi
Prasyarat: DB terisi (ingest wilayah terkait) + model bge-m3/reranker (cache) + .env DATABASE_URL.
"""

import json
import os
from pathlib import Path

from app.adapter import adaptasi
from app.reasoning.generator import ambil_chunks_pendukung, apakah_aman, generate_poin
from app.retrieval.base import Retriever
from app.retrieval.retriever import RetrieverAsli
from app.schemas import L2Assessment

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "l2_sample_amplop_6191.json"


def _muat_assessment() -> L2Assessment:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def main() -> None:
    rt = RetrieverAsli(default_wilayah=os.getenv("DEMO_WILAYAH", "Sleman Timur"))
    print("RetrieverAsli patuh Protocol Retriever:", isinstance(rt, Retriever))

    assessment = _muat_assessment()
    hasil_adaptasi = adaptasi(assessment)
    print(f"\nFixture: {FIXTURE_PATH.name} -> {len(hasil_adaptasi.poin)} poin (dari adaptasi(), bukan buatan tangan)")

    print("\n=== BAGIAN 1: boundary retrieval reasoning (ambil_chunks_pendukung) — tanpa Groq ===")
    for poin in hasil_adaptasi.poin:
        chunks = ambil_chunks_pendukung(poin, rt)
        jalur = "get_by_reference" if poin.dasar_hukum else "search(fallback)"
        aman = apakah_aman(poin)
        print(f"\n[{poin.poin_id}] kategori={poin.kategori!r}  status={poin.status!r}  aman={aman}  jalur={jalur}")
        if not chunks:
            print("   (0 chunk ditemukan — cek apakah dokumen/pasal terkait sudah ter-ingest di DB.)")
        for c in chunks:
            print(f"   - {c.id} (Pasal {c.pasal}, hal {c.halaman}) [{c.dokumen[:40]}...]")
            print(f"     {c.teks[:90].replace(chr(10), ' ')}...")

    print("\n=== BAGIAN 2: generate_poin penuh (reasoning + sitasi dari RAG asli) — poin NON-aman saja ===")
    if not os.getenv("GROQ_API_KEY"):
        print("(GROQ_API_KEY tak diset — dilewati. Set key utk uji reasoning+sitasi end-to-end.)")
        return

    poin_non_aman = [p for p in hasil_adaptasi.poin if not apakah_aman(p)]
    if not poin_non_aman:
        print("(Semua poin di fixture ini 'aman' — tidak ada yg lewat jalur LLM utk didemonstrasikan.)")
        return

    for poin in poin_non_aman:
        hasil = generate_poin(poin, rt, assessment.meta)
        print(f"\n[{poin.poin_id}] status: {hasil.status}  low_confidence: {hasil.low_confidence}")
        print(f"reasoning_pendek: {hasil.reasoning_pendek}")
        print("sitasi (dari RAG asli):")
        for s in hasil.sitasi:
            print(f"   - {s.citation_id} | {s.dokumen[:35]}... Pasal {s.pasal} hal {s.halaman} | verified={s.terverifikasi}")
            print(f"     kutipan: {s.kutipan[:100]}...")
        print(f"rekomendasi: [{hasil.rekomendasi.tipe}] {hasil.rekomendasi.saran[:120]}...")


if __name__ == "__main__":
    main()
