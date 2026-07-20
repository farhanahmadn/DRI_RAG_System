import os

import pytest
from dotenv import load_dotenv

from app.reasoning import llm_client as llm_client_module
from app.reasoning.generator import generate_poin
from app.reasoning.templates import template_aman
from app.retrieval.mock import MockRetriever
from app.schemas import FaktaSpasial, IndikatorJejak

load_dotenv()


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="LP2B-01",
        kategori="Lokasional LP2B",
        bobot=20.0,
        skor=0.0,
        kontribusi=0.0,
        nilai_input="tidak_dalam_lp2b",
        ambang="tidak_dalam_lp2b",
        operator="==",
        formula="in_lp2b == True",
        zona="LP2B",
        referensi_hukum=["UU No. 41 Tahun 2009 Pasal 44"],
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


class _RetrieverYangMelarangDipanggil:
    def search(self, query, filters, top_k=5):
        raise AssertionError("search() tidak boleh dipanggil untuk indikator skor 0")

    def get_by_reference(self, referensi):
        raise AssertionError("get_by_reference() tidak boleh dipanggil untuk indikator skor 0")

    def get_parent(self, chunk_id):
        raise AssertionError("get_parent() tidak boleh dipanggil untuk indikator skor 0")


def test_generate_poin_skor_nol_pakai_template_tanpa_retrieval():
    indikator = _indikator(skor=0.0, kontribusi=0.0)
    hasil = generate_poin(indikator, _RetrieverYangMelarangDipanggil())
    assert hasil == template_aman(indikator)


def test_generate_poin_kdb_prompt_menyuntik_status_melebihi(monkeypatch):
    prompt_tertangkap = {}

    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        prompt_tertangkap["prompt"] = prompt
        return {
            "reasoning_pendek": "KDB melebihi batas.",
            "reasoning_panjang": "Nilai KDB aktual melebihi batas maksimum zona ini.",
            "sitasi": [],
            "saran": "Kurangi luas lantai dasar bangunan.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    indikator = IndikatorJejak(
        poin_id="KDB-01",
        kategori="KDB",
        bobot=20.0,
        skor=20.0,
        kontribusi=20.0,
        nilai_input=90.0,
        ambang=80.0,
        operator="<=",
        formula="kdb_aktual <= kdb_maks",
        zona="C-1",
        referensi_hukum=["RDTR Pasal 1 Ayat 107", "RDTR Lampiran VI"],
    )

    hasil = generate_poin(indikator, MockRetriever())

    assert "STATUS: MELEBIHI" in prompt_tertangkap["prompt"]
    assert hasil.rekomendasi.tipe == "numerik"
    assert hasil.rekomendasi.target == 80.0


def test_generate_poin_kegiatan_x_prompt_menyuntik_klasifikasi_dan_daftar_diizinkan(monkeypatch):
    prompt_tertangkap = {}

    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        prompt_tertangkap["prompt"] = prompt
        return {
            "reasoning_pendek": "Kegiatan ini tidak diizinkan di zona ini.",
            "reasoning_panjang": "Industri besar/pabrik tergolong X (dilarang) pada zona C-1.",
            "sitasi": [],
            "saran": "Pertimbangkan kegiatan alternatif yang diizinkan di zona ini.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    indikator = IndikatorJejak(
        poin_id="KEG-04",
        kategori="Kegiatan: Industri Besar/Pabrik",
        bobot=20.0,
        skor=60.0,
        kontribusi=60.0,
        nilai_input="X",
        ambang="I",
        operator="==",
        formula="",
        zona="C-1",
        referensi_hukum=["RDTR Pasal 1 Ayat 108", "RDTR Lampiran V"],
    )

    hasil = generate_poin(indikator, MockRetriever())

    assert "KLASIFIKASI: X" in prompt_tertangkap["prompt"]
    assert "Rumah toko (ruko) skala kecil" in prompt_tertangkap["prompt"]
    assert hasil.rekomendasi.tipe == "kegiatan"
    assert hasil.rekomendasi.target is None


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_generate_poin_lp2b_skor_tinggi_grounded():
    indikator = _indikator(
        skor=100.0,
        kontribusi=20.0,
        nilai_input="dalam_lp2b",
        fakta_spasial=FaktaSpasial(in_lp2b=True, banjir=False, resapan=False),
    )
    retriever = MockRetriever()

    poin = generate_poin(indikator, retriever)

    assert poin.status == "Tidak Aman"
    assert poin.kontribusi == indikator.kontribusi
    assert poin.rekomendasi.tipe == "lokasional"
    assert poin.rekomendasi.target is None

    chunk_ids_diberikan = {c.id for c in retriever.get_by_reference(indikator.referensi_hukum)}
    assert len(poin.sitasi) > 0
    for sitasi in poin.sitasi:
        assert sitasi.citation_id in chunk_ids_diberikan
        assert sitasi.terverifikasi is True
