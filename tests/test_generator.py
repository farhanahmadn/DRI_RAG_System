import os

import pytest
from dotenv import load_dotenv

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
