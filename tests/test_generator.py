import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.adapter import adaptasi
from app.reasoning import llm_client as llm_client_module
from app.reasoning.generator import _apakah_aman, generate_poin
from app.reasoning.templates import template_aman
from app.retrieval.mock import MockRetriever
from app.schemas import L2Assessment, PoinKonteks

load_dotenv()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _muat_assessment(nama_file: str) -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / nama_file).read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _poin(**overrides) -> PoinKonteks:
    defaults = dict(
        poin_id="itbx",
        kategori="Klasifikasi Kegiatan (ITBX)",
        tipe_rekomendasi="kategorikal",
        status="I",
        fakta={"lolos": True, "reason": "Lolos karena kegiatan Diizinkan (I)"},
        dasar_hukum=[],
    )
    defaults.update(overrides)
    return PoinKonteks(**defaults)


class _RetrieverYangMelarangDipanggil:
    def search(self, query, filters, top_k=5):
        raise AssertionError("search() tidak boleh dipanggil untuk poin yang aman")

    def get_by_reference(self, referensi):
        raise AssertionError("get_by_reference() tidak boleh dipanggil untuk poin yang aman")

    def get_parent(self, chunk_id):
        raise AssertionError("get_parent() tidak boleh dipanggil untuk poin yang aman")


class TestApakahAman:
    def test_itbx_aman(self):
        assert _apakah_aman(_poin(status="I", fakta={"lolos": True, "reason": "x"})) is True

    def test_itbx_tidak_aman_karena_bukan_i(self):
        assert _apakah_aman(_poin(status="B", fakta={"lolos": True, "reason": "x"})) is False

    def test_itbx_tidak_aman_karena_fallback(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        assert _apakah_aman(poin) is False

    def test_intensitas_aman_tanpa_target(self):
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={"target": {}})
        assert _apakah_aman(poin) is True

    def test_intensitas_tidak_aman_dengan_target(self):
        poin = _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0}}},
        )
        assert _apakah_aman(poin) is False

    def test_dampak_aman_tanpa_mitigasi(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}},
        )
        assert _apakah_aman(poin) is True

    def test_dampak_tidak_aman_dengan_mitigasi(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={"mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB"]}},
        )
        assert _apakah_aman(poin) is False


def test_generate_poin_aman_pakai_template_tanpa_retrieval_atau_llm(monkeypatch):
    dipanggil = {"llm": False}
    monkeypatch.setattr(llm_client_module, "generate", lambda *a, **k: dipanggil.__setitem__("llm", True))

    poin = _poin(status="I", fakta={"lolos": True, "reason": "x"})
    hasil = generate_poin(poin, _RetrieverYangMelarangDipanggil())

    assert hasil == template_aman(poin)
    assert dipanggil["llm"] is False


def test_generate_poin_merakit_dari_respons_llm_palsu(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "KDB melampaui batas maksimum.",
            "reasoning_panjang": "Usulan KDB melampaui ambang yang berlaku di zona ini.",
            "sitasi": [],
            "saran": "Kurangi proporsi luas bangunan terhadap luas lahan.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="MELAMPAUI_BATAS",
        fakta={
            "parameter": {"kdb": {"usulan": 70, "ambang_maks": 60, "ambang_min": None, "memenuhi": False, "satuan": "persen"}},
            "target": {"kdb": {"target_kdb": 60.0, "selisih": 10.0, "footprint_maks_m2": 510.0}},
        },
    )

    hasil = generate_poin(poin, MockRetriever())

    assert hasil.status == "MELAMPAUI_BATAS"
    assert hasil.rekomendasi.tipe == "numerik"
    assert hasil.rekomendasi.target == 60.0  # dari calculator, BUKAN dari LLM
    assert hasil.low_confidence is False


def test_generate_poin_sitasi_anchor_diprioritaskan(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
            "saran": "x",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    from app.schemas import DasarHukum

    poin = _poin(
        status="B",
        fakta={"lolos": True, "reason": "x"},
        dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
    )
    hasil = generate_poin(poin, MockRetriever())

    assert len(hasil.sitasi) == 1
    assert hasil.sitasi[0].citation_id == "anchor-0"
    assert hasil.sitasi[0].dokumen == "RDTR Sleman"
    assert hasil.sitasi[0].terverifikasi is True


def test_generate_poin_citation_id_halusinasi_dibuang(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [{"citation_id": "pasal-karangan-99", "kutipan": "tidak ada di daftar"}],
            "saran": "x",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
    hasil = generate_poin(poin, MockRetriever())

    assert hasil.sitasi == []


# --- Live (@pytest.mark.live): panggilan LLM sungguhan, butuh GROQ_API_KEY, tak jalan di suite rutin.

_ALASAN_SKIP_LIVE = "GROQ_API_KEY tidak diset — skip smoke test panggilan LLM nyata."


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason=_ALASAN_SKIP_LIVE)
def test_generate_poin_intensitas_melanggar_grounded_live():
    assessment = _muat_assessment("l2_sample_lolos_bersyarat.json")
    hasil_adaptasi = adaptasi(assessment)
    poin_intensitas = next(p for p in hasil_adaptasi.poin if p.poin_id == "intensitas")

    poin = generate_poin(poin_intensitas, MockRetriever(), assessment.meta)

    assert poin.status == "MELAMPAUI_BATAS"
    assert poin.rekomendasi.tipe == "numerik"
    assert poin.rekomendasi.target == 60.0
    assert poin.low_confidence is False
    assert poin.reasoning_pendek.strip() != ""


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason=_ALASAN_SKIP_LIVE)
def test_generate_poin_itbx_bersyarat_grounded_live():
    # Sintetis (bukan dari fixture nyata — 2 fixture kita berdua status "I") sekadar utk memaksa
    # jalur LLM (status "B" bukan "aman") & membuktikan generate_poin end-to-end sungguhan.
    poin = _poin(
        status="B",
        fakta={
            "lolos": True,
            "reason": "Diizinkan bersyarat sesuai Matriks ITBX zona perumahan",
            "keterangan_ketentuan": ["Wajib menyediakan lahan parkir yang memadai untuk pengunjung."],
        },
    )

    hasil = generate_poin(poin, MockRetriever())

    assert hasil.status == "B"
    assert hasil.low_confidence is False
    assert hasil.reasoning_pendek.strip() != ""
