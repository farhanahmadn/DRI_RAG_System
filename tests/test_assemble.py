import os

import pytest
from dotenv import load_dotenv

from app.reasoning import assemble as assemble_module
from app.reasoning.assemble import _LEVEL_BELUM_TERSEDIA, jalankan_precheck
from app.retrieval.mock import MockRetriever
from app.schemas import (
    FaktaSpasial,
    IndikatorJejak,
    JejakAturanRequest,
    PoinOutput,
    RekomendasiOutput,
)

load_dotenv()


def _indikator_lp2b(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="LP2B-01",
        kategori="Lokasional LP2B",
        bobot=20.0,
        skor=100.0,
        kontribusi=20.0,
        nilai_input="dalam_lp2b",
        ambang="tidak_dalam_lp2b",
        operator="==",
        formula="in_lp2b == True",
        zona="LP2B",
        referensi_hukum=["UU No. 41 Tahun 2009 Pasal 44"],
        fakta_spasial=FaktaSpasial(in_lp2b=True, banjir=False, resapan=False),
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def _indikator_aman(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="KDB-01",
        kategori="KDB",
        bobot=10.0,
        skor=0.0,
        kontribusi=0.0,
        nilai_input=0.4,
        ambang=0.6,
        operator="<=",
        formula="kdb_aktual <= kdb_maks",
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def _request(**overrides) -> JejakAturanRequest:
    defaults = dict(
        skor_total=20.0,
        level=None,
        zona="LP2B",
        indikator=[_indikator_lp2b(), _indikator_aman()],
    )
    defaults.update(overrides)
    return JejakAturanRequest(**defaults)


def _poin_berisiko_lokasional(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="LP2B-01",
        kategori="Lokasional LP2B",
        status="Tidak Aman",
        kontribusi=20.0,
        reasoning_pendek="pendek",
        reasoning_panjang="panjang",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="lokasional", target=None, saran="Ajukan kajian kelayakan strategis.", disclaimer=None),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


def _poin_aman(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="KDB-01",
        kategori="KDB",
        status="Aman",
        kontribusi=0.0,
        reasoning_pendek="aman",
        reasoning_panjang="aman panjang",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="numerik", target=None, saran="Tidak diperlukan tindakan khusus.", disclaimer=None),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


def _patch_guardrail(monkeypatch, poin_by_id: dict[str, PoinOutput]):
    dipanggil = []

    def _stub(indikator, retriever, *, max_retry=2):
        dipanggil.append(indikator.poin_id)
        return poin_by_id[indikator.poin_id]

    monkeypatch.setattr(assemble_module, "generate_poin_dengan_guardrail", _stub)
    return dipanggil


def _patch_log(monkeypatch):
    panggilan = []
    monkeypatch.setattr(
        assemble_module, "log_precheck", lambda request, response: panggilan.append((request, response))
    )
    return panggilan


# --- ringkasan.skor_total & level -------------------------------------------------------


def test_skor_total_pass_through_tidak_dihitung_ulang(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(skor_total=42.0, level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert output.ringkasan.skor_total == 42.0


def test_level_none_jadi_sentinel_todo(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(level=None)
    output = jalankan_precheck(request, MockRetriever())

    assert output.ringkasan.level == _LEVEL_BELUM_TERSEDIA


def test_level_dari_backend_dipakai_apa_adanya(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert output.ringkasan.level == "Tinggi"


# --- kalimat ringkasan -------------------------------------------------------------------


def test_kalimat_tidak_ada_berisiko(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_aman(poin_id="LP2B-01", kategori="Lokasional LP2B"), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(level="Rendah")
    output = jalankan_precheck(request, MockRetriever())

    assert "tidak ditemukan" in output.ringkasan.kalimat.lower()


def test_kalimat_ada_berisiko_sebut_skor_dan_kategori(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(skor_total=20.0, level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert "20.0" in output.ringkasan.kalimat
    assert "Lokasional LP2B" in output.ringkasan.kalimat


# --- kesimpulan.langkah_berdampak & catatan_lokasi ----------------------------------------


def test_langkah_berdampak_hanya_dari_poin_berisiko(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert len(output.kesimpulan.langkah_berdampak) == 1
    assert "Ajukan kajian kelayakan strategis." in output.kesimpulan.langkah_berdampak[0]
    assert not any("Tidak diperlukan tindakan khusus" in s for s in output.kesimpulan.langkah_berdampak)


def test_catatan_lokasi_ada_jika_ada_lokasional_berisiko(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    _patch_log(monkeypatch)

    request = _request(level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert output.kesimpulan.catatan_lokasi is not None
    assert "Lokasional LP2B" in output.kesimpulan.catatan_lokasi


def test_catatan_lokasi_none_jika_tidak_ada_lokasional_berisiko(monkeypatch):
    poin_kdb_berisiko = _poin_aman(poin_id="KDB-01", status="Tidak Aman", kontribusi=5.0)
    poin_kdb_berisiko = poin_kdb_berisiko.model_copy(
        update={"rekomendasi": RekomendasiOutput(tipe="numerik", target=600.0, saran="Kurangi footprint bangunan.", disclaimer=None)}
    )
    _patch_guardrail(
        monkeypatch,
        {
            "LP2B-01": _poin_aman(poin_id="LP2B-01", kategori="Lokasional LP2B"),
            "KDB-01": poin_kdb_berisiko,
        },
    )
    _patch_log(monkeypatch)

    request = _request(level="Sedang")
    output = jalankan_precheck(request, MockRetriever())

    assert output.kesimpulan.catatan_lokasi is None


# --- logging dipanggil ---------------------------------------------------------------------


def test_log_precheck_dipanggil_sekali_dengan_data_benar(monkeypatch):
    _patch_guardrail(monkeypatch, {"LP2B-01": _poin_berisiko_lokasional(), "KDB-01": _poin_aman()})
    panggilan = _patch_log(monkeypatch)

    request = _request(level="Tinggi")
    output = jalankan_precheck(request, MockRetriever())

    assert len(panggilan) == 1
    logged_request, logged_response = panggilan[0]
    assert logged_request == request
    assert logged_response == output


# --- live integration (1x, skip kalau tanpa GROQ_API_KEY) ----------------------------------


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_jalankan_precheck_live_lp2b_dan_aman():
    request = _request(skor_total=20.0, level="Tinggi")
    retriever = MockRetriever()

    output = jalankan_precheck(request, retriever)

    assert output.ringkasan.skor_total == request.skor_total

    poin_by_id = {p.poin_id: p for p in output.poin}
    poin_lp2b = poin_by_id["LP2B-01"]
    poin_aman = poin_by_id["KDB-01"]

    assert poin_lp2b.status == "Tidak Aman"
    chunk_ids = {c.id for c in retriever.get_by_reference(["UU No. 41 Tahun 2009 Pasal 44"])}
    for s in poin_lp2b.sitasi:
        assert s.citation_id in chunk_ids

    assert poin_aman.status == "Aman"
    assert poin_aman.low_confidence is False

    assert output.kesimpulan.catatan_lokasi is not None
