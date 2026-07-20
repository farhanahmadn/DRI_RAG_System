import pytest

from app.reasoning import guardrail as guardrail_module
from app.reasoning.guardrail import (
    generate_poin_dengan_guardrail,
    perbaiki_poin,
)
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.schemas import IndikatorJejak, PoinOutput, RekomendasiOutput, SitasiOutput


def _indikator(**overrides) -> IndikatorJejak:
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
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="uu41-2009-p44",
        level="pasal",
        teks="Pasal 44: dilarang dialihfungsikan.",
        dokumen="UU No. 41 Tahun 2009 tentang LP2B",
        pasal="44",
        halaman=21,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def _poin(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="LP2B-01",
        kategori="Lokasional LP2B",
        status="Aman",
        kontribusi=0.0,
        reasoning_pendek="teks",
        reasoning_panjang="teks panjang",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="lokasional", target=None, saran="saran", disclaimer=None),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


# --- perbaiki_poin: konsistensi numerik ---------------------------------------------------


def test_target_numerik_meleset_ditimpa():
    indikator = _indikator(
        kategori="KDB",
        skor=5.0,
        kontribusi=5.0,
        nilai_input=650.0,
        ambang=0.6,
        operator="<=",
        luas_lahan=1000.0,
    )
    poin = _poin(
        kategori="KDB",
        rekomendasi=RekomendasiOutput(tipe="numerik", target=999.0, saran="saran", disclaimer=None),
    )

    hasil, masalah = perbaiki_poin(poin, indikator, [])

    assert hasil.rekomendasi.target == 600.0
    assert masalah == []


def test_target_non_numerik_ditimpa_jadi_none():
    indikator = _indikator()  # LP2B, lokasional
    poin = _poin(rekomendasi=RekomendasiOutput(tipe="lokasional", target=777.0, saran="saran", disclaimer=None))

    hasil, _ = perbaiki_poin(poin, indikator, [])

    assert hasil.rekomendasi.target is None


# --- perbaiki_poin: sanity sitasi -----------------------------------------------------------


def test_chunks_kosong_paksa_sitasi_kosong():
    indikator = _indikator()
    poin = _poin(
        sitasi=[
            SitasiOutput(
                citation_id="uu41-2009-p44",
                dokumen="UU No. 41 Tahun 2009",
                pasal="44",
                halaman=21,
                kutipan="kutipan",
                terverifikasi=True,
            )
        ]
    )

    hasil, masalah = perbaiki_poin(poin, indikator, [])

    assert hasil.sitasi == []
    assert masalah == []  # chunks kosong bukan indikasi halusinasi, tidak butuh regenerasi


def test_sitasi_fiktif_dibuang():
    indikator = _indikator()
    chunk = _chunk()
    poin = _poin(
        sitasi=[
            SitasiOutput(
                citation_id="fiktif-999",
                dokumen="karangan",
                pasal="99",
                halaman=1,
                kutipan="ngarang",
                terverifikasi=True,
            )
        ]
    )

    hasil, masalah = perbaiki_poin(poin, indikator, [chunk])

    assert hasil.sitasi == []
    assert any("grounded" in m or "sitasi" in m.lower() for m in masalah)


def test_sitasi_cocok_dibangun_ulang_dari_chunk_asli():
    indikator = _indikator()
    chunk = _chunk()
    poin = _poin(
        sitasi=[
            SitasiOutput(
                citation_id=chunk.id,
                dokumen="salah ketik dokumen",
                pasal="salah",
                halaman=0,
                kutipan="kutipan asli dari LLM",
                terverifikasi=False,
            )
        ]
    )

    hasil, masalah = perbaiki_poin(poin, indikator, [chunk])

    assert len(hasil.sitasi) == 1
    s = hasil.sitasi[0]
    assert s.dokumen == chunk.dokumen
    assert s.pasal == chunk.pasal
    assert s.halaman == chunk.halaman
    assert s.kutipan == "kutipan asli dari LLM"
    assert s.terverifikasi is True
    assert masalah == []


def test_terverifikasi_tidak_pernah_true_tanpa_chunk_pendukung():
    indikator = _indikator()
    chunk = _chunk()
    poin = _poin(
        sitasi=[
            SitasiOutput(
                citation_id="fiktif-1",
                dokumen="karangan",
                pasal="1",
                halaman=1,
                kutipan="x",
                terverifikasi=True,
            ),
            SitasiOutput(
                citation_id=chunk.id,
                dokumen=chunk.dokumen,
                pasal=chunk.pasal,
                halaman=chunk.halaman,
                kutipan="kutipan valid",
                terverifikasi=True,
            ),
        ]
    )

    hasil, _ = perbaiki_poin(poin, indikator, [chunk])

    assert len(hasil.sitasi) == 1
    assert all(s.terverifikasi for s in hasil.sitasi)
    assert hasil.sitasi[0].citation_id == chunk.id


# --- perbaiki_poin: status & kontribusi dari jejak ------------------------------------------


def test_status_dan_kontribusi_ditimpa_dari_jejak():
    indikator = _indikator(skor=100.0, kontribusi=20.0)
    poin = _poin(status="SALAH", kontribusi=999.0)

    hasil, _ = perbaiki_poin(poin, indikator, [])

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 20.0


# --- perbaiki_poin: deteksi masalah butuh regenerasi ----------------------------------------


def test_poin_valid_tidak_ada_masalah():
    indikator = _indikator(skor=100.0, kontribusi=20.0)
    chunk = _chunk()
    poin = _poin(
        status="Tidak Aman",
        kontribusi=20.0,
        sitasi=[
            SitasiOutput(
                citation_id=chunk.id,
                dokumen=chunk.dokumen,
                pasal=chunk.pasal,
                halaman=chunk.halaman,
                kutipan="kutipan",
                terverifikasi=True,
            )
        ],
    )

    _, masalah = perbaiki_poin(poin, indikator, [chunk])
    assert masalah == []


def test_reasoning_kosong_terdeteksi_sebagai_masalah():
    indikator = _indikator()
    poin = _poin(reasoning_pendek="", reasoning_panjang="")

    _, masalah = perbaiki_poin(poin, indikator, [])
    assert len(masalah) > 0


def test_saran_kosong_terdeteksi_sebagai_masalah():
    indikator = _indikator()
    poin = _poin(rekomendasi=RekomendasiOutput(tipe="lokasional", target=None, saran="   ", disclaimer=None))

    _, masalah = perbaiki_poin(poin, indikator, [])
    assert len(masalah) > 0


# --- generate_poin_dengan_guardrail: retry & fallback (monkeypatch, tanpa Groq) -------------


def test_fallback_low_confidence_saat_retry_habis(monkeypatch):
    panggilan = []

    def _stub_selalu_gagal(indikator, retriever, *, catatan_perbaikan=None, temperature=0.0):
        panggilan.append((catatan_perbaikan, temperature))
        return _poin(reasoning_pendek="", reasoning_panjang="")

    monkeypatch.setattr(guardrail_module, "generate_poin", _stub_selalu_gagal)

    indikator = _indikator(referensi_hukum=[])
    hasil = generate_poin_dengan_guardrail(indikator, retriever=MockRetriever(), max_retry=2)

    assert hasil == template_low_confidence(indikator)
    assert hasil.low_confidence is True
    assert len(panggilan) == 3  # max_retry + 1


def test_retry_berhasil_di_percobaan_kedua(monkeypatch):
    panggilan = {"n": 0}

    def _stub_gagal_lalu_sukses(indikator, retriever, *, catatan_perbaikan=None, temperature=0.0):
        panggilan["n"] += 1
        if panggilan["n"] == 1:
            return _poin(reasoning_pendek="", reasoning_panjang="")
        return _poin(status="Tidak Aman", kontribusi=indikator.kontribusi)

    monkeypatch.setattr(guardrail_module, "generate_poin", _stub_gagal_lalu_sukses)

    indikator = _indikator(referensi_hukum=[])
    hasil = generate_poin_dengan_guardrail(indikator, retriever=MockRetriever(), max_retry=2)

    assert hasil.low_confidence is False
    assert hasil.reasoning_pendek == "teks"
    assert panggilan["n"] == 2


def test_exception_dihitung_gagal_bukan_crash(monkeypatch):
    panggilan = {"n": 0}

    def _stub_exception_lalu_sukses(indikator, retriever, *, catatan_perbaikan=None, temperature=0.0):
        panggilan["n"] += 1
        if panggilan["n"] == 1:
            raise RuntimeError("simulasi gagal LLM")
        return _poin(status="Tidak Aman", kontribusi=indikator.kontribusi)

    monkeypatch.setattr(guardrail_module, "generate_poin", _stub_exception_lalu_sukses)

    indikator = _indikator(referensi_hukum=[])
    hasil = generate_poin_dengan_guardrail(indikator, retriever=MockRetriever(), max_retry=2)

    assert hasil.low_confidence is False
    assert panggilan["n"] == 2


def test_skor_nol_tidak_menyentuh_generate_poin(monkeypatch):
    def _stub_raise(*args, **kwargs):
        raise AssertionError("generate_poin tidak boleh dipanggil untuk skor 0")

    monkeypatch.setattr(guardrail_module, "generate_poin", _stub_raise)

    indikator = _indikator(skor=0.0, kontribusi=0.0)
    hasil = generate_poin_dengan_guardrail(indikator, retriever=object())

    assert hasil.status == "Aman"
