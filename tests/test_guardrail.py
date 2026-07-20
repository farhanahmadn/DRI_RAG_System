import os

import pytest
from dotenv import load_dotenv

from app.reasoning import guardrail as guardrail_module
from app.reasoning import llm_client as llm_client_module
from app.reasoning.calculator import hitung_target_rekomendasi, pilih_target_utama
from app.reasoning.generator import ambil_chunks_pendukung
from app.reasoning.guardrail import (
    generate_poin_dengan_guardrail,
    perbaiki_poin,
)
from app.reasoning.templates import template_low_confidence
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.schemas import FaktaSpasial, IndikatorJejak, PoinOutput, RekomendasiOutput, SitasiOutput

load_dotenv()


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


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_generate_poin_dengan_guardrail_kdb_live():
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
    retriever = MockRetriever()

    hasil = generate_poin_dengan_guardrail(indikator, retriever)

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 20.0
    assert hasil.rekomendasi.tipe == "numerik"
    assert hasil.rekomendasi.target == pilih_target_utama(hitung_target_rekomendasi(indikator))
    assert hasil.rekomendasi.target == 80.0

    chunk_ids = {c.id for c in ambil_chunks_pendukung(indikator, retriever)}
    assert len(hasil.sitasi) > 0
    for s in hasil.sitasi:
        assert s.citation_id in chunk_ids

    assert hasil.low_confidence is False


# --- Indikator Kegiatan (I/T/B/X) — end-to-end tiap kelas -----------------------------------


def _indikator_kegiatan(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="KEG-01",
        kategori="Kegiatan: Gudang",
        bobot=20.0,
        skor=0.0,
        kontribusi=0.0,
        nilai_input="I",
        ambang="I",
        operator="==",
        formula="",
        zona="C-1",
        referensi_hukum=["RDTR Pasal 1 Ayat 108", "RDTR Lampiran V"],
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def test_kegiatan_klasifikasi_i_aman_template_tanpa_llm():
    indikator = _indikator_kegiatan(skor=0.0, kontribusi=0.0, nilai_input="I")
    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Aman"
    assert hasil.rekomendasi.tipe == "kegiatan"
    assert hasil.rekomendasi.target is None
    assert hasil.low_confidence is False


def _stub_llm_generate_kegiatan(chunk_id: str, saran: str):
    def _stub(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "Penjelasan singkat kegiatan.",
            "reasoning_panjang": "Penjelasan panjang kegiatan berdasarkan klasifikasi ITBX.",
            "sitasi": [{"citation_id": chunk_id, "kutipan": "kutipan relevan"}],
            "saran": saran,
            "disclaimer": None,
        }

    return _stub


def test_kegiatan_klasifikasi_t_berisiko_mock(monkeypatch):
    indikator = _indikator_kegiatan(
        poin_id="KEG-02", kategori="Kegiatan: Gudang", skor=20.0, kontribusi=20.0, nilai_input="T"
    )
    chunk_id = ambil_chunks_pendukung(indikator, MockRetriever())[0].id
    monkeypatch.setattr(
        llm_client_module, "generate", _stub_llm_generate_kegiatan(chunk_id, "Batasi jam operasional gudang.")
    )

    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 20.0
    assert hasil.rekomendasi.tipe == "kegiatan"
    assert hasil.rekomendasi.target is None
    assert len(hasil.sitasi) > 0
    assert all(s.terverifikasi for s in hasil.sitasi)
    assert hasil.low_confidence is False


def test_kegiatan_klasifikasi_b_berisiko_mock(monkeypatch):
    indikator = _indikator_kegiatan(
        poin_id="KEG-03",
        kategori="Kegiatan: Bengkel Kendaraan",
        skor=40.0,
        kontribusi=40.0,
        nilai_input="B",
    )
    chunk_id = ambil_chunks_pendukung(indikator, MockRetriever())[0].id
    monkeypatch.setattr(
        llm_client_module,
        "generate",
        _stub_llm_generate_kegiatan(chunk_id, "Lakukan kajian teknis dan penuhi syarat perizinan tambahan."),
    )

    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 40.0
    assert hasil.rekomendasi.tipe == "kegiatan"
    assert hasil.rekomendasi.target is None
    assert len(hasil.sitasi) > 0
    assert all(s.terverifikasi for s in hasil.sitasi)
    assert hasil.low_confidence is False


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_kegiatan_klasifikasi_x_berisiko_live():
    indikator = _indikator_kegiatan(
        poin_id="KEG-04",
        kategori="Kegiatan: Industri Besar/Pabrik",
        skor=60.0,
        kontribusi=60.0,
        nilai_input="X",
    )
    retriever = MockRetriever()

    hasil = generate_poin_dengan_guardrail(indikator, retriever)

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 60.0
    assert hasil.rekomendasi.tipe == "kegiatan"
    assert hasil.rekomendasi.target is None

    chunk_ids = {c.id for c in ambil_chunks_pendukung(indikator, retriever)}
    assert len(hasil.sitasi) > 0
    for s in hasil.sitasi:
        assert s.citation_id in chunk_ids
        assert s.terverifikasi is True

    assert hasil.low_confidence is False


# --- Indikator numerik KLB & KDH — klon pola KDB --------------------------------------------


def _indikator_numerik(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="NUM-01",
        kategori="KLB",
        bobot=15.0,
        skor=15.0,
        kontribusi=15.0,
        nilai_input=2.8,
        ambang=2.4,
        operator="<=",
        formula="",
        zona="C-1",
        referensi_hukum=["RDTR Lampiran VI"],
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def test_klb_berisiko_mock(monkeypatch):
    indikator = _indikator_numerik(
        kategori="KLB", nilai_input=2.8, ambang=2.4, operator="<=", zona="C-1"
    )
    chunk_id = ambil_chunks_pendukung(indikator, MockRetriever())[0].id
    monkeypatch.setattr(
        llm_client_module,
        "generate",
        _stub_llm_generate_kegiatan(chunk_id, "Kurangi luas total lantai bangunan agar sesuai KLB maksimum."),
    )

    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 15.0
    assert hasil.rekomendasi.tipe == "numerik"
    assert hasil.rekomendasi.target == pilih_target_utama(hitung_target_rekomendasi(indikator))
    assert hasil.rekomendasi.target == 2.4
    assert len(hasil.sitasi) > 0
    assert all(s.terverifikasi for s in hasil.sitasi)
    assert hasil.low_confidence is False


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_kdh_berisiko_live():
    indikator = _indikator_numerik(
        poin_id="NUM-02",
        kategori="KDH",
        bobot=10.0,
        skor=10.0,
        kontribusi=10.0,
        nilai_input=6.0,
        ambang=10.0,
        operator=">=",
        zona="R-1",
    )
    retriever = MockRetriever()

    hasil = generate_poin_dengan_guardrail(indikator, retriever)

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 10.0
    assert hasil.rekomendasi.tipe == "numerik"
    assert hasil.rekomendasi.target == pilih_target_utama(hitung_target_rekomendasi(indikator))
    assert hasil.rekomendasi.target == 10.0

    chunk_ids = {c.id for c in ambil_chunks_pendukung(indikator, retriever)}
    assert len(hasil.sitasi) > 0
    for s in hasil.sitasi:
        assert s.citation_id in chunk_ids
        assert s.terverifikasi is True

    assert hasil.low_confidence is False


# --- Indikator lingkungan: Banjir, Resapan, Sempadan (lokasional) ---------------------------


def test_banjir_rendah_aman_template_tanpa_llm():
    indikator = IndikatorJejak(
        poin_id="BJR-01",
        kategori="Lokasional Banjir",
        bobot=20.0,
        skor=0.0,
        kontribusi=0.0,
        nilai_input="Rendah",
        ambang="Rendah",
        operator="==",
        formula="",
        fakta_spasial=FaktaSpasial(banjir=False, tingkat_banjir="Rendah"),
    )
    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Aman"
    assert hasil.rekomendasi.tipe == "lokasional"
    assert hasil.rekomendasi.target is None
    assert hasil.low_confidence is False


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_banjir_tinggi_berisiko_live():
    indikator = IndikatorJejak(
        poin_id="BJR-02",
        kategori="Lokasional Banjir",
        bobot=20.0,
        skor=60.0,
        kontribusi=60.0,
        nilai_input="Tinggi",
        ambang="Rendah",
        operator="==",
        formula="",
        referensi_hukum=["Metodologi DRI Tingkat 2 Risiko Banjir"],
        fakta_spasial=FaktaSpasial(banjir=True, tingkat_banjir="Tinggi"),
    )
    retriever = MockRetriever()

    hasil = generate_poin_dengan_guardrail(indikator, retriever)

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 60.0
    assert hasil.rekomendasi.tipe == "lokasional"
    assert hasil.rekomendasi.target is None

    chunk_ids = {c.id for c in ambil_chunks_pendukung(indikator, retriever)}
    assert len(hasil.sitasi) > 0
    for s in hasil.sitasi:
        assert s.citation_id in chunk_ids
        assert s.terverifikasi is True

    assert hasil.low_confidence is False


def test_resapan_berisiko_mock(monkeypatch):
    indikator = IndikatorJejak(
        poin_id="RSP-01",
        kategori="Lokasional Resapan Air",
        bobot=20.0,
        skor=20.0,
        kontribusi=20.0,
        nilai_input="dalam_resapan",
        ambang="tidak_dalam_resapan",
        operator="==",
        formula="",
        referensi_hukum=["RTRW Kabupaten Sleman - Kawasan Resapan Air"],
        fakta_spasial=FaktaSpasial(resapan=True),
    )
    chunk_id = ambil_chunks_pendukung(indikator, MockRetriever())[0].id
    monkeypatch.setattr(
        llm_client_module,
        "generate",
        _stub_llm_generate_kegiatan(chunk_id, "Hindari pembangunan pada kawasan resapan air ini."),
    )

    hasil = generate_poin_dengan_guardrail(indikator, MockRetriever())

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 20.0
    assert hasil.rekomendasi.tipe == "lokasional"
    assert hasil.rekomendasi.target is None
    assert len(hasil.sitasi) > 0
    assert all(s.terverifikasi for s in hasil.sitasi)
    assert hasil.low_confidence is False


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip integration test panggilan LLM nyata.",
)
def test_sempadan_melanggar_live():
    indikator = IndikatorJejak(
        poin_id="SMP-01",
        kategori="Lokasional Sempadan Sungai",
        bobot=30.0,
        skor=30.0,
        kontribusi=30.0,
        nilai_input=8.0,
        ambang=15.0,
        operator=">=",
        formula="jarak_sungai_m >= sempadan_minimum_m",
        referensi_hukum=["Permen PUPR 28/2015 Pasal 22"],
        fakta_spasial=FaktaSpasial(jarak_sungai_m=8.0, in_sempadan=True, nama_sungai="Sungai Code", arah="Timur"),
    )
    retriever = MockRetriever()

    hasil = generate_poin_dengan_guardrail(indikator, retriever)

    assert hasil.status == "Tidak Aman"
    assert hasil.kontribusi == 30.0
    assert hasil.rekomendasi.tipe == "lokasional"
    assert hasil.rekomendasi.target is None

    chunk_ids = {c.id for c in ambil_chunks_pendukung(indikator, retriever)}
    assert len(hasil.sitasi) > 0
    for s in hasil.sitasi:
        assert s.citation_id in chunk_ids
        assert s.terverifikasi is True

    teks_narasi = (hasil.reasoning_pendek + " " + hasil.reasoning_panjang + " " + hasil.rekomendasi.saran).lower()
    assert "melebihi" not in teks_narasi

    assert hasil.low_confidence is False
