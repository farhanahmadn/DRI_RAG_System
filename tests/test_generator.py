import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.adapter import adaptasi
from app.reasoning import llm_client as llm_client_module
from app.reasoning.generator import (
    _QUERY_FALLBACK_PER_POIN,
    _SARAN_AMAN,
    _zona_prefix_dari_nama,
    ambil_chunks_pendukung,
    apakah_aman,
    generate_poin,
)
from app.retrieval.base import Chunk
from app.retrieval.mock import MockRetriever
from app.retrieval.retriever import _expand
from app.schemas import DasarHukum, L2Assessment, PoinKonteks

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


class TestApakahAman:
    def test_itbx_aman(self):
        assert apakah_aman(_poin(status="I", fakta={"lolos": True, "reason": "x"})) is True

    def test_itbx_tidak_aman_karena_bukan_i(self):
        assert apakah_aman(_poin(status="B", fakta={"lolos": True, "reason": "x"})) is False

    def test_itbx_tidak_aman_karena_fallback(self):
        poin = _poin(status="I", fakta={"lolos": True, "reason": "x", "fallback_data_kosong": True})
        assert apakah_aman(poin) is False

    def test_intensitas_aman_tanpa_target(self):
        poin = _poin(poin_id="intensitas", tipe_rekomendasi="numerik", status="MEMENUHI_SYARAT", fakta={"target": {}})
        assert apakah_aman(poin) is True

    def test_intensitas_tidak_aman_dengan_target(self):
        poin = _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0}}},
        )
        assert apakah_aman(poin) is False

    def test_dampak_aman_tanpa_mitigasi(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}},
        )
        assert apakah_aman(poin) is True

    def test_dampak_tidak_aman_dengan_mitigasi(self):
        poin = _poin(
            poin_id="dampak",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={"mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB"]}},
        )
        assert apakah_aman(poin) is False


class TestAmbilChunksPendukungDibatasi:
    """Investigasi ITBX APP-2026-6191 (live thd DB nyata): dasar_hukum berlabel non-pasal ("Matriks
    ITBX") + dokumen generik ("RDTR Sleman") bikin get_by_reference sungguhan mengembalikan RATUSAN
    chunk tak terbatas (seluruh korpus) — bukan cuma sedikit seperti di MockRetriever. Prompt yang
    membanjiri LLM bikin ia gagal memilih sitasi sama sekali (low_confidence). Cek di sini murni pakai
    stub Retriever, tanpa DB, supaya regresi kecapatan tetap ketahuan offline."""

    class _RetrieverBanjirReferensi:
        def __init__(self, jumlah: int):
            self._jumlah = jumlah

        def search(self, query, filters, top_k=5):
            return []

        def get_by_reference(self, referensi):
            return [
                Chunk(id=f"chunk-{i}", level="pasal", teks=f"teks {i}", dokumen="RDTR Sleman", pasal=str(i))
                for i in range(self._jumlah)
            ]

        def get_parent(self, chunk_id):
            return None

    def test_get_by_reference_ratusan_chunk_dipotong_ke_top_k(self):
        poin = PoinKonteks(
            poin_id="itbx",
            kategori="Klasifikasi Kegiatan (ITBX)",
            tipe_rekomendasi="kategorikal",
            status="T",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="x")],
        )
        chunks = ambil_chunks_pendukung(poin, self._RetrieverBanjirReferensi(500), top_k_dukungan=3)
        assert len(chunks) == 3

    def test_get_by_reference_sedikit_chunk_tidak_dipotong(self):
        poin = PoinKonteks(
            poin_id="itbx",
            kategori="Klasifikasi Kegiatan (ITBX)",
            tipe_rekomendasi="kategorikal",
            status="T",
            fakta={"lolos": True, "reason": "x"},
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="x")],
        )
        chunks = ambil_chunks_pendukung(poin, self._RetrieverBanjirReferensi(2), top_k_dukungan=3)
        assert len(chunks) == 2


class TestZonaPrefixDariNama:
    """Mapping nama zona induk -> kode prefix, dari Pasal 17/23 dokumen sumber (data/raw/*.pdf,
    diverifikasi via data/parsed/v1/*.md — lihat APP-2026-6191)."""

    def test_zona_perumahan_ke_r(self):
        assert _zona_prefix_dari_nama("Zona Perumahan") == "R"

    def test_zona_perkantoran_ke_kt(self):
        assert _zona_prefix_dari_nama("Zona Perkantoran") == "KT"

    def test_case_insensitive_dan_strip_spasi(self):
        assert _zona_prefix_dari_nama("  zona PERUMAHAN  ") == "R"

    def test_none_return_none(self):
        assert _zona_prefix_dari_nama(None) is None

    def test_string_kosong_return_none(self):
        assert _zona_prefix_dari_nama("") is None

    def test_nama_tak_dikenal_return_none_bukan_menebak(self):
        assert _zona_prefix_dari_nama("Zona Antah Berantah") is None


class TestAmbilChunksPendukungZonaFilter:
    """APP-2026-6191: fallback search() HARUS bawa zona_prefix poin, supaya tak lintas-keluarga
    zona (mis. Lampiran VI Zona Perkantoran "KT" ikut terkutip utk pemohon Zona Perumahan "R")."""

    class _RetrieverPerekamFilter:
        def __init__(self):
            self.filters_diterima = None
            self.query_diterima = None

        def search(self, query, filters, top_k=5):
            self.filters_diterima = filters
            self.query_diterima = query
            return []

        def get_by_reference(self, referensi):
            return []

        def get_parent(self, chunk_id):
            return None

    def test_zona_poin_diteruskan_sbg_zona_prefix(self):
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona="Zona Perumahan",
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.filters_diterima.zona_prefix == "R"

    def test_zona_none_hasilkan_filter_tanpa_zona_prefix(self):
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona=None,
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.filters_diterima.zona_prefix is None

    def test_zona_tak_dikenal_hasilkan_filter_tanpa_zona_prefix(self):
        # Nama zona yang tak ada di mapping -> JANGAN menebak, search tanpa filter zona (perilaku lama).
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona="Zona Misterius",
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.filters_diterima.zona_prefix is None

    def test_zona_subzone_diteruskan_sbg_filter_zona_exact(self):
        # APP-2026-8090: kalau poin.zona_subzone terisi (mis. "P-1"), filter HARUS exact `zona`
        # (jauh lebih presisi drpd zona_prefix keluarga "P" yg tak bisa beda P-1/P-2/P-3).
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona="Zona Pertanian", zona_subzone="P-1",
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.filters_diterima.zona == "P-1"
        assert retriever.filters_diterima.zona_prefix is None  # exact & prefix tak digabung sekaligus

    def test_intensitas_dgn_subzone_pakai_query_lebih_tajam(self):
        # APP-2026-8090 (verifikasi live thd DB): query "kdb" polos kalah oleh pasal definisional
        # umum walau filter zona sudah benar — tabel ambang Lampiran VI baru naik ke rank #1 dgn
        # query lebih spesifik. AMAN dipakai di sini krn filter zona EXACT (bukan cuma keluarga).
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="intensitas", kategori="Intensitas Bangunan (KDB/KLB/KDH)", tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS", fakta={}, dasar_hukum=[], zona="Zona Pertanian", zona_subzone="P-1",
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.query_diterima == "ambang KDB KLB KDH maksimal minimal"
        assert retriever.filters_diterima.zona == "P-1"

    def test_intensitas_tanpa_subzone_tetap_query_generik(self):
        # Tanpa sub-zona presisi (cuma zona_prefix keluarga), query TETAP generik "kdb" — query
        # tajam TANPA filter exact terbukti bikin sub-zona (mis. R-2/R-3/R-4) skor berdekatan &
        # berisiko kutip tabel sub-zona yang salah.
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="intensitas", kategori="Intensitas Bangunan (KDB/KLB/KDH)", tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS", fakta={}, dasar_hukum=[], zona="Zona Perumahan", zona_subzone=None,
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.query_diterima == "kdb"
        assert retriever.filters_diterima.zona_prefix == "R"

    def test_dampak_dgn_subzone_query_tak_berubah(self):
        # Query lebih tajam HANYA berlaku utk poin_id="intensitas" — dampak/itbx tetap pakai
        # _QUERY_FALLBACK_PER_POIN spt biasa, walau zona_subzone tersedia.
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona="Zona Pertanian", zona_subzone="P-1",
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.query_diterima == "dampak tata guna lahan"

    def test_zona_subzone_kosong_fallback_ke_zona_prefix(self):
        # zona_subzone None (BE tak kirim/tak yakin) -> fallback ke perilaku lama (zona_prefix).
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi", fakta={}, dasar_hukum=[], zona="Zona Pertanian", zona_subzone=None,
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.filters_diterima.zona_prefix == "P"
        assert retriever.filters_diterima.zona is None


class TestQueryFallbackDampak:
    """Fix #5: poin dampak (dasar_hukum selalu kosong) skrng punya kata kunci fallback pendek
    yang match entri _EXPANSION baru di app/retrieval/retriever.py — bukan poin.kategori panjang
    apa adanya. Relevansi sitasi sungguhan (mis. menarik pasal Resapan Air/zero delta Q) butuh
    run live thd DB nyata — DITANDAI tertunda, bukan diklaim terverifikasi di sini."""

    def test_dampak_ada_di_query_fallback_bukan_kategori_mentah(self):
        assert "dampak" in _QUERY_FALLBACK_PER_POIN
        assert _QUERY_FALLBACK_PER_POIN["dampak"] != "Dampak Tata Guna Lahan"

    def test_query_fallback_dampak_tereskpansi_via_retriever_asli(self):
        query = _QUERY_FALLBACK_PER_POIN["dampak"]
        hasil_ekspansi = _expand(query)

        assert hasil_ekspansi != query  # benar-benar diperkaya, bukan diteruskan mentah
        for istilah in ("runoff", "sumur resapan", "kolam retensi", "zero delta q", "rth"):
            assert istilah in hasil_ekspansi.lower()

    def test_ambil_chunks_pendukung_dampak_pakai_query_pendek_bukan_kategori(self):
        dipanggil = {}

        class _RetrieverPencatatQuery:
            def search(self, query, filters, top_k=5):
                dipanggil["query"] = query
                return []

            def get_by_reference(self, referensi):
                return []

            def get_parent(self, chunk_id):
                return None

        poin = PoinKonteks(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={},
            dasar_hukum=[],
        )
        ambil_chunks_pendukung(poin, _RetrieverPencatatQuery())

        assert dipanggil["query"] == "dampak tata guna lahan"


def test_generate_poin_aman_tetap_panggil_llm_untuk_reasoning_dan_sitasi(monkeypatch):
    # APP-2026-3468: poin aman TIDAK LAGI short-circuit ke template generik — reasoning_pendek/
    # panjang & sitasi TETAP dihasilkan LLM (menjelaskan KENAPA lolos), hanya saran/target yang
    # ditemplate deterministik (tak ada tindakan lanjut utk direkomendasikan).
    dipanggil = {"llm": False}

    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        dipanggil["llm"] = True
        return {
            "reasoning_pendek": "Kegiatan termasuk kategori Diizinkan (I) di zona ini.",
            "reasoning_panjang": "Kegiatan yang diusulkan termasuk kategori Diizinkan (I) sesuai Matriks ITBX zona perumahan.",
            "sitasi": [{"citation_id": "anchor-0", "kutipan": "Kutipan dari anchor."}],
            "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa oleh template.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        status="I",
        fakta={"lolos": True, "reason": "x"},
        dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="Data KBLI referensi")],
    )
    assert apakah_aman(poin) is True

    hasil = generate_poin(poin, MockRetriever())

    assert dipanggil["llm"] is True
    assert hasil.reasoning_pendek == "Kegiatan termasuk kategori Diizinkan (I) di zona ini."
    assert hasil.reasoning_panjang.strip() != ""
    assert len(hasil.sitasi) == 1  # sitasi TETAP dari LLM, bukan dikosongkan seperti template lama
    assert hasil.sitasi[0].citation_id == "anchor-0"


def test_generate_poin_intensitas_langkah_konkret_semua_parameter_melanggar(monkeypatch):
    # APP-2026-8090 (live nyata): KDB & KDH melanggar BERSAMAAN — langkah_konkret HARUS kasih
    # keduanya (beda dari `target`, yg cuma 1 angka representatif).
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x", "reasoning_panjang": "x", "sitasi": [],
            "saran": "1. Sesuaikan KDB.\n2. Sesuaikan KDH.", "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="MELAMPAUI_BATAS",
        fakta={
            "parameter": {
                "kdb": {"usulan": 40, "ambang_maks": 10, "ambang_min": None, "memenuhi": False, "satuan": "persen"},
                "kdh": {"usulan": 29.4, "ambang_maks": None, "ambang_min": 88, "memenuhi": False, "satuan": "persen"},
            },
            "target": {
                "kdb": {"target_kdb": 10.0, "selisih": 30.0, "footprint_maks_m2": 85.0},
                "kdh": {"target_kdh": 88.0, "selisih": 58.6, "rth_dibutuhkan_m2": 748.0},
            },
        },
    )
    hasil = generate_poin(poin, MockRetriever())

    assert len(hasil.rekomendasi.langkah_konkret) == 2
    parameter_terlihat = {l.parameter for l in hasil.rekomendasi.langkah_konkret}
    assert parameter_terlihat == {"KDB", "KDH"}
    kdb_item = next(l for l in hasil.rekomendasi.langkah_konkret if l.parameter == "KDB")
    assert kdb_item.nilai_saat_ini == 40
    assert kdb_item.nilai_target == 10.0
    assert "85.0" in kdb_item.deskripsi


def test_generate_poin_dampak_langkah_konkret_dari_target_mitigasi(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x", "reasoning_panjang": "x", "sitasi": [],
            "saran": "1. Turunkan indikator limpasan.", "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
        status="Tinggi",
        fakta={
            "dinilai": True,
            "mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB"]},
            "target_mitigasi": {"runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.85},
        },
    )
    hasil = generate_poin(poin, MockRetriever())

    assert len(hasil.rekomendasi.langkah_konkret) == 1
    assert hasil.rekomendasi.langkah_konkret[0].parameter == "runoff_change_index"
    assert hasil.rekomendasi.langkah_konkret[0].nilai_target == 2.5


def test_generate_poin_aman_saran_dan_target_tetap_ditemplate(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [],
            "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa oleh template.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="MEMENUHI_SYARAT",
        fakta={
            "parameter": {"kdb": {"usulan": 50, "ambang_maks": 60, "ambang_min": None, "memenuhi": True, "satuan": "persen"}},
            "target": {},
        },
    )
    assert apakah_aman(poin) is True

    hasil = generate_poin(poin, MockRetriever())

    assert hasil.rekomendasi.saran == _SARAN_AMAN
    assert hasil.rekomendasi.target is None
    assert hasil.rekomendasi.langkah_konkret == []
    assert hasil.low_confidence is False


def test_generate_poin_tidak_aman_pakai_saran_llm_apa_adanya(monkeypatch):
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [],
            "saran": "Saran spesifik dari LLM.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})
    assert apakah_aman(poin) is False

    hasil = generate_poin(poin, MockRetriever())
    assert hasil.rekomendasi.saran == "Saran spesifik dari LLM."


def test_generate_poin_tidak_dinilai_saran_bukan_saran_aman(monkeypatch):
    """APP-2026-003: poin "Tidak Dinilai" (gate berhenti sebelum poin ini dievaluasi) BUKAN
    "memenuhi ketentuan" — `_SARAN_AMAN` di sini kontradiktif dgn fakta (belum pernah diperiksa),
    pola sama dgn bug kesimpulan APP-2026-8376 yg sudah diperbaiki sebelumnya."""

    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [],
            "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa oleh template.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        tipe_rekomendasi="numerik",
        status="Tidak Dinilai",
        fakta={"dinilai": False},
    )
    # apakah_aman() jg True utk kasus ini (tak ada "target") — makanya perlu cabang terpisah
    # keyed on `status`, BUKAN cuma mengandalkan apakah_aman().
    assert apakah_aman(poin) is True

    hasil = generate_poin(poin, MockRetriever())

    assert hasil.rekomendasi.saran != _SARAN_AMAN
    assert "tidak dievaluasi" in hasil.rekomendasi.saran.lower()
    assert hasil.rekomendasi.target is None


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


def test_generate_poin_dampak_dapat_target_mitigasi_numerik(monkeypatch):
    """Poin dampak (tipe_rekomendasi='numerik-mitigasi') kini juga dapat `target` numerik dari
    `fakta['target_mitigasi']` (app/reasoning/calculator.py::hitung_target_mitigasi_dampak) — bukan
    None lagi selamanya seperti sebelumnya. Angka dari calculator, BUKAN dari LLM."""
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "Dampak tergolong Tinggi, perlu mitigasi.",
            "reasoning_panjang": "Indikator limpasan perlu ditekan di bawah 2.5.",
            "sitasi": [],
            "saran": "1. Turunkan KDB.\n2. Naikkan KDH hingga indikator limpasan di bawah 2.5.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="dampak",
        kategori="Dampak Tata Guna Lahan",
        tipe_rekomendasi="numerik-mitigasi",
        status="Tinggi",
        fakta={
            "dinilai": True,
            "mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB", "naikkan KDH/RTH"]},
            "target_mitigasi": {
                "runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.85,
            },
        },
    )

    hasil = generate_poin(poin, MockRetriever())

    assert hasil.rekomendasi.tipe == "numerik-mitigasi"
    assert hasil.rekomendasi.target == 2.5  # dari calculator, BUKAN dari LLM
    assert hasil.low_confidence is False


def test_generate_poin_dampak_tanpa_target_mitigasi_tetap_none(monkeypatch):
    """Kalau target_mitigasi kosong (mis. threshold_bands tak lengkap) -> target tetap None, TIDAK
    error/crash — perilaku fallback aman dipertahankan."""
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x", "reasoning_panjang": "x", "sitasi": [], "saran": "x", "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="dampak", kategori="Dampak Tata Guna Lahan", tipe_rekomendasi="numerik-mitigasi",
        status="Tinggi",
        fakta={"dinilai": True, "mitigasi": {"perlu_mitigasi": True, "arah": []}, "target_mitigasi": {}},
    )

    hasil = generate_poin(poin, MockRetriever())
    assert hasil.rekomendasi.target is None


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


def test_generate_poin_sitasi_chunk_kutipan_selalu_verbatim_bukan_dari_llm(monkeypatch):
    # Fix #3: kutipan utk sitasi berbasis chunk WAJIB chunk.teks apa adanya — kutipan yang ditulis
    # LLM di sini cuma dipakai LLM utk MEMILIH citation_id yang relevan, isinya sendiri diabaikan
    # (LLM tak boleh mengarang/menulis-ulang teks kutipan pasal).
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "x",
            "reasoning_panjang": "x",
            "sitasi": [
                {"citation_id": "rdtr-p1-a108", "kutipan": "Ini kutipan karangan LLM, bukan teks pasal asli."}
            ],
            "saran": "x",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(status="B", fakta={"lolos": True, "reason": "x"})  # tanpa dasar_hukum -> fallback search "kegiatan"
    retriever = MockRetriever()
    hasil = generate_poin(poin, retriever)

    assert len(hasil.sitasi) == 1
    chunk_asli = next(c for c in retriever._chunks if c.id == "rdtr-p1-a108")
    assert hasil.sitasi[0].kutipan == chunk_asli.teks
    assert hasil.sitasi[0].kutipan != "Ini kutipan karangan LLM, bukan teks pasal asli."


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
    assessment = _muat_assessment("l2_sample_amplop_6191.json")
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
