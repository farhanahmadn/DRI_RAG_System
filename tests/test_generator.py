import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.adapter import adaptasi
from app.reasoning import llm_client as llm_client_module
from app.reasoning.generator import (
    _MAKS_CHAR_INDUK,
    _MAKS_CHUNK_DGN_INDUK,
    _butuh_konteks_induk,
    _QUERY_FALLBACK_PER_POIN,
    _SARAN_AMAN,
    _SARAN_TIDAK_DINILAI,
    _pilih_chunks_referensi,
    _saran_mitigasi_dampak,
    _saran_tidak_dinilai,
    _zona_prefix_dari_nama,
    ambil_chunks_pendukung,
    ambil_konteks_induk,
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

    def test_intensitas_tanpa_subzone_tetap_pakai_query_tajam(self):
        """Dulu cabang ini sengaja memakai query generik "kdb" demi "aman": tanpa filter exact,
        sub-zona satu keluarga (R-2/R-3/R-4) skornya berdekatan dan tabel yang salah bisa terkutip.

        Pengukuran membalik penilaian itu. Atas 21 keluarga zona dengan filter & label IDENTIK,
        "kdb" memberi nDCG@3 0.157 / Recall@3 30.2% dan tabel ambang yang benar TIDAK PERNAH sampai
        peringkat 1 (0/21) — jadi cabang ini (74.3% request nyata) bukan "aman", melainkan menjawab
        tanpa tabel ambang sama sekali. Query tajam: 0.856 / 90.5%, peringkat 1 pada 16/21.

        Ketidakpastian sub-zona tetap nyata dan TIDAK dibantah angka itu — ia ditangani lewat
        caveat (lihat test_caveat_subzona_*), bukan dengan melemahkan query.
        """
        retriever = self._RetrieverPerekamFilter()
        poin = PoinKonteks(
            poin_id="intensitas", kategori="Intensitas Bangunan (KDB/KLB/KDH)", tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS", fakta={}, dasar_hukum=[], zona="Zona Perumahan", zona_subzone=None,
        )
        ambil_chunks_pendukung(poin, retriever)
        assert retriever.query_diterima == "ambang KDB KLB KDH maksimal minimal"
        # Filter TETAP tingkat keluarga — membuka query tidak boleh ikut melonggarkan filter,
        # karena itulah satu-satunya pagar yang mencegah sitasi lintas keluarga zona (APP-2026-6191).
        assert retriever.filters_diterima.zona_prefix == "R"
        assert retriever.filters_diterima.zona is None


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
    assert "85" in kdb_item.deskripsi


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
    assert hasil.rekomendasi.langkah_konkret[0].parameter == "Indeks Limpasan (Runoff)"
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


class TestSaranTidakDinilai:
    """APP-2026-2428: _saran_tidak_dinilai echo `limitations` verbatim (FAITHFUL) kalau BE
    mengirimnya, fallback ke pesan generik lama kalau tidak."""

    def test_dgn_limitations_echo_verbatim(self):
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tidak Dinilai",
            fakta={
                "dinilai": False,
                "limitations": "Permohonan bersinggungan dengan lebih dari 1 persil (memerlukan pengecekan manual)",
            },
        )
        saran = _saran_tidak_dinilai(poin)
        assert "Permohonan bersinggungan dengan lebih dari 1 persil" in saran
        assert saran != _SARAN_TIDAK_DINILAI

    def test_tanpa_limitations_fallback_generik(self):
        poin = _poin(
            poin_id="intensitas",
            tipe_rekomendasi="numerik",
            status="Tidak Dinilai",
            fakta={"dinilai": False},
        )
        assert _saran_tidak_dinilai(poin) == _SARAN_TIDAK_DINILAI

    def test_limitations_kosong_string_fallback_generik(self):
        poin = _poin(
            poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Tidak Dinilai",
            fakta={"dinilai": False, "limitations": ""},
        )
        assert _saran_tidak_dinilai(poin) == _SARAN_TIDAK_DINILAI


class TestSaranMitigasiDampak:
    """APP-2026-8025/-5067: _saran_mitigasi_dampak echo `target_mitigasi['saran_be']` verbatim
    (dirakit calculator.py::hitung_target_mitigasi_dampak dari rekomendasi_mitigasi BE) kalau ada,
    None kalau tidak (fixture lama / dampak tak perlu mitigasi)."""

    def test_dgn_saran_be_echo_verbatim(self):
        poin = _poin(
            poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Tinggi",
            fakta={"target_mitigasi": {"runoff_change_index_maks": 2.5, "saran_be": "Saran resmi BE."}},
        )
        assert _saran_mitigasi_dampak(poin) == "Saran resmi BE."

    def test_tanpa_saran_be_return_none(self):
        poin = _poin(
            poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Tinggi",
            fakta={"target_mitigasi": {"runoff_change_index_maks": 2.5}},
        )
        assert _saran_mitigasi_dampak(poin) is None

    def test_target_mitigasi_kosong_return_none(self):
        poin = _poin(poin_id="dampak", tipe_rekomendasi="numerik-mitigasi", status="Sedang", fakta={})
        assert _saran_mitigasi_dampak(poin) is None

    def test_bukan_poin_dampak_selalu_none(self):
        poin = _poin(
            poin_id="intensitas", tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS",
            fakta={"target_mitigasi": {"saran_be": "seharusnya tak pernah dibaca utk poin ini"}},
        )
        assert _saran_mitigasi_dampak(poin) is None


def test_generate_poin_dampak_mitigasi_saran_be_ditimpa_target_dan_langkah_konkret_terisi(monkeypatch):
    """APP-2026-8025: end-to-end generate_poin() dampak Tinggi dgn rekomendasi_mitigasi BE — saran
    HARUS ditimpa verbatim, target & langkah_konkret (3 item konkret) tetap dari calculator.py
    (dipanggil via poin.fakta['target_mitigasi'] yg sudah dirakit adapter.py)."""
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "Dampak Tinggi krn indeks limpasan melebihi ambang.",
            "reasoning_panjang": "x",
            "sitasi": [],
            "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa BE.",
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
                "runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.923,
                "penyesuaian_lahan": {
                    "luas_bangunan_maks_m2": 11551.06, "luas_rth_min_m2": 3850.35,
                    "kdb_maks_persen": 75, "kdh_min_persen": 25,
                },
                "dimensi_minimum_resapan": {"nilai": 19.01, "satuan": "m³"},
                "saran_be": "Untuk menurunkan dampak dari TINGGI menjadi SEDANG, sesuaikan lahan.",
            },
        },
    )
    hasil = generate_poin(poin, MockRetriever())

    assert hasil.rekomendasi.saran == "Untuk menurunkan dampak dari TINGGI menjadi SEDANG, sesuaikan lahan."
    assert hasil.rekomendasi.target == 2.5
    assert len(hasil.rekomendasi.langkah_konkret) == 3
    assert {l.parameter for l in hasil.rekomendasi.langkah_konkret} == {
        "Luas Bangunan (Atap)", "Luas RTH", "Dimensi Minimum Sumur Resapan",
    }


def test_generate_poin_dampak_tidak_dinilai_multi_persil_saran_echo_limitations(monkeypatch):
    """APP-2026-2428: end-to-end generate_poin() utk poin dampak Tidak Dinilai dgn limitations —
    saran/target/langkah_konkret HARUS ditemplate (bukan dari LLM), reasoning tetap dari LLM."""
    def _stub_generate(prompt, json_schema, *, schema_name="response", system=None, temperature=0.0, max_tokens=1024):
        return {
            "reasoning_pendek": "Dampak tidak dapat dinilai karena poligon bersinggungan >1 persil.",
            "reasoning_panjang": "x",
            "sitasi": [],
            "saran": "Saran dari LLM ini HARUS diabaikan/ditimpa oleh template.",
            "disclaimer": None,
        }

    monkeypatch.setattr(llm_client_module, "generate", _stub_generate)

    poin = _poin(
        poin_id="dampak",
        kategori="Dampak Tata Guna Lahan",
        tipe_rekomendasi="numerik-mitigasi",
        status="Tidak Dinilai",
        fakta={
            "dinilai": False,
            "limitations": "Permohonan bersinggungan dengan lebih dari 1 persil (memerlukan pengecekan manual)",
        },
    )
    hasil = generate_poin(poin, MockRetriever())

    assert "lebih dari 1 persil" in hasil.rekomendasi.saran
    assert hasil.rekomendasi.saran != _SARAN_AMAN
    assert hasil.rekomendasi.target is None
    assert hasil.rekomendasi.langkah_konkret == []


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


class TestPilihChunksReferensiSadarZona:
    """APP-2026-2428 — chunk dari `get_by_reference` dipilih menurut KECOCOKAN ZONA pemohon,
    bukan urutan DB. Bug aslinya: pemohon Zona Pertanian (P-1) disodori Lampiran V.B zona
    Cagar Alam/R-2/R-3 (3 teratas urutan DB) sementara tabel zonanya sendiri ada di urutan 47,
    lalu LLM mengutip Cagar Alam itu & lolos guardrail dgn terverifikasi=True."""

    @staticmethod
    def _poin(zona: str | None = "Zona Pertanian", subzona: str | None = "P-1") -> PoinKonteks:
        return PoinKonteks(
            poin_id="itbx",
            kategori="Klasifikasi Kegiatan (ITBX)",
            tipe_rekomendasi="kategorikal",
            status="I",
            fakta={"lolos": True, "reason": "x"},
            zona=zona,
            zona_subzone=subzona,
            dasar_hukum=[DasarHukum(dokumen="RDTR Sleman", pasal="Matriks ITBX", kutipan="x")],
        )

    @staticmethod
    def _chunk(cid: str, zona: str | None) -> Chunk:
        return Chunk(id=cid, level="tabel", teks=f"teks {cid}", dokumen="RDTR Sleman Tengah", zona=zona)

    def test_zona_keluarga_lain_dibuang_walau_paling_atas_urutan_db(self):
        # Persis kasus APP-2026-2428: CA/R-2/R-3 di urutan teratas, P-1 jauh di belakang.
        chunks = [
            self._chunk("vb-ca", "CA"),
            self._chunk("vb-r-2", "R-2"),
            self._chunk("vb-r-3", "R-3"),
            self._chunk("vb-p-1", "P-1"),
        ]

        hasil = _pilih_chunks_referensi(chunks, self._poin(), top_k=3)

        assert [c.id for c in hasil] == ["vb-p-1"]

    def test_subzona_persis_didahulukan_atas_keluarga_zona(self):
        chunks = [
            self._chunk("vb-p-1-lp2b", "P-1 LP2B"),  # satu keluarga (P), bukan sub-zona persis
            self._chunk("vb-p-1", "P-1"),            # sub-zona persis -> harus naik ke atas
        ]

        hasil = _pilih_chunks_referensi(chunks, self._poin(), top_k=2)

        assert [c.id for c in hasil] == ["vb-p-1", "vb-p-1-lp2b"]

    def test_chunk_tanpa_zona_tetap_dipertahankan(self):
        # Prosa umum (mis. Pasal 43 ttg klasifikasi I/T/B/X) tak terikat zona — sah dikutip zona
        # mana pun, jangan ikut dibuang bersama zona yang salah.
        chunks = [self._chunk("vb-ca", "CA"), self._chunk("p43", None)]

        hasil = _pilih_chunks_referensi(chunks, self._poin(), top_k=3)

        assert [c.id for c in hasil] == ["p43"]

    def test_zona_pemohon_tak_dikenal_pertahankan_perilaku_lama(self):
        # Tanpa sub-zona DAN nama zona tak ada di _ZONA_KODE_PREFIX -> jangan menebak, potong saja.
        chunks = [self._chunk("a", "CA"), self._chunk("b", "R-2"), self._chunk("c", "P-1")]

        hasil = _pilih_chunks_referensi(chunks, self._poin(zona="Zona Antah Berantah", subzona=None), top_k=2)

        assert [c.id for c in hasil] == ["a", "b"]

    def test_hanya_nama_zona_induk_tanpa_subzona_saring_per_keluarga(self):
        chunks = [self._chunk("vb-kt", "KT"), self._chunk("vb-r-4", "R-4"), self._chunk("vb-r-2", "R-2")]

        hasil = _pilih_chunks_referensi(chunks, self._poin(zona="Zona Perumahan", subzona=None), top_k=3)

        assert [c.id for c in hasil] == ["vb-r-4", "vb-r-2"]

    def test_semua_kandidat_salah_zona_jatuh_ke_search_berfilter(self):
        """Kalau tak ada satu pun kandidat yang cocok zona, lebih baik daftar kosong -> pemanggil
        jatuh ke search() yang SUDAH berfilter zona, drpd menyodorkan Lampiran zona lain."""

        class _RetrieverSalahZonaSemua:
            def __init__(self):
                self.query_search = None

            def search(self, query, filters, top_k=5):
                self.query_search = (query, filters)
                return [Chunk(id="hasil-search", level="tabel", teks="t", dokumen="d", zona="P-1")]

            def get_by_reference(self, referensi):
                return [
                    Chunk(id="vb-ca", level="tabel", teks="t", dokumen="d", zona="CA"),
                    Chunk(id="vb-r-2", level="tabel", teks="t", dokumen="d", zona="R-2"),
                ]

            def get_parent(self, chunk_id):
                return None

        retriever = _RetrieverSalahZonaSemua()

        hasil = ambil_chunks_pendukung(self._poin(), retriever)

        assert [c.id for c in hasil] == ["hasil-search"]
        assert retriever.query_search is not None, "harus jatuh ke search(), bukan diam-diam kosong"
        assert retriever.query_search[1].zona == "P-1"


class TestAmbilKonteksInduk:
    """Small-to-big: chunk ayat dilengkapi teks pasal induknya. 62% dari 895 chunk ayat di korpus
    nyata memuat "sebagaimana dimaksud pada ayat (N)" — merujuk teks yang tak ikut terkirim ke LLM.
    Chunker & get_parent sudah menyiapkan ini sejak awal, tapi tak pernah tersambung ke reasoning."""

    class _RetrieverInduk:
        def __init__(self, induk_by_child=None, raise_exc=None):
            self._induk = induk_by_child or {}
            self._raise = raise_exc
            self.dipanggil = []

        def search(self, query, filters, top_k=5):
            return []

        def get_by_reference(self, referensi):
            return []

        def get_parent(self, chunk_id):
            self.dipanggil.append(chunk_id)
            if self._raise:
                raise self._raise
            return self._induk.get(chunk_id)

    @staticmethod
    def _ayat(cid: str, parent: str, teks: str | None = None) -> Chunk:
        # Default memuat rujukan silang — itu prasyarat ekspansi induk (lihat _butuh_konteks_induk).
        return Chunk(
            id=cid, level="ayat", parent_id=parent, dokumen="d",
            teks=teks if teks is not None else f"({cid}) sebagaimana dimaksud pada ayat (1) berlaku.",
        )

    @staticmethod
    def _pasal(cid: str, teks: str) -> Chunk:
        return Chunk(id=cid, level="pasal", teks=teks, dokumen="d")

    def test_ayat_dapat_teks_induknya(self):
        anak = self._ayat("p41-a3", "p41")
        rt = self._RetrieverInduk({"p41-a3": self._pasal("p41", "Pasal 41 lengkap")})

        assert ambil_konteks_induk([anak], rt) == {"p41-a3": "Pasal 41 lengkap"}

    def test_chunk_bukan_ayat_dilewati(self):
        rt = self._RetrieverInduk()
        chunks = [self._pasal("p41", "x"), Chunk(id="vb-p-1", level="tabel", teks="t", dokumen="d")]

        assert ambil_konteks_induk(chunks, rt) == {}
        assert rt.dipanggil == [], "get_parent tak perlu dipanggil utk pasal/tabel"

    def test_induk_sama_hanya_dilampirkan_sekali(self):
        # Dua ayat dari pasal yang sama -> teks induk identik, jangan digandakan di prompt.
        a1, a2 = self._ayat("p41-a1", "p41"), self._ayat("p41-a3", "p41")
        rt = self._RetrieverInduk({"p41-a1": self._pasal("p41", "Pasal 41 lengkap"),
                                   "p41-a3": self._pasal("p41", "Pasal 41 lengkap")})

        hasil = ambil_konteks_induk([a1, a2], rt)

        assert list(hasil) == ["p41-a1"]

    def test_dibatasi_beberapa_chunk_teratas(self):
        """Batas ada karena alasan nyata: pasal induk terpanjang di korpus 30.748 char — tanpa
        batas, satu pasal bisa memicu ulang 413/429 (APP-2026-9461)."""
        chunks = [self._ayat(f"p{i}-a1", f"p{i}") for i in range(5)]
        rt = self._RetrieverInduk({f"p{i}-a1": self._pasal(f"p{i}", f"teks {i}") for i in range(5)})

        hasil = ambil_konteks_induk(chunks, rt)

        assert len(hasil) == _MAKS_CHUNK_DGN_INDUK
        assert len(rt.dipanggil) == _MAKS_CHUNK_DGN_INDUK

    def test_induk_kepanjangan_dipotong(self):
        anak = self._ayat("p1-a1", "p1")
        rt = self._RetrieverInduk({"p1-a1": self._pasal("p1", "x" * (_MAKS_CHAR_INDUK + 5000))})

        hasil = ambil_konteks_induk([anak], rt)["p1-a1"]

        assert len(hasil) < _MAKS_CHAR_INDUK + 50
        assert hasil.endswith("[…dipotong]")

    def test_get_parent_gagal_tidak_menggagalkan_generasi(self):
        anak = self._ayat("p1-a1", "p1")
        rt = self._RetrieverInduk(raise_exc=RuntimeError("DB putus"))

        assert ambil_konteks_induk([anak], rt) == {}  # tidak raise

    def test_induk_tak_ditemukan_dilewati(self):
        anak = self._ayat("p1-a1", "p1")

        assert ambil_konteks_induk([anak], self._RetrieverInduk({})) == {}


class TestButuhKonteksInduk:
    """Saringan yang mencegah ekspansi induk jadi pemborosan. Tanpa ini, definisi Pasal 1 ikut
    ditempeli induknya — padahal definisi sudah mandiri DAN induknya pasal terpanjang di korpus
    (30.748 char). Terbukti live: chunk p1-a117 menarik 2.511 char berisi 126 definisi tak terkait."""

    def test_ayat_dgn_rujukan_silang_butuh_induk(self):
        assert _butuh_konteks_induk("(3) Ketentuan sebagaimana dimaksud pada ayat (1) berlaku.")
        assert _butuh_konteks_induk("(2) Lokasi pada ayat (1) huruf b.")

    def test_definisi_mandiri_tidak_butuh_induk(self):
        # Bentuk nyata definisi Pasal 1 (p1-a117) — self-contained.
        assert not _butuh_konteks_induk(
            "117. Koefisien Dasar Bangunan yang selanjutnya disingkat KDB adalah angka persentase "
            "perbandingan antara luas seluruh lantai dasar bangunan dan luas lahan."
        )

    def test_tidak_peka_huruf_besar_kecil(self):
        assert _butuh_konteks_induk("(3) SEBAGAIMANA DIMAKSUD pada ayat (1).")


def test_ayat_mandiri_tidak_menarik_induk():
    """Integrasi saringan ke ambil_konteks_induk: chunk mandiri tak memicu get_parent sama sekali."""
    rt = TestAmbilKonteksInduk._RetrieverInduk({"p1-a117": Chunk(id="p1", level="pasal", teks="x" * 5000, dokumen="d")})
    mandiri = Chunk(
        id="p1-a117", level="ayat", parent_id="p1", dokumen="d",
        teks="117. Koefisien Dasar Bangunan adalah angka persentase perbandingan luas lantai dasar.",
    )

    assert ambil_konteks_induk([mandiri], rt) == {}
    assert rt.dipanggil == [], "get_parent tak perlu dipanggil utk chunk yang sudah mandiri"


class TestCaveatSubzona:
    """Pasangan wajib dari pembukaan gating query tajam di `ambil_chunks_pendukung`."""

    def _poin(self, poin_id="intensitas", subzona=None):
        return PoinKonteks(
            poin_id=poin_id, kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik", status="MELAMPAUI_BATAS", fakta={}, dasar_hukum=[],
            zona="Zona Perumahan", zona_subzone=subzona,
        )

    def test_caveat_subzona_muncul_saat_subzona_kosong(self):
        from app.reasoning.generator import caveat_subzona
        from app.reasoning.prompts import CAVEAT_SUBZONA_TAK_TERKONFIRMASI

        assert caveat_subzona(self._poin()) == [CAVEAT_SUBZONA_TAK_TERKONFIRMASI]

    def test_caveat_subzona_hilang_saat_subzona_presisi_ada(self):
        from app.reasoning.generator import caveat_subzona

        assert caveat_subzona(self._poin(subzona="R-2")) == [],             "filter exact -> tabelnya memang milik sub-zona pemohon, tak ada yang perlu disangkal"

    def test_caveat_subzona_absen_saat_intensitas_tidak_dinilai(self):
        """BE tak mengirim penilaian intensitas -> adapter tetap merakit poinnya dgn status
        "Tidak Dinilai" dan tanpa parameter apa pun. Tak ada ambang yang dikutip, jadi caveat di situ
        memperingatkan angka yang tidak ada. Syaratnya harus sama dgn assemble._intensitas_dinilai."""
        from app.reasoning.generator import caveat_subzona

        poin = PoinKonteks(
            poin_id="intensitas", kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik", status="Tidak Dinilai", fakta={"dinilai": False},
            dasar_hukum=[], zona="Zona Perumahan", zona_subzone=None,
        )

        assert caveat_subzona(poin) == []

    def test_caveat_subzona_tidak_dipasang_di_poin_lain(self):
        from app.reasoning.generator import caveat_subzona

        for pid in ("itbx", "dampak"):
            assert caveat_subzona(self._poin(poin_id=pid)) == [],                 f"{pid} tak mengutip ambang KDB/KLB/KDH; caveat di sana cuma derau"

    def test_caveat_masuk_blok_wajib_disebut_di_prompt(self):
        from app.reasoning.generator import caveat_subzona
        from app.reasoning.prompts import build_user_prompt

        poin = self._poin()
        prompt = build_user_prompt(poin, [], None, catatan_tambahan=caveat_subzona(poin))

        assert "Catatan (WAJIB disebutkan dalam reasoning)" in prompt
        assert "tidak dikonfirmasi oleh back-end" in prompt

    def test_caveat_tambahan_tidak_menghapus_caveat_backend(self):
        """meta.caveats dari BE dan catatan kode harus SAMA-SAMA masuk, bukan saling menimpa."""
        from app.reasoning.prompts import build_user_prompt
        from app.schemas import MetaL2

        meta = MetaL2(caveats=["Caveat asli dari back-end."])
        prompt = build_user_prompt(self._poin(), [], meta, catatan_tambahan=["Catatan dari kode."])

        assert "Caveat asli dari back-end." in prompt
        assert "Catatan dari kode." in prompt
