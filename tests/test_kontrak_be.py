"""Pemetaan keluaran internal ke Spesifikasi Kontrak Variabel Output AI Reasoning (L3) v1.0.0 dari BE."""

import json
from pathlib import Path

import pytest

from app.kontrak_be import AdvisoryL3, ke_kontrak_be
from app.reasoning import assemble as assemble_module
from app.reasoning.guardrail import DiagnosaPoin
from app.retrieval.mock import MockRetriever
from app.schemas import L2Envelope, LangkahKonkretOutput, PoinOutput, RekomendasiOutput, SitasiOutput

FIXTURES = Path(__file__).parent / "fixtures"


def _muat(nama="l2_sample_multizona_0022.json", ubah=None):
    payload = json.loads((FIXTURES / nama).read_text(encoding="utf-8"))
    if ubah:
        ubah(payload["data"])
    return L2Envelope.model_validate(payload).data


def _stub_poin(poin, retriever, assessment, *, max_retry=2):
    sitasi = [SitasiOutput(citation_id="anchor-0", dokumen="RDTR Sleman", pasal="Matriks ITBX (P)", halaman=0,
                           kutipan="Data KBLI referensi", terverifikasi=True)]
    pelanggar = poin.status in ("MELAMPAUI_BATAS", "B", "Sedang", "Tinggi")
    langkah = [LangkahKonkretOutput(parameter="KDB", deskripsi="KDB harus turun ke maksimal 10%",
                                    nilai_saat_ini=68.36, nilai_target=10.0, satuan="persen")] if pelanggar else []
    saran = ("1. Turunkan KDB.\n2. Tambah RTH." if pelanggar
             else "Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan.")
    out = PoinOutput(
        poin_id=poin.poin_id, kategori=poin.kategori, status=poin.status,
        reasoning_pendek=f"Pendek {poin.poin_id}.", reasoning_panjang=f"Panjang {poin.poin_id}.", sitasi=sitasi,
        rekomendasi=RekomendasiOutput(tipe=poin.tipe_rekomendasi, saran=saran, langkah_konkret=langkah),
        low_confidence=False)
    return out, DiagnosaPoin(poin_id=poin.poin_id, berhasil=True, percobaan=1, jumlah_chunk=1)


@pytest.fixture
def jalankan(monkeypatch):
    monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub_poin)
    monkeypatch.setattr(assemble_module, "log_precheck", lambda *a, **k: None)
    monkeypatch.setattr(assemble_module.llm_client, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sintesis dimatikan")))
    return lambda a: ke_kontrak_be(assemble_module.jalankan_precheck(a, MockRetriever()))


def _satu_zona(d):
    d["gate_hukum"]["itbx_zones"] = d["gate_hukum"]["itbx_zones"][:1]


def _semua(poin):
    for p in poin:
        yield p
        yield from _semua(p.sub_poin)


class TestPohonSatuZona:
    def test_dua_akar_dan_id_daun_sesuai_spesifikasi(self, jalankan):
        out = jalankan(_muat(ubah=_satu_zona))
        assert [p.poin_id for p in out.poin] == ["gate_hukum", "gate_hidrologis"]
        assert [s.poin_id for s in out.poin[0].sub_poin] == ["itbx_evaluasi", "intensitas_evaluasi"]
        assert [s.poin_id for s in out.poin[1].sub_poin] == ["dampak"]

    def test_status_akar(self, jalankan):
        out = jalankan(_muat(ubah=_satu_zona))
        assert out.poin[0].status == "LOLOS"
        assert out.poin[1].status == "MEMENUHI_SYARAT"  # dampak Rendah

    def test_rekomendasi_daun_punya_bentuk_kamus_data(self, jalankan):
        intensitas_p = jalankan(_muat(ubah=_satu_zona)).poin[0].sub_poin[1]
        r = intensitas_p.rekomendasi
        assert isinstance(r.saran, str) and isinstance(r.poin, list) and isinstance(r.alasan, str)
        assert all(isinstance(x, str) for x in r.langkah_konkret)

    def test_dasar_regulasi_dari_sitasi_dan_digabung_di_akar(self, jalankan):
        akar = jalankan(_muat(ubah=_satu_zona)).poin[0]
        d = akar.sub_poin[0].dasar_regulasi[0]
        assert (d.judul, d.deskripsi, d.kategori) == (
            "RDTR Sleman - Matriks ITBX (P)", "Data KBLI referensi", "Regulasi Daerah")
        assert d.citation_id == "anchor-0" and d.terverifikasi is True
        assert len(akar.dasar_regulasi) == 1  # duplikat antar anak digabung

    def test_ringkasan_gate_memuat_status_itbx_dan_intensitas(self, jalankan):
        kalimat = jalankan(_muat(ubah=_satu_zona)).ringkasan_gate.kalimat
        assert kalimat.endswith("status ITBX Diizinkan (I), intensitas memenuhi syarat.")


class TestPohonMultiZona:
    def test_zona_menjadi_simpul_dengan_anak_itbx_dan_intensitas(self, jalankan):
        akar = jalankan(_muat()).poin[0]
        assert [z.poin_id for z in akar.sub_poin] == ["zona_P", "zona_R"]
        assert [s.poin_id for s in akar.sub_poin[1].sub_poin] == ["itbx_R", "intensitas_R"]
        assert akar.sub_poin[0].kategori == "Penilaian Tata Ruang: Zona Pertanian (P)"
        assert akar.sub_poin[0].zona.persentase == 50.4

    def test_dampak_tetap_satu(self, jalankan):
        out = jalankan(_muat())
        assert sum(1 for p in _semua(out.poin) if p.poin_id == "dampak") == 1

    def test_ringkasan_gate_per_zona(self, jalankan):
        kalimat = jalankan(_muat()).ringkasan_gate.kalimat
        assert "Zona Pertanian: status ITBX Diizinkan (I), intensitas memenuhi syarat." in kalimat
        assert "Zona Perumahan: status ITBX Diizinkan (I), intensitas memenuhi syarat." in kalimat


class TestRekomendasiDanFormat:
    def test_saran_multi_butir_berformat_bullet(self, jalankan):
        def melampaui(d):
            _satu_zona(d)
            tahap = d["gate_hukum"]["tahapan"]["intensitas"]  # jalur satu zona memakai tahapan.intensitas
            tahap.update(status="MELAMPAUI_BATAS", lolos=False)
            tahap["parameter"]["kdb"].update(memenuhi=False, usulan=68.36)
        daun = jalankan(_muat(ubah=melampaui)).poin[0].sub_poin[1]
        assert daun.rekomendasi.saran == "- Turunkan KDB.\n- Tambah RTH."
        assert daun.rekomendasi.poin == ["Turunkan KDB.", "Tambah RTH."]
        assert daun.rekomendasi.langkah_konkret == ["KDB harus turun ke maksimal 10%"]
        assert daun.rekomendasi.langkah_terstruktur[0].nilai_target == 10.0  # data kaya tetap dibawa

    def test_gate_hukum_status_tolak_dan_terbatas(self, jalankan):
        def x(d):
            d["gate_hukum"]["itbx_zones"][1].update(status="X")
        assert jalankan(_muat(ubah=x)).poin[0].status == "TIDAK_LOLOS"

        def t(d):
            d["gate_hukum"]["itbx_zones"][0].update(status="T")
        assert jalankan(_muat(ubah=t)).poin[0].status == "LOLOS_TERBATAS"

    def test_hidrologi_perlu_mitigasi_untuk_dampak_sedang(self, jalankan):
        def sedang(d):
            _satu_zona(d)
            d["impact_assessment"].update(impact_category="SEDANG", runoff_change_index=2.0, rekomendasi_mitigasi=None)
        out = jalankan(_muat(ubah=sedang))
        assert out.poin[1].status == "PERLU_MITIGASI" and out.rekomendasi_sistem == "Setuju Bersyarat"


class TestKamusDataLengkap:
    def test_semua_field_wajib_terisi_dan_json_valid(self, jalankan):
        out = jalankan(_muat(ubah=_satu_zona))
        data = json.loads(out.model_dump_json())
        assert data["rekomendasi_sistem"] in ("Setuju", "Setuju Bersyarat", "Lolos Terbatas", "Tolak")
        kes = data["kesimpulan"]
        assert kes["ringkasan"] and kes["ringkasan_tindak_lanjut"] == kes["langkah_berdampak"]
        assert kes["catatan_lokasi"]
        assert data["ringkasan_gate"]["kalimat"] and data["ringkasan_dampak"]["impact_category"]
        assert isinstance(data["ringkasan_dampak"]["is_kawasan_resapan"], bool) and data["ringkasan_dampak"]["poin"]
        assert data["narasi_rekomendasi"]["paragraf_gate_intensitas"] and data["narasi_rekomendasi"]["paragraf_dampak"]
        assert len(data["rekomendasi_per_zona"]) >= 1
        zona = data["rekomendasi_per_zona"][0]
        for kunci in ("nama_zona", "kode_zona", "luas_m2", "persentase", "status_itbx", "status_intensitas",
                      "status_keputusan", "rekomendasi", "pertimbangan", "saran", "alasan", "rekomendasi_poin"):
            assert kunci in zona
        for p in _semua(out.poin):
            assert p.poin_id and p.kategori and p.status and p.reasoning_pendek and p.reasoning_panjang
            assert p.sub_poin is not None

    def test_teks_tanpa_titik_koma(self, jalankan):
        teks = json.dumps(json.loads(jalankan(_muat()).model_dump_json())["ringkasan_gate"], ensure_ascii=False)
        assert ";" not in teks

    def test_advisory_model_menerima_dump_sendiri(self, jalankan):
        out = jalankan(_muat())
        assert AdvisoryL3.model_validate(out.model_dump()).rekomendasi_sistem == out.rekomendasi_sistem
