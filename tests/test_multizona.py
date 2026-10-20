"""Lahan multi-zona (permintaan BE 2026-10-20): ITBX dan intensitas dinilai dan dijelaskan PER ZONA, hidrologi satu."""

import copy
import json
from pathlib import Path

import pytest

from app.adapter import adaptasi, zona_terpisah
from app.reasoning import assemble as assemble_module
from app.reasoning.guardrail import DiagnosaPoin
from app.reasoning.prompts import build_user_prompt
from app.retrieval.mock import MockRetriever
from app.schemas import L2Envelope, PoinOutput, RekomendasiOutput

FIXTURE = Path(__file__).parent / "fixtures" / "l2_sample_multizona_0022.json"


def _muat(ubah=None):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if ubah:
        ubah(payload["data"])
    return L2Envelope.model_validate(payload).data


def _stub_poin(poin, retriever, assessment, *, max_retry=2):
    aman = poin.status in ("I", "MEMENUHI_SYARAT", "Rendah")
    saran = ("Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan." if aman
             else f"Perbaiki {poin.poin_id} di {poin.zona}.")
    out = PoinOutput(
        poin_id=poin.poin_id, kategori=poin.kategori, status=poin.status,
        reasoning_pendek=f"Penjelasan {poin.poin_id} untuk {poin.zona}.", reasoning_panjang="x", sitasi=[],
        rekomendasi=RekomendasiOutput(tipe=poin.tipe_rekomendasi, saran=saran), low_confidence=False)
    return out, DiagnosaPoin(poin_id=poin.poin_id, berhasil=True, percobaan=1, jumlah_chunk=1)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(assemble_module, "generate_poin_terdiagnosis", _stub_poin)
    monkeypatch.setattr(assemble_module, "log_precheck", lambda *a, **k: None)

    def _llm_gagal(*a, **k):
        raise RuntimeError("sintesis dimatikan di tes")

    monkeypatch.setattr(assemble_module.llm_client, "generate", _llm_gagal)
    return lambda a: assemble_module.jalankan_precheck(a, MockRetriever())


class TestAdapterMultiZona:
    def test_lima_poin_per_zona_plus_satu_dampak(self):
        poin = adaptasi(_muat()).poin
        assert [(p.poin_id, p.zona_info.kode_zona if p.zona_info else None) for p in poin] == [
            ("itbx", "P"), ("intensitas", "P"), ("itbx", "R"), ("intensitas", "R"), ("dampak", None)]
        assert [p.zona_subzone for p in poin[:4]] == ["P-1", "P-1", "R-3", "R-3"]
        assert poin[3].fakta["parameter"]["kdb"]["ambang_maks"] == 60  # ambang Zona Perumahan, bukan Pertanian

    def test_satu_zona_tak_berubah(self):
        def satu(d):
            d["gate_hukum"]["itbx_zones"] = d["gate_hukum"]["itbx_zones"][:1]
        a = _muat(satu)
        assert zona_terpisah(a) is None
        assert [p.poin_id for p in adaptasi(a).poin] == ["itbx", "intensitas", "dampak"]

    def test_zona_tanpa_intensitas_menjadi_tidak_dinilai(self):
        def buang(d):
            d["gate_hukum"]["intensitas_zones"] = d["gate_hukum"]["intensitas_zones"][:1]
        poin = adaptasi(_muat(buang)).poin
        assert poin[3].status == "Tidak Dinilai" and poin[3].fakta == {"dinilai": False}

    def test_prompt_menyebut_zona_tanpa_menambah_angka_luas(self):
        p = adaptasi(_muat()).poin[2]
        teks = build_user_prompt(p, [], None, None)
        assert "ZONA_YANG_DINILAI: Zona Perumahan (kode R)" in teks and "HANYA untuk zona ini" in teks
        baris_zona = next(b for b in teks.splitlines() if b.startswith("- ZONA_YANG_DINILAI"))
        assert not any(c.isdigit() for c in baris_zona)  # luas/persentase tidak ditambahkan sebagai angka baru


class TestKeluaranMultiZona:
    def test_poin_ber_identitas_zona_dan_dampak_tetap_satu(self, pipeline):
        out = pipeline(_muat())
        assert [p.poin_id for p in out.poin] == ["itbx_P", "intensitas_P", "itbx_R", "intensitas_R", "dampak"]
        assert out.poin[0].zona.nama_zona == "Zona Pertanian" and out.poin[0].zona.persentase == 50.4
        assert out.poin[2].kategori == "Klasifikasi Kegiatan (ITBX) - Zona Perumahan"
        assert out.poin[4].zona is None

    def test_dua_rekomendasi_per_zona(self, pipeline):
        out = pipeline(_muat())
        assert [(r.kode_zona, r.rekomendasi, r.status_itbx, r.status_intensitas) for r in out.rekomendasi_per_zona] == [
            ("P", "Setuju", "I", "MEMENUHI_SYARAT"), ("R", "Setuju", "I", "MEMENUHI_SYARAT")]
        assert out.rekomendasi_per_zona[0].poin_ids == ["itbx_P", "intensitas_P"]
        assert "Zona Pertanian" in out.rekomendasi_per_zona[0].alasan
        assert out.rekomendasi_per_zona[1].luas_m2 == 612.86

    def test_zona_x_menjadi_tidak_setuju_dan_zona_lain_tetap(self, pipeline):
        def ubah(d):
            d["gate_hukum"]["itbx_zones"][1]["status"] = "X"
            d["gate_hukum"]["keputusan_sistem_per_zona"][1]["status_keputusan"] = "TIDAK_LOLOS"
            d["gate_hukum"]["intensitas_zones"][1]["status"] = "TIDAK_DIEVALUASI"
            d["gate_hukum"]["final_gate_status"] = "Tidak Lolos"
        out = pipeline(_muat(ubah))
        assert [r.rekomendasi for r in out.rekomendasi_per_zona] == ["Setuju", "Tidak Setuju"]
        assert out.rekomendasi_per_zona[1].status_intensitas == "Tidak Dinilai"

    def test_catatan_global_membawa_pemberitahuan_be(self, pipeline):
        out = pipeline(_muat())
        assert any("bersinggungan dengan 2 zona RDTR" in c for c in out.catatan_global)

    def test_dampak_rendah_tak_memakai_rekomendasi_mitigasi_be(self, pipeline):
        """BE mengirim rekomendasi_mitigasi bernilai ganjil untuk dampak Rendah (lantai dasar maks 10.960 m²)."""
        out = pipeline(_muat())
        dampak = out.poin[4]
        assert "10.960" not in (dampak.rekomendasi.disclaimer or "") and dampak.rekomendasi.target is None
        assert "10.960" not in " ".join(l.deskripsi for l in dampak.rekomendasi.langkah_konkret)

    def test_fallback_sintesis_menyebut_nama_zona(self, pipeline):
        out = pipeline(_muat())
        assert "Zona Pertanian:" in out.narasi_rekomendasi.paragraf_gate_intensitas
        assert "Zona Perumahan:" in out.narasi_rekomendasi.paragraf_gate_intensitas

    def test_satu_zona_tanpa_rekomendasi_per_zona(self, pipeline):
        def satu(d):
            d["gate_hukum"]["itbx_zones"] = d["gate_hukum"]["itbx_zones"][:1]
        out = pipeline(_muat(satu))
        assert out.rekomendasi_per_zona == [] and [p.poin_id for p in out.poin] == ["itbx", "intensitas", "dampak"]


class TestKonsistensiPerZona:
    def test_cek_per_zona_tak_memakai_tahapan_agregat(self):
        from app.adapter import cek_konsistensi_poin_intensitas
        poin = adaptasi(_muat()).poin
        assert cek_konsistensi_poin_intensitas(poin[1]) == []
        rusak = poin[3].model_copy(update={"status": "MELAMPAUI_BATAS"})
        assert cek_konsistensi_poin_intensitas(rusak) == ["status=MELAMPAUI_BATAS tapi semua parameter.memenuhi=True"]


def test_dampak_rendah_tak_mengirim_target_be_ke_prompt():
    """Prompt dampak Rendah tidak boleh memuat angka mitigasi ganjil dari BE (10.960,79 m², dimensi 0 m³)."""
    d = adaptasi(_muat()).poin[-1]
    teks = build_user_prompt(d, [], None, None)
    assert "10960" not in teks and "RINCIAN_MITIGASI_KONKRET" not in teks and "TARGET_MITIGASI_KUANTITATIF" not in teks


def test_itbx_status_i_tak_membawa_syarat_lampiran_vb():
    from app.reasoning.generator import _saring_pengisi_poin
    from app.retrieval.base import Chunk
    vb = Chunk(id="rdtr-sleman-tengah-vb-r-3", level="tabel", dokumen="Perbup", pasal="", teks="Lampiran V.B syarat")
    pasal = Chunk(id="rdtr-sleman-tengah-p59", level="pasal", dokumen="Perbup", pasal="59", teks="Pasal 59")
    p = adaptasi(_muat()).poin[2]  # itbx zona R, status I
    assert _saring_pengisi_poin([vb], p) == []
    assert _saring_pengisi_poin([vb, pasal], p) == [pasal]
    b = p.model_copy(update={"status": "B"})
    assert _saring_pengisi_poin([vb], b) == [vb]  # status B membutuhkan syaratnya
