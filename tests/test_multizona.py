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
        assert [r.rekomendasi for r in out.rekomendasi_per_zona] == ["Setuju", "Tolak"]
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

    def test_satu_zona_tetap_satu_entri_rekomendasi_per_zona(self, pipeline):
        """Kontrak BE v1.0.0: rekomendasi_per_zona terisi minimal satu elemen, juga untuk lahan satu zona."""
        def satu(d):
            d["gate_hukum"]["itbx_zones"] = d["gate_hukum"]["itbx_zones"][:1]
        out = pipeline(_muat(satu))
        assert [p.poin_id for p in out.poin] == ["itbx", "intensitas", "dampak"]
        assert len(out.rekomendasi_per_zona) == 1
        z = out.rekomendasi_per_zona[0]
        assert (z.nama_zona, z.kode_zona, z.persentase) == ("Zona Pertanian", "P-1", 100.0)
        assert z.rekomendasi == out.rekomendasi_sistem == "Setuju"  # semua I, intensitas patuh, dampak Rendah



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
    b_tanpa_syarat_be = p.model_copy(update={"status": "B", "fakta": {}})
    assert _saring_pengisi_poin([vb], b_tanpa_syarat_be) == [vb]  # status B tanpa syarat dari BE butuh V.B
    b_dengan_syarat_be = p.model_copy(update={"status": "B", "fakta": {"keterangan_ketentuan": ["Syarat BE."]}})
    assert _saring_pengisi_poin([vb, pasal], b_dengan_syarat_be) == [pasal]  # syarat BE sudah ada: V.B tak dikirim ke LLM


class TestPembersihanTeks:
    def test_klb_tidak_diberi_persen_tapi_kdb_kdh_tetap(self):
        from app.reasoning.guardrail import _bersihkan_label_teknis as b
        assert b("ambang KDB maksimum 10 %, KLB maksimum 1 %, dan KDH minimum 88 %") == \
            "ambang KDB maksimum 10 %, KLB maksimum 1, dan KDH minimum 88 %"
        assert b("KLB 0.08 berada di bawah ambang maksimal 1 %.") == "KLB 0.08 berada di bawah ambang maksimal 1."
        assert b("KLB 0.08, dan KDH 89.31 % memenuhi") == "KLB 0.08, dan KDH 89.31 % memenuhi"

    def test_boolean_data_matriks_dibuang(self):
        from app.reasoning.guardrail import _bersihkan_label_teknis as b
        assert "False" not in b("berdasarkan data lengkap (FALLBACK_DATA_KOSONG: False), maka lolos")
        assert "False" not in b("data matriks RDTR yang lengkap (kelengkapan data matriks RDTR: False), sehingga")

    def test_reason_zona_tanpa_luas(self):
        from app.adapter import _reason_tanpa_luas
        assert _reason_tanpa_luas("Lolos karena kegiatan Diizinkan (I) di Zona Pertanian seluas 623,55 m") == \
            "Lolos karena kegiatan Diizinkan (I) di Zona Pertanian"
        p = adaptasi(_muat()).poin[0]
        assert "seluas" not in p.fakta["reason"]

    def test_prompt_dampak_tak_memuat_status_itbx_kawasan(self):
        d = adaptasi(_muat()).poin[-1]
        assert "status ITBX kegiatan di kawasan itu" not in build_user_prompt(d, [], None, None)


class TestPerbaikanCampuran:
    def test_klb_berunit_m2_dibersihkan_tapi_luas_lain_aman(self):
        from app.reasoning.guardrail import _bersihkan_label_teknis as b
        assert b("menurunkan KLB ke maksimal 1 m² dengan mengurangi luas total lantai hingga 1244 m².") == \
            "menurunkan KLB ke maksimal 1 dengan mengurangi luas total lantai hingga 1244 m²."
        assert b("KLB 0.08 dengan luas lantai 1244 m²") == "KLB 0.08 dengan luas lantai 1244 m²"

    def test_itbx_x_saran_deterministik_tanpa_karangan(self):
        from app.reasoning.generator import saran_itbx_deterministik

        def _x(d):
            d["gate_hukum"]["itbx_zones"][1].update(status="X", keterangan_ketentuan=[])
        poin = adaptasi(_muat(_x)).poin[2]
        saran = saran_itbx_deterministik(poin)
        assert "berstatus X pada Zona Perumahan" in saran and "keputusan akhir ada pada petugas" in saran
        for karangan in ("industri", "komersial", "pindah"):
            assert karangan not in saran.lower()


class TestTigaZona:
    def _tiga(self, d):
        gate = d["gate_hukum"]
        zona3 = copy.deepcopy(gate["itbx_zones"][1])
        zona3.update(nama_zona="Zona Perdagangan dan Jasa", kode_zona="K", luas_m2=100.0, persentase=8.0)
        gate["itbx_zones"].append(zona3)
        inten3 = copy.deepcopy(gate["intensitas_zones"][1])
        inten3.update(nama_zona="Zona Perdagangan dan Jasa", kode_zona="K-3", luas_irisan_m2=100.0, persentase=8.0)
        gate["intensitas_zones"].append(inten3)
        gate["keputusan_sistem_per_zona"].append(dict(gate["keputusan_sistem_per_zona"][1], nama_zona="Zona Perdagangan dan Jasa", kode_zona="K"))
        gate["total_zones"] = 3

    def test_tiga_zona_menghasilkan_tujuh_poin_dan_tiga_rekomendasi(self, pipeline):
        out = pipeline(_muat(self._tiga))
        assert [p.poin_id for p in out.poin] == [
            "itbx_P", "intensitas_P", "itbx_R", "intensitas_R", "itbx_K", "intensitas_K", "dampak"]
        assert [r.kode_zona for r in out.rekomendasi_per_zona] == ["P", "R", "K"]
        assert out.poin[5].zona.nama_zona == "Zona Perdagangan dan Jasa"


class TestKontrakBeV1:
    def test_ringkasan_dampak_membawa_label_resapan_dan_butir(self, pipeline):
        out = pipeline(_muat())
        rd = out.ringkasan_dampak
        assert rd.kawasan_label == "Kawasan Resapan Air" and rd.is_kawasan_resapan is True
        assert any("indeks perubahan limpasan" in b for b in rd.poin) and any("Koefisien limpasan (C)" in b for b in rd.poin)
        assert any("8,04% bangunan/atap dan 89,31% ruang terbuka hijau" in b for b in rd.poin)
        assert not any(";" in b for b in rd.poin)

    def test_catatan_lokasi_dari_data_spasial(self, pipeline):
        out = pipeline(_muat())
        assert out.kesimpulan.catatan_lokasi == (
            "Lokasi berada di blok I.B.3 (SWP B), termasuk kawasan resapan air, pada koridor jalan lokal.")

    def test_catatan_lokasi_menyebut_ketentuan_khusus_aktif(self, pipeline):
        def kkop(d):
            d["lokasi"]["ketentuan_khusus"]["kkop"] = True
        out = pipeline(_muat(kkop))
        assert "Ketentuan khusus yang tercatat: kawasan keselamatan operasi penerbangan (KKOP)." in out.kesimpulan.catatan_lokasi

    def test_ringkasan_dan_tindak_lanjut_kesimpulan(self, pipeline):
        out = pipeline(_muat())
        k = out.kesimpulan
        assert k.ringkasan.startswith("Permohonan direkomendasikan Setuju.") and ";" not in k.ringkasan
        assert "Zona Pertanian: ITBX Diizinkan (I), intensitas memenuhi syarat" in k.ringkasan
        assert k.ringkasan_tindak_lanjut == k.langkah_berdampak and k.langkah_berdampak

    def test_rekomendasi_per_zona_sesuai_kamus_data(self, pipeline):
        z = pipeline(_muat()).rekomendasi_per_zona[0]
        assert z.pertimbangan == "Status ITBX Diizinkan (I), intensitas memenuhi syarat." and z.rekomendasi_poin
        assert z.saran == "\n".join(f"- {b}" for b in z.rekomendasi_poin)
        assert z.status_keputusan == "LOLOS"

    def test_status_keputusan_zona_terbatas_dan_tolak(self, pipeline):
        def ubah(d):
            d["gate_hukum"]["itbx_zones"][0].update(status="T")
            d["gate_hukum"]["itbx_zones"][1].update(status="X")
        out = pipeline(_muat(ubah))
        assert [(r.rekomendasi, r.status_keputusan) for r in out.rekomendasi_per_zona] == [
            ("Lolos Terbatas", "LOLOS_TERBATAS"), ("Tolak", "TIDAK_LOLOS")]
        assert out.rekomendasi_sistem == "Tolak"


class TestRingkasanDanBoilerplate:
    def test_ringkasan_deterministik_kalimat_per_zona(self, pipeline):
        def campur(d):
            d["gate_hukum"]["itbx_zones"][1].update(status="X")
        out = pipeline(_muat(campur))
        r = out.kesimpulan.ringkasan
        assert r.startswith("Permohonan direkomendasikan Tolak. Zona Pertanian: ITBX Diizinkan (I), intensitas memenuhi syarat. ")
        assert "Zona Perumahan: ITBX X, intensitas" in r and r.endswith("Dampak hidrologi rendah.")
        assert ";" not in r

    def test_boilerplate_tidak_dievaluasi_dibuang_bila_ada_butir_lain(self):
        from app.reasoning.assemble import _butir_bermakna
        assert _butir_bermakna(["Perbaiki KDB.", "Tidak dievaluasi karena proses berhenti."]) == ["Perbaiki KDB."]
        assert _butir_bermakna(["Tidak dievaluasi karena proses berhenti."]) == ["Tidak dievaluasi karena proses berhenti."]


class TestGateEfektifDanSitasiItbx:
    def test_itbx_t_membuat_ringkasan_gate_lolos_terbatas_walau_gate_be_lolos(self, pipeline):
        def t(d):
            d["gate_hukum"]["itbx_zones"][0].update(status="T")
        out = pipeline(_muat(t))
        assert out.ringkasan_gate.final_gate_status == "Lolos Terbatas"
        assert out.ringkasan_gate.kalimat.startswith("Permohonan lolos terbatas pemeriksaan gate hukum")
        assert out.rekomendasi_sistem == "Lolos Terbatas"

    def test_itbx_x_membuat_ringkasan_gate_tidak_lolos(self, pipeline):
        def x(d):
            d["gate_hukum"]["itbx_zones"][1].update(status="X")
        assert pipeline(_muat(x)).ringkasan_gate.final_gate_status == "Tidak Lolos"

    def test_sitasi_lampiran_vb_dilampirkan_untuk_itbx_bersyarat(self):
        from app.reasoning.generator import _lengkapi_sitasi_itbx
        from app.retrieval.base import Chunk

        class _R:
            def search(self, query, filters, top_k=5, **kw):
                return [Chunk(id="rdtr-sleman-tengah-vb-p-1", level="tabel", dokumen="Perbup", pasal="",
                              teks="Lampiran V.B syarat", zona="P-1")]

        poin = adaptasi(_muat()).poin[0].model_copy(
            update={"status": "B", "fakta": {"keterangan_ketentuan": ["Syarat BE."]}})
        hasil = _lengkapi_sitasi_itbx([], poin, _R())
        assert [s.citation_id for s in hasil] == ["rdtr-sleman-tengah-vb-p-1"]
        assert _lengkapi_sitasi_itbx([], poin.model_copy(update={"status": "I"}), _R()) == []
