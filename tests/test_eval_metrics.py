"""Meta-test: pastikan eval/metrics.py BENERAN menangkap pelanggaran, bukan cuma selalu lulus.

Murni logic, tanpa panggilan Groq — L2Assessment/OutputL3/PoinOutput dibuat tangan.
"""

import json
from pathlib import Path

from eval.metrics import cek_faithfulness, cek_json_valid, cek_numerik, cek_sitasi_grounded
from app.retrieval.base import Chunk
from app.schemas import (
    KesimpulanOutput,
    L2Assessment,
    OutputL3,
    PoinOutput,
    RekomendasiOutput,
    RingkasanDampakOutput,
    RingkasanGateOutput,
    SitasiOutput,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _assessment() -> L2Assessment:
    payload = json.loads((FIXTURES_DIR / "l2_sample_amplop_6191.json").read_text(encoding="utf-8"))
    return L2Assessment.model_validate(payload["data"])


def _poin(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="intensitas",
        kategori="Intensitas Bangunan (KDB/KLB/KDH)",
        status="MELAMPAUI_BATAS",
        reasoning_pendek="KDB melampaui ambang maksimum.",
        reasoning_panjang="Usulan KDB melampaui ambang maksimum yang berlaku di zona ini.",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="numerik", target=60.0, saran="Kurangi proporsi luas bangunan.", disclaimer=None),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


def _output(**overrides) -> OutputL3:
    defaults = dict(
        ringkasan_gate=RingkasanGateOutput(final_gate_status="Lolos Bersyarat", decisive_stage="intensitas", kalimat="x"),
        ringkasan_dampak=RingkasanDampakOutput(impact_category=None, impact_score=None, kalimat="x"),
        poin=[_poin()],
        rekomendasi_sistem="Setuju Bersyarat",
        kesimpulan=KesimpulanOutput(langkah_berdampak=[], catatan_lokasi=None),
        catatan_global=[],
        low_confidence_keseluruhan=False,
    )
    defaults.update(overrides)
    return OutputL3(**defaults)


def _chunk(**overrides) -> Chunk:
    defaults = dict(id="chunk-1", level="tabel", teks="teks chunk", dokumen="dokumen uji", pasal=None, halaman=1)
    defaults.update(overrides)
    return Chunk(**defaults)


def _expected(**overrides) -> dict:
    defaults = dict(rekomendasi_sistem="Setuju Bersyarat", poin_id_fokus="intensitas")
    defaults.update(overrides)
    return defaults


# --- cek_faithfulness ------------------------------------------------------------------


def test_faithfulness_lulus_kasus_bersih():
    ok, _ = cek_faithfulness(_output(), _expected(status_diharapkan="MELAMPAUI_BATAS"))
    assert ok is True


def test_faithfulness_gagal_rekomendasi_sistem_salah():
    ok, detail = cek_faithfulness(_output(rekomendasi_sistem="Setuju"), _expected(rekomendasi_sistem="Setuju Bersyarat"))
    assert ok is False
    assert "rekomendasi_sistem" in detail


def test_faithfulness_gagal_status_poin_fokus_salah():
    ok, detail = cek_faithfulness(_output(), _expected(status_diharapkan="MEMENUHI_SYARAT"))
    assert ok is False
    assert "status" in detail


def test_faithfulness_gagal_frasa_terlarang_muncul():
    output = _output(poin=[_poin(reasoning_panjang="Bangunan ini sudah memenuhi seluruh standar yang berlaku.")])
    ok, detail = cek_faithfulness(output, _expected(arah_terlarang=["memenuhi seluruh standar"]))
    assert ok is False
    assert "terlarang" in detail


def test_faithfulness_lulus_frasa_tak_muncul():
    ok, _ = cek_faithfulness(_output(), _expected(arah_terlarang=["memenuhi seluruh standar"]))
    assert ok is True


def test_faithfulness_gagal_syarat_masih_generik():
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Tidak diperlukan tindakan khusus untuk poin ini."))])
    ok, detail = cek_faithfulness(output, _expected(harus_ada_syarat=True))
    assert ok is False
    assert "generik" in detail


def test_faithfulness_lulus_syarat_konkret():
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="kategorikal", saran="Wajib menyiapkan lahan parkir untuk pengunjung."))])
    ok, _ = cek_faithfulness(output, _expected(harus_ada_syarat=True))
    assert ok is True


def test_faithfulness_gagal_mitigasi_tak_disebut():
    output = _output(
        poin=[
            _poin(
                reasoning_pendek="Dampak tergolong tinggi.",
                reasoning_panjang="Dampak tergolong tinggi berdasarkan penilaian.",
                rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Perbaiki desain bangunan."),
            )
        ]
    )
    ok, detail = cek_faithfulness(output, _expected(harus_ada_mitigasi=True))
    assert ok is False
    assert "mitigasi" in detail


def test_faithfulness_lulus_mitigasi_disebut():
    output = _output(poin=[_poin(reasoning_panjang="Dampak tergolong tinggi.", rekomendasi=RekomendasiOutput(tipe="numerik-mitigasi", saran="Pertimbangkan penambahan sumur resapan."))])
    ok, _ = cek_faithfulness(output, _expected(harus_ada_mitigasi=True))
    assert ok is True


def test_faithfulness_gagal_low_confidence_tak_cocok():
    ok, detail = cek_faithfulness(_output(poin=[_poin(low_confidence=False)]), _expected(low_confidence_diharapkan=True))
    assert ok is False
    assert "low_confidence" in detail


def test_faithfulness_lulus_low_confidence_cocok():
    ok, _ = cek_faithfulness(_output(poin=[_poin(low_confidence=True)]), _expected(low_confidence_diharapkan=True))
    assert ok is True


def test_faithfulness_gagal_caveat_kosong():
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="numerik", saran="x", disclaimer=None))])
    ok, detail = cek_faithfulness(output, _expected(caveat_diharapkan=True))
    assert ok is False
    assert "caveat" in detail


def test_faithfulness_lulus_caveat_terisi():
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="numerik", saran="x", disclaimer="Catatan penting."))])
    ok, _ = cek_faithfulness(output, _expected(caveat_diharapkan=True))
    assert ok is True


# --- cek_sitasi_grounded -----------------------------------------------------------------


def test_grounded_lulus_anchor_selalu_valid_meski_tak_diretrieve():
    output = _output(poin=[_poin(sitasi=[SitasiOutput(citation_id="anchor-0", dokumen="d", pasal="", halaman=0, kutipan="k", terverifikasi=True)])])
    ok, _ = cek_sitasi_grounded(output, [], _expected())
    assert ok is True


def test_grounded_gagal_citation_id_non_anchor_karangan():
    output = _output(poin=[_poin(sitasi=[SitasiOutput(citation_id="chunk-karangan", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)])])
    ok, detail = cek_sitasi_grounded(output, [_chunk(id="chunk-asli")], _expected())
    assert ok is False
    assert "karangan" in detail.lower()


def test_grounded_gagal_citation_ids_diharapkan_tak_disitasi():
    chunk = _chunk(id="chunk-lain")
    output = _output(poin=[_poin(sitasi=[SitasiOutput(citation_id="chunk-lain", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)])])
    ok, detail = cek_sitasi_grounded(output, [chunk], _expected(citation_ids_diharapkan=["chunk-diharapkan"]))
    assert ok is False
    assert "citation_ids_diharapkan" in detail


def test_grounded_lulus_rag_kosong_sitasi_kosong():
    ok, _ = cek_sitasi_grounded(_output(poin=[_poin(sitasi=[])]), [], _expected())
    assert ok is True


def test_grounded_lulus_chunk_asli_ditemukan():
    chunk = _chunk(id="chunk-diharapkan")
    output = _output(poin=[_poin(sitasi=[SitasiOutput(citation_id="chunk-diharapkan", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)])])
    ok, _ = cek_sitasi_grounded(output, [chunk], _expected(citation_ids_diharapkan=["chunk-diharapkan"]))
    assert ok is True


# --- cek_numerik -------------------------------------------------------------------------


def test_numerik_gagal_target_beda_dari_calculator():
    assessment = _assessment()  # kdb usulan 90 / ambang_maks 60 -> target_kdb sebenarnya 60.0
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="numerik", target=999.0, saran="s"))])
    ok, detail = cek_numerik(output, _expected(), assessment)
    assert ok is False
    assert "target" in detail


def test_numerik_lulus_target_sesuai_calculator():
    assessment = _assessment()
    output = _output(poin=[_poin(rekomendasi=RekomendasiOutput(tipe="numerik", target=60.0, saran="s"))])
    ok, _ = cek_numerik(output, _expected(), assessment)
    assert ok is True


def test_numerik_gagal_non_numerik_target_tidak_none():
    assessment = _assessment()
    output = _output(poin=[_poin(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", target=5.0, saran="s"))])
    ok, detail = cek_numerik(output, _expected(poin_id_fokus="itbx"), assessment)
    assert ok is False
    assert "None" in detail


def test_numerik_lulus_non_numerik_target_none():
    assessment = _assessment()
    output = _output(poin=[_poin(poin_id="itbx", rekomendasi=RekomendasiOutput(tipe="kategorikal", target=None, saran="s"))])
    ok, _ = cek_numerik(output, _expected(poin_id_fokus="itbx"), assessment)
    assert ok is True


# --- cek_json_valid ------------------------------------------------------------------------


def test_json_valid_lulus_output_valid():
    ok, _ = cek_json_valid(_output())
    assert ok is True
