"""Meta-test: pastikan eval/metrics.py BENERAN menangkap pelanggaran, bukan cuma selalu lulus.

Murni logic, tanpa panggilan Groq — PoinOutput dibuat tangan, bukan lewat LLM.
"""

from eval.metrics import cek_faithfulness, cek_json_valid, cek_numerik, cek_sitasi_grounded
from app.retrieval.base import Chunk
from app.schemas import (
    IndikatorJejak,
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RekomendasiOutput,
    RingkasanOutput,
    SitasiOutput,
)


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="X-01",
        kategori="KDH",
        bobot=10.0,
        skor=10.0,
        kontribusi=10.0,
        nilai_input=6.0,
        ambang=10.0,
        operator=">=",
        formula="",
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def _poin(**overrides) -> PoinOutput:
    defaults = dict(
        poin_id="X-01",
        kategori="KDH",
        status="Tidak Aman",
        kontribusi=10.0,
        reasoning_pendek="Nilai KDH kurang dari ambang minimum.",
        reasoning_panjang="Nilai KDH aktual berada di bawah ambang minimum yang ditetapkan.",
        sitasi=[],
        rekomendasi=RekomendasiOutput(tipe="numerik", target=10.0, saran="Tingkatkan area hijau.", disclaimer=None),
        low_confidence=False,
    )
    defaults.update(overrides)
    return PoinOutput(**defaults)


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="chunk-1",
        level="tabel",
        teks="teks chunk",
        dokumen="dokumen uji",
        pasal=None,
        halaman=1,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


# --- cek_faithfulness ------------------------------------------------------------------


def test_faithfulness_lulus_kasus_bersih():
    poin = _poin(status="Tidak Aman")
    ok, _ = cek_faithfulness(poin, {"status": "Tidak Aman", "arah_terlarang": "melebihi"})
    assert ok is True


def test_faithfulness_gagal_status_salah():
    poin = _poin(status="Aman")
    ok, detail = cek_faithfulness(poin, {"status": "Tidak Aman", "arah_terlarang": None})
    assert ok is False
    assert "status" in detail


def test_faithfulness_gagal_kata_terlarang_muncul():
    poin = _poin(
        reasoning_panjang="Nilai KDH aktual melebihi ambang yang seharusnya."
    )
    ok, detail = cek_faithfulness(poin, {"status": "Tidak Aman", "arah_terlarang": "melebihi"})
    assert ok is False
    assert "terlarang" in detail


def test_faithfulness_lulus_kata_terlarang_sbg_bagian_kata_lain():
    # "kurang" sbg substring "kurangi"/"pengurangan" TIDAK boleh dianggap pelanggaran — itu saran
    # yang benar (mis. "kurangi luas bangunan" utk kasus melebihi), bukan klaim arah yang salah.
    poin = _poin(
        reasoning_pendek="Nilai KDB melebihi ambang.",
        reasoning_panjang="KDB aktual melebihi batas maksimum zona ini.",
        rekomendasi=RekomendasiOutput(
            tipe="numerik", target=80.0, saran="Kurangi luas bangunan agar sesuai ambang.", disclaimer=None
        ),
    )
    ok, _ = cek_faithfulness(poin, {"status": "Tidak Aman", "arah_terlarang": "kurang"})
    assert ok is True


def test_faithfulness_lulus_saran_boleh_pakai_kata_arah_berlawanan():
    # Kata arah terlarang di SARAN (bukan reasoning) tidak dihitung — saran wajar berisi instruksi
    # arah berlawanan (mis. "tingkatkan" utk kasus kurang dari minimum).
    poin = _poin(
        reasoning_pendek="Nilai KDH kurang dari ambang minimum.",
        reasoning_panjang="KDH aktual berada di bawah ambang minimum yang ditetapkan.",
        rekomendasi=RekomendasiOutput(
            tipe="numerik", target=10.0, saran="Area hijau melebihi target akan lebih baik.", disclaimer=None
        ),
    )
    ok, _ = cek_faithfulness(poin, {"status": "Tidak Aman", "arah_terlarang": "melebihi"})
    assert ok is True


# --- cek_sitasi_grounded -----------------------------------------------------------------


def test_grounded_gagal_chunks_kosong_tapi_sitasi_terisi():
    poin = _poin(
        sitasi=[
            SitasiOutput(citation_id="chunk-1", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)
        ]
    )
    ok, detail = cek_sitasi_grounded(poin, [], {"citation_ids_diharapkan": []})
    assert ok is False
    assert "kosong" in detail.lower()


def test_grounded_gagal_citation_id_tidak_ada_di_chunks():
    chunk = _chunk(id="chunk-asli")
    poin = _poin(
        sitasi=[
            SitasiOutput(citation_id="chunk-karangan", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)
        ]
    )
    ok, detail = cek_sitasi_grounded(poin, [chunk], {"citation_ids_diharapkan": []})
    assert ok is False
    assert "karangan" in detail.lower()


def test_grounded_gagal_tidak_ada_citation_diharapkan_yang_disitasi():
    chunk = _chunk(id="chunk-lain")
    poin = _poin(
        sitasi=[
            SitasiOutput(citation_id="chunk-lain", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)
        ]
    )
    ok, detail = cek_sitasi_grounded(poin, [chunk], {"citation_ids_diharapkan": ["chunk-yang-diharapkan"]})
    assert ok is False
    assert "citation_ids_diharapkan" in detail


def test_grounded_lulus_kasus_bersih():
    chunk = _chunk(id="chunk-diharapkan")
    poin = _poin(
        sitasi=[
            SitasiOutput(citation_id="chunk-diharapkan", dokumen="d", pasal="", halaman=1, kutipan="k", terverifikasi=True)
        ]
    )
    ok, _ = cek_sitasi_grounded(poin, [chunk], {"citation_ids_diharapkan": ["chunk-diharapkan"]})
    assert ok is True


def test_grounded_lulus_rag_kosong_sitasi_kosong():
    poin = _poin(sitasi=[])
    ok, _ = cek_sitasi_grounded(poin, [], {"citation_ids_diharapkan": []})
    assert ok is True


# --- cek_numerik -------------------------------------------------------------------------


def test_numerik_gagal_target_beda_dari_calculator():
    indikator = _indikator(kategori="KDH", nilai_input=6.0, ambang=10.0, operator=">=")
    poin = _poin(rekomendasi=RekomendasiOutput(tipe="numerik", target=999.0, saran="s", disclaimer=None))
    ok, detail = cek_numerik(poin, indikator)
    assert ok is False
    assert "target" in detail


def test_numerik_lulus_target_sesuai_calculator():
    indikator = _indikator(kategori="KDH", nilai_input=6.0, ambang=10.0, operator=">=")
    poin = _poin(rekomendasi=RekomendasiOutput(tipe="numerik", target=10.0, saran="s", disclaimer=None))
    ok, _ = cek_numerik(poin, indikator)
    assert ok is True


def test_numerik_lulus_non_numerik_none_sama_dengan_none():
    indikator = _indikator(kategori="Lokasional LP2B", nilai_input="dalam_lp2b", ambang="tidak_dalam_lp2b", operator="==")
    poin = _poin(rekomendasi=RekomendasiOutput(tipe="lokasional", target=None, saran="s", disclaimer=None))
    ok, _ = cek_numerik(poin, indikator)
    assert ok is True


# --- cek_json_valid ------------------------------------------------------------------------


def test_json_valid_lulus_output_valid():
    output = OutputPreCheck(
        ringkasan=RingkasanOutput(skor_total=10.0, level="Sedang", kalimat="kalimat"),
        poin=[_poin()],
        kesimpulan=KesimpulanOutput(langkah_berdampak=[], catatan_lokasi=None),
    )
    ok, _ = cek_json_valid(output)
    assert ok is True
