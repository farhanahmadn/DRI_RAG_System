from app.reasoning.templates import template_aman, template_low_confidence
from app.schemas import LangkahKonkretOutput, PoinKonteks


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


class TestTemplateAman:
    def test_itbx_status_dan_tipe_apa_adanya(self):
        poin = template_aman(_poin())
        assert poin.status == "I"
        assert poin.rekomendasi.tipe == "kategorikal"
        assert poin.rekomendasi.target is None
        assert poin.low_confidence is False
        assert poin.sitasi == []

    def test_intensitas_tanpa_target_menghasilkan_target_none(self):
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MEMENUHI_SYARAT",
            fakta={"target": {}},
        )
        hasil = template_aman(poin)
        assert hasil.rekomendasi.tipe == "numerik"
        assert hasil.rekomendasi.target is None

    def test_intensitas_dengan_target_terisi(self):
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdb": {"target_kdb": 60.0, "selisih": 10.0}}},
        )
        hasil = template_aman(poin)
        assert hasil.rekomendasi.target == 60.0

    def test_dampak_target_selalu_none(self):
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Sedang",
            fakta={"mitigasi": {"perlu_mitigasi": False, "arah": []}},
        )
        hasil = template_aman(poin)
        assert hasil.rekomendasi.target is None
        assert hasil.rekomendasi.langkah_konkret == []


class TestTemplateLowConfidence:
    def test_status_apa_adanya_dan_low_confidence_true(self):
        poin = _poin(status="X", fakta={"lolos": False, "reason": "Tidak ditemukan di matriks"})
        hasil = template_low_confidence(poin)
        assert hasil.status == "X"
        assert hasil.low_confidence is True
        assert hasil.sitasi == []
        assert hasil.rekomendasi.disclaimer is not None

    def test_intensitas_tetap_hitung_target_dari_fakta(self):
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={"target": {"kdh": {"target_kdh": 20.0, "selisih": 5.0}}},
        )
        hasil = template_low_confidence(poin)
        assert hasil.rekomendasi.target == 20.0

    def test_intensitas_langkah_konkret_ikut_terisi(self):
        # Bug ditemukan live (2026-08-18): poin jatuh low_confidence, target sudah benar (10) tapi
        # langkah_konkret KOSONG — template ini lupa di-wire saat langkah_konkret ditambahkan ke
        # generate_poin()/perbaiki_poin(). Ambang tetap dari kalkulator (bukan LLM), HARUSNYA selalu
        # tersedia persis seperti target, tak peduli narasi LLM gagal atau tidak.
        poin = _poin(
            poin_id="intensitas",
            kategori="Intensitas Bangunan (KDB/KLB/KDH)",
            tipe_rekomendasi="numerik",
            status="MELAMPAUI_BATAS",
            fakta={
                "parameter": {"kdb": {"usulan": 40, "satuan": "persen"}},
                "target": {"kdb": {"target_kdb": 10.0, "selisih": 30.0, "footprint_maks_m2": 85.0}},
            },
        )
        hasil = template_low_confidence(poin)
        assert hasil.rekomendasi.target == 10.0
        assert len(hasil.rekomendasi.langkah_konkret) == 1
        assert hasil.rekomendasi.langkah_konkret[0].parameter == "KDB"
        assert isinstance(hasil.rekomendasi.langkah_konkret[0], LangkahKonkretOutput)

    def test_dampak_target_dan_langkah_konkret_ikut_terisi(self):
        # Bug kedua ditemukan bersamaan: _target_untuk_poin SEBELUMNYA cuma menangani "numerik"
        # (intensitas) — dampak ("numerik-mitigasi") yang jatuh low_confidence SELALU target=None
        # walau fakta['target_mitigasi'] sebenarnya ada.
        poin = _poin(
            poin_id="dampak",
            kategori="Dampak Tata Guna Lahan",
            tipe_rekomendasi="numerik-mitigasi",
            status="Tinggi",
            fakta={
                "mitigasi": {"perlu_mitigasi": True, "arah": ["turunkan KDB"]},
                "target_mitigasi": {"runoff_change_index_maks": 2.5, "kategori_target": "Sedang", "index_saat_ini": 2.85},
            },
        )
        hasil = template_low_confidence(poin)
        assert hasil.rekomendasi.target == 2.5
        assert len(hasil.rekomendasi.langkah_konkret) == 1
        assert hasil.rekomendasi.langkah_konkret[0].parameter == "runoff_change_index"

    def test_itbx_langkah_konkret_selalu_kosong(self):
        # kategorikal (itbx) tak punya target numerik -> langkah_konkret selalu [], bukan error.
        poin = _poin(status="X", fakta={"lolos": False, "reason": "Tidak ditemukan di matriks"})
        hasil = template_low_confidence(poin)
        assert hasil.rekomendasi.langkah_konkret == []
