from app.reasoning.templates import template_aman, template_low_confidence
from app.schemas import PoinKonteks


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
