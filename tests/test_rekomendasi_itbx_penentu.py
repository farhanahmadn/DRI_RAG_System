from app.reasoning.rekomendasi import gate_efektif, turunkan_rekomendasi


def test_itbx_i_mengalahkan_label_lolos_bersyarat():
    assert gate_efektif("Lolos Bersyarat", "I") == "Lolos"
    assert turunkan_rekomendasi("Lolos Bersyarat", "Rendah", "I") == "Setuju"


def test_itbx_i_tetap_bersyarat_bila_dampak_tinggi():
    assert turunkan_rekomendasi("Lolos Bersyarat", "Tinggi", "I") == "Setuju Bersyarat"


def test_itbx_b_tetap_bersyarat_dan_tx_tetap():
    assert turunkan_rekomendasi("Lolos Bersyarat", "Rendah", "B") == "Setuju Bersyarat"
    assert turunkan_rekomendasi("Lolos", "Rendah", "T") == "Lolos Terbatas"
    assert turunkan_rekomendasi("Lolos", "Rendah", "X") == "Tolak"


def test_tanpa_status_itbx_gate_apa_adanya():
    assert gate_efektif("Lolos Bersyarat", None) == "Lolos Bersyarat"


def test_zona_x_tidak_dievaluasi_parameter_null_diterima():
    from app.schemas import IntensitasZona
    z = IntensitasZona(status="TIDAK_DIEVALUASI", parameter=None)
    assert z.parameter == {}


def test_scrub_satuan_klb_pada_kalimat_pembanding():
    from app.reasoning.guardrail import _bersihkan_label_teknis
    t = _bersihkan_label_teknis("KLB usulan tidak lebih dari 1 m² dan KLB lebih tinggi 0,37 m² dari ambang.")
    assert "m²" not in t and "tidak lebih dari 1" in t and "lebih tinggi 0,37" in t


def test_intensitas_tidak_dinilai_tanpa_llm():
    from app.reasoning.guardrail import generate_poin_terdiagnosis
    from app.schemas import PoinKonteks
    p = PoinKonteks(poin_id="intensitas", kategori="K", tipe_rekomendasi="numerik", status="Tidak Dinilai",
                    fakta={"dinilai": False}, zona="Zona Perumahan")
    hasil, diag = generate_poin_terdiagnosis(p, retriever=None, assessment=None)
    assert hasil.sitasi == [] and not hasil.low_confidence and diag.berhasil
    assert "Pasal" not in hasil.reasoning_panjang and "Lampiran" not in hasil.reasoning_panjang
