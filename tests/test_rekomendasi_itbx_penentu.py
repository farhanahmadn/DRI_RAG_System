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
