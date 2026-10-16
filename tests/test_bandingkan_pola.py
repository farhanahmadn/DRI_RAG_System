import json

from scripts.bandingkan_pola_keluaran import (
    bandingkan,
    jaccard,
    laporan,
    muat_terakhir,
    sel_dari_nomor,
    trigram,
)


def _resp(dampak_panjang, saran="Pertahankan RTH."):
    return {"poin": [
        {"poin_id": "itbx", "reasoning_panjang": "Kegiatan bersyarat di zona perumahan.", "rekomendasi": {"saran": "x"}},
        {"poin_id": "intensitas", "reasoning_panjang": "Intensitas memenuhi ambang.", "rekomendasi": {"saran": "y"}},
        {"poin_id": "dampak", "reasoning_pendek": "Dampak rendah.", "reasoning_panjang": dampak_panjang,
         "rekomendasi": {"saran": saran}}]}


def test_angka_diabaikan_agar_kerangka_yang_dibandingkan():
    assert trigram("KDB 20.89% di bawah 50%") == trigram("KDB 33,5% di bawah 60%")


def test_jaccard():
    assert jaccard({1, 2}, {1, 2}) == 1.0 and jaccard({1}, {2}) == 0.0 and jaccard(set(), set()) == 0.0


def test_sel_dari_nomor():
    assert sel_dari_nomor("APP-2026-UJI2-3-RENDAH-NONRESAPAN") == ("UJI2", "RENDAH", "NONRESAPAN")


def _log(tmp_path, entri):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps({"request": {"application_number": n}, "response": r}) for n, r in entri),
                 encoding="utf-8")
    return p


def test_hasil_terakhir_per_nomor_menang_dan_nomor_lain_diabaikan(tmp_path):
    p = _log(tmp_path, [("APP-2026-UJI-1-RENDAH-RESAPAN", _resp("lama")),
                        ("APP-2026-6191", _resp("diabaikan")),
                        ("APP-2026-UJI-1-RENDAH-RESAPAN", _resp("baru"))])
    hasil = muat_terakhir(p)
    assert list(hasil) == ["APP-2026-UJI-1-RENDAH-RESAPAN"]
    assert hasil["APP-2026-UJI-1-RENDAH-RESAPAN"]["poin"][2]["reasoning_panjang"] == "baru"


def test_templat_terdeteksi_bila_sel_sama_jauh_lebih_mirip(tmp_path):
    kerangka = "dampak hidrologi rendah karena indeks limpasan air rendah dari luas bangunan dan rth yang diusulkan"
    unik = "sesuatu yang sama sekali berbeda mengenai tampungan genangan banjir dan drainase kawasan permukiman"
    entri = [
        ("APP-2026-UJI-1-RENDAH-RESAPAN", _resp(kerangka + " 800 m2")),
        ("APP-2026-UJI-2-TINGGI-RESAPAN", _resp(unik)),
        ("APP-2026-UJI2-1-RENDAH-RESAPAN", _resp(kerangka + " 450 m2")),
        ("APP-2026-UJI2-2-TINGGI-RESAPAN", _resp(unik + " tambahan")),
    ]
    hasil = muat_terakhir(_log(tmp_path, entri))
    v = bandingkan(hasil)[("dampak", "reasoning_panjang")]
    assert sum(v["sama"]) / len(v["sama"]) > sum(v["beda"]) / len(v["beda"])
    assert "TEMPLAT" in laporan(hasil)


def test_laporan_butuh_dua_set(tmp_path):
    hasil = muat_terakhir(_log(tmp_path, [("APP-2026-UJI-1-RENDAH-RESAPAN", _resp("x"))]))
    assert "Perlu hasil KEDUA set" in laporan(hasil)
