"""Pembuat payload uji L2 SINTETIS (bentuk BE 2026-10-15) untuk menguji pola keluaran LLM.

Set kedua: empat payload dengan KEGIATAN dan ANGKA berbeda dari set pertama, tetapi kombinasi dampak dan
status resapan yang sama (rendah/tinggi x resapan/non-resapan). Tujuannya: pada tingkat dampak yang sama,
apakah LLM menghasilkan pola/templat yang sama, atau jawaban yang menyesuaikan kegiatan dan angka?

PENTING: ini data SINTETIS. Kelas bangunan dan koefisien C atap, syarat ITBX, dan teks BE tiruan adalah
rekaan yang masuk akal, bukan keluaran BE. Angka dampak dihitung dengan rumus BE yang tampak pada payload
asli (C_sebelum 0,3 untuk lahan kosong; indeks = C_sesudah / C_sebelum).

Jalankan dari root repo:  python -m scripts.buat_payload_uji
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "payload_uji"
C_TAMAN = C_BEFORE = 0.3
BANDS = {"Rendah": "index < 1.5", "Sedang": "1.5-2.5", "Tinggi": "2.5-4.0", "Sangat Tinggi": "> 4.0"}
POLIGON = [[[110.332403846, -7.688919756], [110.332749952, -7.689269609], [110.332881472, -7.688933476],
            [110.332597665, -7.688775699], [110.332403846, -7.688919756]]]
KUNCI_KHUSUS = ["kawasan_resapan_air", "rawan_kekeringan", "rawan_gempa_bumi", "rawan_gunung_api",
                "rawan_banjir_lahar", "cagar_budaya", "sempadan", "kkop", "hankam", "tempat_evakuasi"]
# R-3 hierarki jalan lokal (data/parsed/v1/structured/intensitas_zona.jsonl)
AMBANG = {True: (50.0, 3.0, 30.0, "kawasan resapan air"), False: (60.0, 4.8, 25.0, "kawasan non-resapan air")}

SKENARIO = [
    dict(n=1, tinggi=False, resapan=True, blok="I.C.3", swp="C", kegiatan="Rumah Tinggal", kbli="Rumah",
         itbx="I", izin=["Rumah Tinggal"], bersyarat=[], ketentuan=[],
         kelas="Perumahan Tunggal", c_atap=0.5, L=900.0, bangunan=150.0, rth=480.0, lantai=2),
    dict(n=2, tinggi=True, resapan=True, blok="I.F.3", swp="F", kegiatan="Pergudangan", kbli="Pergudangan",
         itbx="B", izin=[], bersyarat=["Pergudangan"],
         ketentuan=["Pergudangan hanya untuk menunjang kebutuhan masyarakat sekitar.",
                    "Luas keseluruhan kegiatan pergudangan maksimal 30% dari luas blok.",
                    "Kendaraan pengangkut dibatasi kendaraan ringan."],
         kelas="Industri dan Pergudangan", c_atap=0.9, L=2400.0, bangunan=2000.0, rth=100.0, lantai=2),
    dict(n=3, tinggi=False, resapan=False, blok="I.D.2", swp="D", kegiatan="Penyediaan Makan Minum",
         kbli="Penyediaan", itbx="B", izin=[], bersyarat=["Penyediaan Makan Minum"],
         ketentuan=["Jam operasional dibatasi maksimal pukul 22.00.",
                    "Wajib mengelola limbah dan sampah secara mandiri."],
         kelas="Perdagangan dan Jasa Lingkungan", c_atap=0.7, L=650.0, bangunan=140.0, rth=260.0, lantai=2),
    dict(n=4, tinggi=True, resapan=False, blok="I.A.4", swp="A", kegiatan="Perdagangan Eceran Minimarket",
         kbli="Perdagangan", itbx="B", izin=[], bersyarat=["Perdagangan Eceran Minimarket"],
         ketentuan=["Hanya melayani kebutuhan masyarakat sekitar.", "Luas lantai penjualan maksimal 200 m2."],
         kelas="Perdagangan Pusat Kota", c_atap=0.85, L=1100.0, bangunan=980.0, rth=40.0, lantai=2),
]


def f(x, n=3):
    return round(x, n)


def fmt_id(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def bangun(sk: dict) -> tuple[str, dict]:
    tinggi, resapan = sk["tinggi"], sk["resapan"]
    L, b, r, lantai, c_atap = sk["L"], sk["bangunan"], sk["rth"], sk["lantai"], sk["c_atap"]
    permohonan = b + r
    sisa = L - permohonan
    p_sisa, p_perm = sisa / L, permohonan / L
    p_terbangun, p_taman = b / permohonan, r / permohonan
    c_after = p_sisa * C_TAMAN + p_perm * (p_terbangun * c_atap + p_taman * C_TAMAN)
    idx = c_after / C_BEFORE
    kategori = "TINGGI" if tinggi else "RENDAH"
    assert (2.5 <= idx < 4.0) if tinggi else (idx < 1.5), (sk["n"], idx)

    kdb, kdh, klb = b / L * 100, r / L * 100, b * lantai / L
    amb_kdb, amb_klb, amb_kdh, label = AMBANG[resapan]
    kdb_ok, klb_ok, kdh_ok = kdb <= amb_kdb, klb <= amb_klb, kdh >= amb_kdh
    lolos_int = kdb_ok and klb_ok and kdh_ok
    alasan = None if lolos_int else "Melanggar Aturan: " + ", ".join(x for x in (
        None if kdb_ok else f"KDB ({kdb:.2f}%) melebihi batas maksimal Zona Perumahan ({amb_kdb:.2f}%)",
        None if kdh_ok else f"KDH ({kdh:.2f}%) kurang dari batas minimal Zona Perumahan ({amb_kdh:.2f}%)",
    ) if x)
    parameter = {
        "kdb": {"usulan": f(kdb, 2), "ambang_maks": amb_kdb, "memenuhi": kdb_ok, "satuan": "persen"},
        "klb": {"usulan": f(klb, 2), "ambang_maks": amb_klb, "memenuhi": klb_ok, "satuan": "rasio"},
        "kdh": {"usulan": f(kdh, 2), "ambang_min": amb_kdh, "memenuhi": kdh_ok, "satuan": "persen"},
    }
    kutipan = f"KDB maks {amb_kdb:.0f}%, KLB maks {amb_klb}, KDH min {amb_kdh:.0f}% ({label})"
    dasar_int = [{"dokumen": "RDTR Sleman", "pasal": "Ketentuan Intensitas Ruang (R-3)", "kutipan": kutipan}]
    status_int = "MEMENUHI_SYARAT" if lolos_int else "MELAMPAUI_BATAS"

    if lolos_int:
        final, rek, status_keputusan = ("Lolos", "Setuju", "LOLOS") if sk["itbx"] == "I" else (
            "Lolos Bersyarat", "Setuju Bersyarat", "LOLOS_BERSYARAT")
    else:
        final, rek, status_keputusan = "Tidak Lolos", "Tidak Setuju", "TIDAK_LOLOS"

    mitigasi = None
    if tinggi:
        b_maks = min(L, L * (0.75 - C_BEFORE) / (c_atap - C_BEFORE))
        r_min = L - b_maks
        kdb_m, kdh_m = round(b_maks / L * 100), 100 - round(b_maks / L * 100)
        dim = (c_after - 2.5 * C_BEFORE) * 0.013659 * b / (5 * 3.14 * 0.086)
        mitigasi = {
            "status_dampak": "TINGGI", "target_kategori": "SEDANG", "target_indeks_maks": 2.5, "c_target_sedang": 0.75,
            "dimensi_minimum": {
                "nilai": f(dim, 2), "satuan": "m³",
                "rumus": "(C_sesudah - 2.5 x C_sebelum) x 0.013659 x Luas Lahan terbangun / (5 x 3.14 x 0.086)",
                "parameter": {"c_sesudah": f(c_after), "c_sebelum": C_BEFORE, "c_target_sedang": 0.75,
                              "luas_lahan_terbangun_m2": b, "faktor_pembilang": 0.013659, "faktor_penyebut": 1.3502}},
            "rekomendasi_penyesuaian_lahan": {
                "luas_bangunan_maks_m2": f(b_maks, 2), "luas_rth_min_m2": f(r_min, 2),
                "kdb_maks_persen": kdb_m, "kdh_min_persen": kdh_m,
                "catatan": (f"Agar indeks runoff turun ke <= 2.5 (kategori Sedang), luas bangunan maksimal adalah "
                            f"{fmt_id(b_maks)} m² dan RTH minimal {fmt_id(r_min)} m².")},
            "saran": (f"Untuk menurunkan dampak dari TINGGI menjadi SEDANG (indeks <= 2.5): pemohon disarankan "
                      f"menyesuaikan luas lantai dasar bangunan menjadi maksimal {fmt_id(b_maks)} m² "
                      f"(KDB maks {kdb_m}%) dan menyediakan RTH minimal {fmt_id(r_min)} m² (KDH min {kdh_m}%), "
                      f"atau menyediakan fasilitas sumur/kolam resapan air hujan dengan dimensi kapasitas "
                      f"minimum {f(dim, 2)} m³."),
        }

    nama_label = "Kawasan Resapan Air" if resapan else "Kawasan Non-Resapan Air"
    kalimat = (f"Rencana kegiatan berpotensi meningkatkan dampak banjir lokal sebesar {f(idx)}x dengan kategori "
               f"dampak {kategori.capitalize()} berdasarkan rumus perhitungan limpasan pada {nama_label}")
    dasar_reg = []
    if resapan:
        dasar_reg.append({"judul": "Peraturan Bupati Sleman No. 80 Tahun 2023 (RDTR Sleman)", "pasal": "Pasal 53",
                          "deskripsi": "Ketentuan khusus zona resapan air, kewajiban penerapan zero delta Q policy "
                                       "(debit limpasan tidak bertambah), dan larangan kegiatan yang menyebabkan "
                                       "pencemaran air tanah.", "kategori": "Regulasi Daerah"})
    dasar_reg.append({"judul": "FHWA HEC-22 Urban Drainage Design Manual, 4th Edition",
                      "pasal": "Section 3.2.2.1 Runoff Coefficient",
                      "deskripsi": "Metode perhitungan komposit koefisien limpasan (C) dan pereduksian debit puncak.",
                      "kategori": "Standar Teknis Internasional"})
    label_perm = {"kawasan_hidrologi": "Zona Perumahan", "kode_kawasan": "R", "status_kawasan": sk["itbx"]}
    atap_info = {"luas_m2": b, "proporsi": f(b / L), "proporsi_dlm_permohonan": f(p_terbangun), "C": c_atap,
                 "kelas": sk["kelas"]}

    itbx = {
        "nama_zona": "Zona Perumahan", "kode_zona": "R", "status": sk["itbx"], "lolos": True,
        "kbli_diusulkan": sk["kbli"], "kegiatan_diusulkan": sk["kegiatan"],
        "kegiatan_diizinkan": sk["izin"], "kegiatan_terbatas": [], "kegiatan_bersyarat": sk["bersyarat"],
        "keterangan_ketentuan": sk["ketentuan"],
        "reason": ("Lolos karena kegiatan Diizinkan (I) di zona Zona Perumahan" if sk["itbx"] == "I"
                   else "Lolos Bersyarat (B) di Zona Perumahan"),
        "dasar_hukum": [{"dokumen": "RDTR Sleman", "pasal": "Matriks ITBX (R-3)", "kutipan": "Data KBLI referensi"}],
    }
    itbx_tanpa_nama = {k: v for k, v in itbx.items() if k not in ("nama_zona", "kode_zona")}
    inten = {
        "status": status_int, "lolos": lolos_int, "nama_zona": "Zona Perumahan", "kode_zona": "R-3",
        "hierarki_jalan": "lokal", "kawasan_resapan": resapan, "zona_evaluasi": "Zona Perumahan",
        "keterangan_kawasan": "Evaluasi intensitas mencakup seluruh zona permohonan.",
        "parameter": parameter, "luas_tapak_m2": b, "jumlah_lantai": lantai, "luas_rth_usulan_m2": r,
        "reason": alasan, "dasar_hukum": dasar_int,
    }
    nama = f"APP-2026-UJI2-{sk['n']}-{'TINGGI' if tinggi else 'RENDAH'}-{'RESAPAN' if resapan else 'NONRESAPAN'}"
    data = {
        "application_id": str(uuid.uuid4()), "application_number": nama,
        "is_revision": False, "previous_decision": None, "previous_decision_raw": None,
        "previous_decided_at": None, "revision_count": 0, "riwayat_revisi": [],
        "pemohon": {
            "nama": "Penguji Sistem", "email": "penguji@example.test", "kontak": "+620000000000",
            "alamat": "Alamat uji, Kalurahan Caturharjo, Kapanewon Sleman, Sleman",
            "status": "RISK_ANALYSIS", "tanggal_pengajuan": {},
            "usulan_teknis": {"luas_tanah": L, "luas_bangunan": b, "luas_rth": r, "jumlah_lantai": lantai},
        },
        "keputusan": {"ada": False, "final_decision": "MENUNGGU KEPUTUSAN", "kesimpulan_kesesuaian": "MENUNGGU_KEPUTUSAN"},
        "timestamp": "2026-10-16T05:00:00.000Z",
        "lokasi": {
            "koordinat": {"lat": -7.688990626701707, "lon": 110.33266290793831},
            "geojson": {"type": "Polygon", "coordinates": POLIGON},
            "rdtr_zone": "Zona Perumahan", "rdtr_subzone": "R-3", "rdtr_subzone_source": "geojson_overlay",
            "swp": sk["swp"], "blok": sk["blok"], "blok_list": [sk["blok"]], "kawasan_resapan": resapan,
            "ketentuan_khusus": {k: (resapan if k == "kawasan_resapan_air" else False) for k in KUNCI_KHUSUS},
            "luas_lahan_m2": L,
        },
        "gate_hukum": {
            "final_gate_status": final, "rekomendasi_sistem": rek, "decisive_stage": "intensitas",
            "conflict_flag": False, "is_multi_zone": False, "total_zones": 1, "pemberitahuan": None,
            "keputusan_sistem_per_zona": [{
                "nama_zona": "Zona Perumahan", "kode_zona": "R", "luas_m2": L, "persentase": 100,
                "status_keputusan": status_keputusan, "rekomendasi_sistem": rek, "itbx_status": sk["itbx"],
                "intensitas_status": status_int, "catatan": f"Kegiatan ({sk['itbx']}) di Zona Perumahan."}],
            "itbx_zones": [{"nama_zona": "Zona Perumahan", "kode_zona": "R", "luas_m2": L, "persentase": 100,
                            "status": sk["itbx"], "status_keputusan": status_keputusan,
                            "rekomendasi_sistem": rek, **itbx_tanpa_nama}],
            "intensitas_zones": [{"nama_zona": "Zona Perumahan", "kode_zona": "R-3", "hierarki_jalan": "lokal",
                                  "kawasan_resapan": resapan, "luas_irisan_m2": L, "persentase": 100,
                                  "status": status_int, "lolos": lolos_int, "parameter": parameter,
                                  "luas_tapak_m2": b, "jumlah_lantai": lantai, "luas_rth_usulan_m2": r,
                                  "reason": alasan, "dasar_hukum": dasar_int}],
            "tahapan": {"itbx": itbx,
                        "lp2b": {"status": "AMAN", "lolos": True, "reason": None},
                        "sempadan": {"status": "AMAN", "lolos": True, "reason": None},
                        "intensitas": inten},
        },
        "impact_assessment": {
            "dinilai": True, **label_perm, "is_kawasan_resapan": resapan, "kawasan_label": nama_label,
            "c_before": C_BEFORE, "c_after": f(c_after), "delta_c": f(c_after - C_BEFORE),
            "runoff_change_index": f(idx), "impact_category": kategori,
            "calculation_details": (f"C_Sebelum = {C_BEFORE}; C_Sesudah = {f(c_after)} "
                                    f"(Luas Persil: {L} m2, Permohonan: {permohonan:.0f} m2)"),
            "threshold_bands": BANDS,
            "reasoning_pendek": kalimat + ".",
            "reasoning_panjang": (f"Berdasarkan rumus perhitungan hidrologi (perubahan koefisien C dari {C_BEFORE} "
                                  f"menjadi {f(c_after)}), kegiatan ini berpotensi meningkatkan dampak banjir lokal "
                                  f"sebesar {f(idx)}x dengan kategori dampak {kategori.capitalize()} ()."),
            "ringkasan_dampak": kalimat + (" (perlu mitigasi)." if tinggi else " (kategori aman)."),
            "rekomendasi_mitigasi": mitigasi, "dasar_regulasi": dasar_reg,
            "existing_surface_details": {"kelas": "Daerah tak Terbangun / RTH", **label_perm, "luas_persil_m2": L,
                                         "luas_persil_source": "land_area", "c_before_source": "rth_baseline_taman",
                                         "proporsi": 1, "C": C_BEFORE},
            "proposed_surface_details": {**label_perm, "Bangunan/Atap": f(b / L * 100, 1),
                                         "Taman/RTH": f(r / L * 100, 1), "Sisa Persil Eksisting": f(p_sisa * 100, 1)},
            "detailed_surface_breakdown": {
                **label_perm, "luas_persil_m2": L, "luas_persil_source": "land_area",
                "luas_permohonan_m2": permohonan, "luas_sisa_m2": sisa, "zona_x_diexclude": False,
                "proporsi_sisa": f(p_sisa), "proporsi_permohonan": f(p_perm), "proporsi_terbangun": f(p_terbangun),
                "proporsi_taman": f(p_taman), "is_persil_found": False, "is_multi_persil": False, "persil_count": 0,
                "persil_unioned": False, "luas_usulan_melebihi_persil": False,
                "kdb_usulan_persen": f(kdb, 2), "kdh_usulan_persen": f(kdh, 2),
                "Bangunan/Atap": atap_info,
                "Taman/RTH": {"luas_m2": r, "proporsi": f(r / L), "proporsi_dlm_permohonan": f(p_taman),
                              "C": C_TAMAN, "kelas": "Taman / RTH"},
                "Sisa Persil Eksisting": {"luas_m2": sisa, "proporsi": f(p_sisa), "C": C_TAMAN,
                                          "kelas": "Daerah tak Terbangun / RTH"}},
            "c_coefficients": {
                "Kawasan_Hidrologi": "Zona Perumahan", "Kode_Kawasan": "R", "C_Atap_KBLI": c_atap,
                "Kelas_Atap": sk["kelas"], "C_Taman": C_TAMAN, "C_Sisa_Persil": C_TAMAN,
                "Kelas_Eksisting": "Daerah tak Terbangun / RTH", "Luas_Persil_m2": L, "Luas_Permohonan_m2": permohonan,
                "Luas_Sisa_m2": sisa, "Proporsi_Sisa": f(p_sisa), "Proporsi_Permohonan": f(p_perm),
                "Proporsi_Terbangun": f(p_terbangun), "Proporsi_Taman": f(p_taman),
                "Luas_Persil_Source": "land_area", "C_Before_Source": "rth_baseline_taman"},
            "luas_lahan_m2": L, "luas_persil_source": "land_area", "c_before_source": "rth_baseline_taman",
            "luas_usulan_melebihi_persil": False, "data_confidence": "High", "limitations": None, "multi_persil": False,
        },
    }
    berkas = f"hidrologi_set2_{sk['n']}_{'tinggi' if tinggi else 'rendah'}_{'resapan' if resapan else 'nonresapan'}.json"
    return berkas, {"statusCode": 200, "message": "L2 Spatial Risk Assessment completed successfully", "data": data}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sk in SKENARIO:
        berkas, payload = bangun(sk)
        (OUT / berkas).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        ia = payload["data"]["impact_assessment"]
        gi = payload["data"]["gate_hukum"]["tahapan"]["intensitas"]
        print(f"{berkas:<52} {sk['kegiatan']:<32} indeks={ia['runoff_change_index']:<6} {ia['impact_category']:<7} "
              f"intensitas={gi['status']}")


if __name__ == "__main__":
    main()
