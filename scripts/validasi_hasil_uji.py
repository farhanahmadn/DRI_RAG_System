"""Validasi otomatis hasil uji L3 terhadap PAYLOAD MASUKAN dan dokumen RDTR.

Membaca folder hasil (keluaran `scripts.jalankan_uji_semua`), mencocokkan tiap hasil dengan payload asalnya
lewat nomor permohonan, lalu menjalankan pemeriksaan DETERMINISTIK yang tidak bergantung pada penilaian
subjektif: kebenaran status dan angka, konsistensi resapan vs non-resapan, sitasi pengisi, gaya penulisan,
dan klaim terlarang. Pembacaan makna (apakah penjelasan masuk akal) tetap perlu dilakukan manusia.

Juga membaca logs/precheck.jsonl untuk jumlah percobaan, penolakan guardrail, dan penyebab fallback sintesis.

Jalankan dari root repo:  python -m scripts.validasi_hasil_uji [folder_hasil]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "precheck.jsonl"
SUMBER_PAYLOAD = [ROOT / "tests" / "payload_uji", ROOT / "tests" / "fixtures"]

_ENUM = re.compile(r"\b(?:MELAMPAUI_BATAS|MEMENUHI_SYARAT|TIDAK_DIEVALUASI)\b")
_BOLD_GANJIL = lambda t: t.count("**") % 2 == 1  # noqa: E731
_NEGASI = re.compile(r"tidak\s+(?:berlaku|mengikat|diwajibkan|termasuk|berada)|bukan\s+(?:merupakan\s+)?(?:kawasan|kewajiban)"
                     r"|di\s+luar\s+(?:kawasan|blok)", re.I)
_PISAH = re.compile(r"(?<=[.!?])\s+|\n+")


def fmt_id(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def semua_teks(poin: dict) -> str:
    r = poin.get("rekomendasi") or {}
    return " ".join(str(x or "") for x in (poin.get("reasoning_pendek"), poin.get("reasoning_panjang"),
                                           r.get("saran"), r.get("disclaimer")))


def muat_payload() -> dict[str, dict]:
    hasil = {}
    for folder in SUMBER_PAYLOAD:
        for berkas in folder.glob("*.json"):
            try:
                data = json.loads(berkas.read_text(encoding="utf-8")).get("data") or {}
            except (ValueError, OSError):
                continue
            if data.get("application_number"):
                hasil[data["application_number"]] = data
    return hasil


def muat_log() -> dict[str, dict]:
    hasil: dict[str, dict] = {}
    if LOG.exists():
        for baris in LOG.read_text(encoding="utf-8").splitlines():
            if baris.strip():
                rec = json.loads(baris)
                nomor = (rec.get("request") or {}).get("application_number")
                if nomor:
                    hasil[nomor] = rec
    return hasil


def periksa(nomor: str, hasil: dict, data: dict, rekaman: dict | None) -> list[tuple[str, str, str]]:
    """Daftar (kode, 'OK'|'GAGAL'|'INFO', pesan)."""
    out: list[tuple[str, str, str]] = []

    def cek(kode, ok, pesan_ok, pesan_gagal):
        # Poin yang jatuh ke fallback (low_confidence) bukan keluaran LLM: pemeriksaan ISI-nya tidak adil dan
        # dilewati (penyebabnya dilaporkan di INFO). Pemeriksaan angka deterministik (V1, V2) tetap jalan.
        poin_untuk = {"V3": "dampak", "V4": "dampak", "V5.dampak": "dampak", "V8.reasoning": "dampak",
                      "V6": "itbx", "V5.itbx": "itbx", "V5.intensitas": "intensitas"}
        for awalan, poin_id in poin_untuk.items():
            if kode.startswith(awalan) and fallback.get(poin_id):
                out.append((kode, "DILEWATI", f"poin {poin_id} adalah fallback (low_confidence), bukan keluaran LLM"))
                return
        out.append((kode, "OK" if ok else "GAGAL", pesan_ok if ok else pesan_gagal))

    poin = {p["poin_id"]: p for p in hasil["poin"]}
    fallback = {pid: bool(p["low_confidence"]) for pid, p in poin.items()}
    lokasi, gate, ia = data["lokasi"], data["gate_hukum"], data["impact_assessment"]
    L = float(lokasi["luas_lahan_m2"])
    inten = gate["tahapan"]["intensitas"]
    resapan = (lokasi.get("ketentuan_khusus") or {}).get("kawasan_resapan_air")
    kegiatan = gate["tahapan"]["itbx"].get("kegiatan_diusulkan") or ""

    # --- V1 status dari BE ---
    cek("V1.itbx", poin["itbx"]["status"] == gate["tahapan"]["itbx"]["status"], "status ITBX = BE", "status ITBX berbeda dari BE")
    cek("V1.intensitas", poin["intensitas"]["status"] == inten["status"], "status intensitas = BE", "status intensitas berbeda")
    cek("V1.dampak", poin["dampak"]["status"].lower() == ia["impact_category"].lower(), "kategori dampak = BE", "kategori dampak berbeda")

    # --- V2 angka intensitas (dari ambang BE) ---
    if inten["status"] == "MELAMPAUI_BATAS":
        p = inten["parameter"]
        kebutuhan = []
        if not p["kdb"]["memenuhi"]:
            kebutuhan.append(("lantai dasar maks", L * p["kdb"]["ambang_maks"] / 100))
        if not p["kdh"]["memenuhi"]:
            kebutuhan.append(("RTH min", L * p["kdh"]["ambang_min"] / 100))
        teks = semua_teks(poin["intensitas"]) + " " + json.dumps(poin["intensitas"]["rekomendasi"]["langkah_konkret"])
        for nama, nilai in kebutuhan:
            kandidat = {f"{nilai:.1f}".rstrip("0").rstrip("."), f"{nilai:.1f}".replace(".", ","), f"{nilai:.0f}"}
            cek(f"V2.{nama}", any(k in teks for k in kandidat), f"{nama} {nilai:g} m² benar & muncul", f"{nama} {nilai:g} m² TIDAK muncul")
    else:
        cek("V2.aman", not poin["intensitas"]["rekomendasi"]["langkah_konkret"], "intensitas lolos: tanpa langkah", "intensitas lolos tapi ada langkah")

    # --- V3 dampak: mitigasi BE / pemeliharaan ---
    dampak = poin["dampak"]
    langkah = dampak["rekomendasi"]["langkah_konkret"]
    mit = ia.get("rekomendasi_mitigasi")
    if mit:
        pen = mit["rekomendasi_penyesuaian_lahan"]
        nilai = {l["parameter"]: l["nilai_target"] for l in langkah}
        cek("V3.angka_be", abs(nilai.get("Luas Bangunan (Atap)", -1) - pen["luas_bangunan_maks_m2"]) < 1
            and abs(nilai.get("Luas RTH", -1) - pen["luas_rth_min_m2"]) < 1,
            "langkah dampak = angka mitigasi BE", f"langkah dampak tidak cocok BE ({nilai})")
        cek("V3.saran_be", fmt_id(pen["luas_bangunan_maks_m2"]) in dampak["rekomendasi"]["saran"],
            "saran memuat angka BE", "saran tidak memuat angka BE")
    else:
        cek("V3.langkah_rendah", len(langkah) >= 2 and "tidak diperlukan" not in dampak["rekomendasi"]["saran"].lower(),
            f"{len(langkah)} langkah konkret & saran bukan 'tidak perlu tindakan'", "dampak rendah tanpa langkah / saran 'tidak perlu'")

    # --- V4 resapan konsisten ---
    ids = " ".join(s["citation_id"] for s in dampak["sitasi"])
    teks_d = semua_teks(dampak)
    if resapan:
        cek("V4.p53", "p53" in ids, "Pasal 53 dikutip", "Pasal 53 TIDAK dikutip padahal resapan")
        cek("V4.zdq", "zero delta" in teks_d.lower(), "zero delta Q disebut", "zero delta Q tidak disebut")
        cek("V4.ayat2", "p53-a2" not in ids, "tanpa ayat (2) yang tak berlaku utk R-3", "ayat (2) ikut dikutip")
    else:
        cek("V4.tanp53", "p53" not in ids and "p50" not in ids, "tanpa Pasal 53/50 di sitasi", "Pasal 53/50 ikut dikutip di non-resapan")
        salah = [k for k in _PISAH.split(teks_d) if re.search(r"zero\s+delta\s+q|pasal\s+53", k, re.I) and not _NEGASI.search(k)]
        cek("V4.klaim", not salah, "tidak memakai ketentuan resapan sebagai berlaku", f"klaim resapan di non-resapan: {salah[:1]}")

    # --- V5 sitasi pengisi ---
    def sitasi_bermasalah(p, pola):
        return [s["citation_id"] for s in p["sitasi"] if re.search(pola, s["citation_id"]) or s.get("pasal") in ("1", "44")]
    cek("V5.dampak", not [s for s in dampak["sitasi"] if s.get("pasal") in ("1", "44") or "-vi-" in s["citation_id"]],
        "sitasi dampak tanpa pengisi", "sitasi dampak memuat pengisi (Pasal 1/44/Lampiran VI)")
    cek("V5.itbx", not [s for s in poin["itbx"]["sitasi"] if "-vi-" in s["citation_id"]], "ITBX tanpa Lampiran VI", "ITBX mengutip Lampiran VI")
    cek("V5.intensitas", not [s for s in poin["intensitas"]["sitasi"] if "-vb-" in s["citation_id"]], "intensitas tanpa Lampiran V.B", "intensitas mengutip V.B")

    # --- V6 ITBX ---
    t_itbx = semua_teks(poin["itbx"])
    cek("V6.tanpa_intensitas", not re.search(r"\b(?:KDB|KLB|KDH)\b", t_itbx), "ITBX tidak membahas KDB/KLB/KDH", "ITBX membahas KDB/KLB/KDH")
    cek("V6.tanpa_otomatis", not re.search(r"otomatis", t_itbx, re.I), "tanpa klaim 'otomatis'", "klaim 'otomatis'")
    cek("V6.tanpa_telah_memenuhi", not re.search(r"(?:sudah|telah)\s+memenuhi", t_itbx, re.I), "tanpa klaim 'telah memenuhi'", "klaim 'telah memenuhi'")
    cek("V6.tanpa_admin", not re.search(r"(?:menyertakan|sertakan|melampirkan|menyiapkan|siapkan)\s+(?:dokumen|berkas|perhitungan)|menambah\s+blok", t_itbx, re.I),
        "tanpa kewajiban administratif karangan", "kewajiban administratif karangan")

    # --- V7 gaya penulisan (semua teks) ---
    semua = " ".join([semua_teks(p) for p in hasil["poin"]] + [hasil["narasi_rekomendasi"]["paragraf_gate_intensitas"],
                      hasil["narasi_rekomendasi"]["paragraf_dampak"], " ".join(hasil["kesimpulan"]["langkah_berdampak"]),
                      hasil["kesimpulan"].get("catatan_lokasi") or "", hasil["ringkasan_gate"]["kalimat"], hasil["ringkasan_dampak"]["kalimat"]])
    cek("V7.titik_koma", ";" not in semua, "tanpa titik koma", "ada titik koma")
    cek("V7.enum", not _ENUM.search(semua), "tanpa label enum teknis", "label enum bocor")
    cek("V7.bold", not any(_BOLD_GANJIL(semua_teks(p)) for p in hasil["poin"]), "** berpasangan", "** tak berpasangan")
    cek("V7.newline", "\\n" not in semua, "tanpa \\n literal", "\\n literal")
    cek("V7.citation_id", "rdtr-sleman-tengah-" not in semua and "anchor-" not in semua, "tanpa citation_id bocor", "citation_id bocor")
    cek("V7.nomor_ganda", not any(re.match(r"\s*\d+[.)]\s", x) for x in hasil["kesimpulan"]["langkah_berdampak"]), "butir kesimpulan tanpa nomor", "butir kesimpulan bernomor")

    # --- V8 tipe bangunan ---
    kata = [k for k in re.findall(r"[a-z]{4,}", kegiatan.lower()) if k not in {"lainnya", "yang"}]
    cek("V8.ringkasan", kegiatan in hasil["ringkasan_dampak"]["kalimat"], "ringkasan_dampak menyebut kegiatan", "ringkasan_dampak tanpa kegiatan")
    cek("V8.reasoning", any(k in (dampak["reasoning_pendek"] + dampak["reasoning_panjang"]).lower() for k in kata),
        "penjelasan dampak menyebut kegiatan", "penjelasan dampak TIDAK menyebut kegiatan")

    # --- V9 batas ketat ---
    if inten["status"] == "MELAMPAUI_BATAS" and mit:
        cek("V9.catatan", "lebih ketat" in (dampak["rekomendasi"].get("disclaimer") or "") and "lebih ketat" in hasil["narasi_rekomendasi"]["paragraf_dampak"]
            and any("lebih ketat" in x for x in hasil["kesimpulan"]["langkah_berdampak"]),
            "catatan batas lebih ketat di disclaimer, narasi, kesimpulan", "catatan batas lebih ketat tidak lengkap")

    # --- V10 kesehatan alur ---
    cek("V10.low_conf", not hasil["low_confidence_keseluruhan"] and not hasil["catatan_global"], "tidak ada low_confidence", f"low_confidence: {hasil['catatan_global']}")
    if rekaman:
        diag = rekaman.get("diagnostik") or []
        info = {d["poin_id"]: d["percobaan"] for d in diag}
        tolak = [(d["poin_id"], e["percobaan"], (e["masalah"] or [e.get("exception") or ""])[0][:90])
                 for d in diag for e in d.get("riwayat_percobaan", []) if e["sebab"] != "berhasil"]
        out.append(("INFO.percobaan", "INFO", f"percobaan per poin={info}; sintesis={rekaman.get('sintesis')}; durasi={rekaman.get('durasi_detik')}s"))
        for pid, ke, alasan in tolak:
            out.append(("INFO.tolak", "INFO", f"{pid} percobaan-{ke} ditolak: {alasan}"))
    return out


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "hasil_uji").iterdir())[-1]
    payload, log = muat_payload(), muat_log()
    total_ok = total_gagal = 0
    for berkas in sorted(folder.glob("*.json")):
        if berkas.name.startswith("_"):
            continue
        hasil = json.loads(berkas.read_text(encoding="utf-8"))
        nomor = next((n for n, d in payload.items() if berkas.stem and n in json.dumps(hasil) or False), None)
        # cocokkan lewat nama berkas -> payload
        for sumber in SUMBER_PAYLOAD:
            kandidat = sumber / berkas.name
            if kandidat.exists():
                nomor = json.loads(kandidat.read_text(encoding="utf-8"))["data"]["application_number"]
        data = payload[nomor]
        hasil_cek = periksa(nomor, hasil, data, log.get(nomor))
        gagal = [h for h in hasil_cek if h[1] == "GAGAL"]
        ok = [h for h in hasil_cek if h[1] == "OK"]
        total_ok += len(ok)
        total_gagal += len(gagal)
        print(f"\n=== {nomor}: {len(ok)} lolos, {len(gagal)} gagal")
        for kode, status, pesan in hasil_cek:
            if status != "OK":
                print(f"  [{status}] {kode}: {pesan}")
    print(f"\nTOTAL: {total_ok} pemeriksaan lolos, {total_gagal} gagal")


if __name__ == "__main__":
    main()
