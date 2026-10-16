"""Ukur apakah keluaran LLM bersifat TEMPLAT (mirip antar permohonan berdampak sama) atau UNIK.

Membaca `logs/precheck.jsonl` (hasil run payload uji APP-2026-UJI-* set pertama dan APP-2026-UJI2-* set kedua),
mengambil hasil TERAKHIR tiap nomor permohonan, lalu membandingkan teks poin antar:
  - SEL SAMA : set1 dan set2 dengan kombinasi dampak + status resapan yang sama, tetapi KEGIATAN dan ANGKA berbeda.
  - SEL BEDA : pembanding dasar, set yang sama tetapi kombinasi berbeda (mis. rendah vs tinggi).

Ukuran: kemiripan Jaccard atas trigram kata, setelah semua angka diganti "#" (agar yang terukur adalah
KERANGKA kalimat, bukan angkanya). 0 = tidak ada kerangka sama, 1 = identik.

Cara membaca: kemiripan SEL SAMA jauh di atas SEL BEDA berarti LLM memakai templat per tingkat dampak
(kerangka sama walau kegiatan/angka berbeda). Bila keduanya rendah dan mirip, jawabannya menyesuaikan
masukan. Kemiripan tinggi pada `saran`/langkah yang berasal dari kode (bukan LLM) wajar dan bukan temuan.

Jalankan dari root repo:  python -m scripts.bandingkan_pola_keluaran [path/ke/precheck.jsonl]
"""

from __future__ import annotations

import json
import re
import sys
from itertools import product
from pathlib import Path

LOG_BAWAAN = Path(__file__).resolve().parent.parent / "logs" / "precheck.jsonl"
_RE_NOMOR = re.compile(r"APP-2026-(UJI2?)-(\d)-(TINGGI|RENDAH)-(RESAPAN|NONRESAPAN)")
BIDANG = (("itbx", "reasoning_panjang"), ("intensitas", "reasoning_panjang"),
          ("dampak", "reasoning_pendek"), ("dampak", "reasoning_panjang"), ("dampak", "saran"))


def muat_terakhir(path: Path) -> dict[str, dict]:
    """application_number -> response (baris TERAKHIR menang)."""
    hasil: dict[str, dict] = {}
    for baris in path.read_text(encoding="utf-8").splitlines():
        if not baris.strip():
            continue
        rec = json.loads(baris)
        nomor = (rec.get("request") or {}).get("application_number") or ""
        if _RE_NOMOR.fullmatch(nomor):
            hasil[nomor] = rec["response"]
    return hasil


def sel_dari_nomor(nomor: str) -> tuple[str, str, str]:
    """(set, kategori, status_resapan)."""
    m = _RE_NOMOR.fullmatch(nomor)
    return m.group(1), m.group(3), m.group(4)


def trigram(teks: str) -> set[tuple[str, ...]]:
    normal = re.sub(r"\d+(?:[.,]\d+)*", "#", teks.lower())
    kata = re.findall(r"[a-z#]+", normal)
    return {tuple(kata[i:i + 3]) for i in range(len(kata) - 2)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def ambil_teks(response: dict, poin_id: str, bidang: str) -> str:
    for p in response.get("poin", []):
        if p.get("poin_id") == poin_id:
            return p.get(bidang) if bidang != "saran" else (p.get("rekomendasi") or {}).get("saran", "") or ""
    return ""


def bandingkan(hasil: dict[str, dict]) -> dict:
    """{ (poin, bidang): {"sama": [skor...], "beda": [skor...]} }."""
    keluar: dict = {k: {"sama": [], "beda": []} for k in BIDANG}
    nomor = sorted(hasil)
    for a, b in product([n for n in nomor if sel_dari_nomor(n)[0] == "UJI"],
                        [n for n in nomor if sel_dari_nomor(n)[0] == "UJI2"]):
        _, kat_a, res_a = sel_dari_nomor(a)
        _, kat_b, res_b = sel_dari_nomor(b)
        jenis = "sama" if (kat_a, res_a) == (kat_b, res_b) else "beda"
        for poin_id, bidang in BIDANG:
            ta = trigram(ambil_teks(hasil[a], poin_id, bidang))
            tb = trigram(ambil_teks(hasil[b], poin_id, bidang))
            keluar[(poin_id, bidang)][jenis].append(jaccard(ta, tb))
    return keluar


def rata(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def laporan(hasil: dict[str, dict]) -> str:
    set1 = [n for n in hasil if sel_dari_nomor(n)[0] == "UJI"]
    set2 = [n for n in hasil if sel_dari_nomor(n)[0] == "UJI2"]
    baris = [f"Permohonan terbaca: set pertama={len(set1)}, set kedua={len(set2)}"]
    if not set1 or not set2:
        baris.append("Perlu hasil KEDUA set di logs/precheck.jsonl untuk membandingkan.")
        return "\n".join(baris)
    baris.append(f"{'poin.bidang':<28}{'SEL SAMA':>10}{'SEL BEDA':>10}{'selisih':>10}  kesimpulan")
    for (poin_id, bidang), v in bandingkan(hasil).items():
        s, b = rata(v["sama"]), rata(v["beda"])
        if s is None or b is None:
            continue
        selisih = s - b
        if selisih > 0.15 and s > 0.30:
            kesimpulan = "TEMPLAT per tingkat dampak"
        elif s > 0.30:
            kesimpulan = "mirip secara umum (bukan khas dampak)"
        else:
            kesimpulan = "unik / menyesuaikan masukan"
        baris.append(f"{poin_id + '.' + bidang:<28}{s:>10.2f}{b:>10.2f}{selisih:>10.2f}  {kesimpulan}")
    return "\n".join(baris)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_BAWAAN
    print(laporan(muat_terakhir(path)))


if __name__ == "__main__":
    main()
