"""eval/bangun_eval_set_operasi.py — bangkitkan eval set pada TITIK OPERASI PRODUKSI.

Kenapa berkas ini ada. `tests/eval_set.jsonl` menguji 22 query bebas-teks (LP2B, sempadan sungai,
GSB, banjir...), padahal produksi **tidak pernah menerbitkan query-query itu**. Diperiksa terhadap
payload back-end nyata (`logs/precheck.jsonl`) dan kode:

  * poin `itbx`       : SELALU punya `dasar_hukum` dari BE -> tidak pernah `search()`,
                        selalu lewat `get_by_reference` (jalur ini belum pernah dievaluasi)
  * poin `intensitas` : TIDAK PERNAH punya `dasar_hukum` -> selalu `search("kdb")` /
                        `search("ambang KDB KLB KDH maksimal minimal")` saat sub-zona diketahui
  * poin `dampak`     : selalu `search("dampak tata guna lahan")`

Jadi produksi hanya menerbitkan SEGELINTIR string query tetap; yang benar-benar bervariasi adalah
**filter zona/sub-zona**. Dari 22 query lama, hanya 2 yang dipakai produksi. Eval set ini menguji
sumbu yang benar: query tetap x zona yang nyata ada di korpus, plus jalur rujukan.

LABEL DITURUNKAN DARI ATURAN, BUKAN OPINI. Untuk titik operasi ini, chunk otoritatif ditentukan
struktur regulasinya sendiri dan sudah ada sebagai data di kolom `chunks.zona`:

  * ambang KDB/KLB/KDH sub-zona Z -> tabel-chunk Lampiran VI zona Z
  * ketentuan kegiatan T/B zona Z -> tabel-chunk Lampiran V.B zona Z

Label dibangkitkan deterministik dari DB, bukan diketik tangan — bisa diregenerasi saat korpus
berubah, dan tidak bergantung penilaian siapa pun. Topik yang labelnya memang penilaian (mis.
dampak hidrologi, yang tak punya tabel per-zona) diwarisi dari eval set lama dan DITANDAI
`sumber_label: "seeded"` supaya pembaca laporan tahu persis mana yang objektif dan mana yang tidak.

CLI:
  python -m eval.bangun_eval_set_operasi
  python -m eval.bangun_eval_set_operasi --out tests/eval_set_operasi.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_LOG_PRECHECK = _AKAR / "logs" / "precheck.jsonl"

# String query PERSIS seperti yang diterbitkan app/reasoning/generator.py. Diimpor, bukan disalin,
# supaya eval set ikut basi kalau produksi mengganti querinya (ketimbang diam-diam menguji yang lama).
from app.reasoning.generator import (  # noqa: E402
    _QUERY_INTENSITAS_TAJAM,
    _QUERY_FALLBACK_PER_POIN,
)

# Rujukan generik yang SELALU dikirim BE utk poin itbx (terhitung 349/349 payload di log).
_RUJUKAN_ITBX = "RDTR Sleman Matriks ITBX"

# Pasal yang mengatur poin `dampak`, DIBATASI dua ini secara sadar. Keduanya "ketentuan khusus"
# yang menyebut kode zona eksplisit di dalam teksnya, sehingga label bisa diturunkan dari aturan:
#   Pasal 53 — kawasan resapan air (langsung soal infiltrasi/limpasan, inti poin dampak)
#   Pasal 50 — kawasan rawan bencana (gempa, gunung api, banjir lahar, kekeringan, longsor)
# Pasal lain TIDAK diikutkan walau teksnya mengandung kata "resapan"/"sempadan": penyaringan
# kata kunci saja ikut menjaring Pasal 44 (luas minimal bidang tanah -> itu poin intensitas),
# Pasal 1 (definisi istilah), dan Pasal 58. Yang menentukan di sini adalah pasal apa yang
# MENGATUR dampak, bukan pasal apa yang menyebut katanya.
_PASAL_DAMPAK = ("50", "53")

# "... pada Zona Badan Jalan dengan kode BJ, Sub-zona Tanaman Pangan dengan kode P-1, ..."
_RE_KODE_ZONA = re.compile(r"dengan kode ([A-Z]{1,4}(?:-\d+)?)")


def _keluarga(zona: str) -> str:
    """'P-1 LP2B' -> 'P', 'RTH-2' -> 'RTH', 'CA' -> 'CA'. Sejalan dgn generator._keluarga_zona."""
    return zona.strip().upper().split("-")[0].split()[0]


def _muat_zona_korpus(conn, wilayah: str) -> dict[str, dict[str, list[str]]]:
    """Petakan zona -> {'vi': [id...], 'vb': [id...]} dari tabel-chunk yang BENAR-BENAR ada di DB.

    Dibaca dari kolom `zona` + penanda lampiran di `id`, bukan dari daftar zona yang di-hardcode —
    zona yang belum ter-ingest otomatis tidak menghasilkan topik, ketimbang menghasilkan topik yang
    mustahil dijawab dan menyeret metrik turun tanpa sebab.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, zona FROM chunks "
            "WHERE dokumen ILIKE %s AND level = 'tabel' AND zona IS NOT NULL ORDER BY id",
            [f"%{wilayah}%"],
        )
        baris = cur.fetchall()

    per_zona: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"vi": [], "vb": []})
    for cid, zona in baris:
        if re.search(r"-vi-", cid):
            per_zona[zona]["vi"].append(cid)
        elif re.search(r"-vb-", cid):
            per_zona[zona]["vb"].append(cid)
    return dict(per_zona)


def _muat_ayat_dampak(conn, wilayah: str) -> dict[str, list[str]]:
    """kode zona -> ayat Pasal 50/53 yang menyebut kode itu eksplisit.

    Inilah yang membuat label `dampak` bisa diturunkan dari aturan, bukan dari penilaian:
    ayatnya sendiri yang menyatakan berlaku untuk zona mana. Terhitung 6 ayat memuat kode zona
    (Pasal 50 ayat 3/4/8/9, Pasal 53 ayat 2/3), mencakup 31 dari 32 zona ber-tag di korpus.
    """
    per_zona: dict[str, set[str]] = defaultdict(set)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, teks FROM chunks WHERE dokumen ILIKE %s AND level = 'ayat' "
            "AND pasal = ANY(%s) ORDER BY id",
            [f"%{wilayah}%", list(_PASAL_DAMPAK)])
        for cid, teks in cur.fetchall():
            for kode in {k.upper() for k in _RE_KODE_ZONA.findall(teks or "")}:
                per_zona[kode].add(cid)
    return {z: sorted(ids) for z, ids in per_zona.items()}


def _kode_dasar(zona: str) -> str:
    """'P-1 LP2B' -> 'P-1'. Satu-satunya zona ber-tag yang tak disebut Pasal 50/53 adalah
    'P-1 LP2B' — label gabungan di korpus, yang dasarnya tetap sub-zona P-1."""
    return zona.split(" ")[0].upper()


def _bobot_zona_nyata() -> Counter:
    """Frekuensi zona/sub-zona pada payload BE nyata — dipakai sbg bobot pelaporan, bukan filter.

    Zona langka TETAP diuji (metrik tak-tertimbang dilaporkan berdampingan); bobot ini hanya supaya
    rata-rata tertimbang mencerminkan beban yang sesungguhnya ditanggung sistem.
    """
    bobot: Counter = Counter()
    if not _LOG_PRECHECK.exists():
        return bobot
    for baris in _LOG_PRECHECK.read_text(encoding="utf-8", errors="replace").splitlines():
        if not baris.strip():
            continue
        try:
            rec = json.loads(baris)
        except Exception:
            continue
        req = rec.get("request") or {}
        if not req.get("application_number"):
            continue
        lok = req.get("lokasi") or {}
        sub, induk = lok.get("rdtr_subzone"), lok.get("rdtr_zone")
        if sub:
            bobot[sub] += 1
        elif induk:
            bobot[f"induk:{induk}"] += 1
    return bobot


def _topik(
    tid: str, jalur: str, poin: str, query: str, filt: dict, relevan: list[str],
    sumber_label: str, catatan: str, bobot: int = 0,
) -> dict:
    return {
        "id": tid, "jalur": jalur, "poin": poin, "query": query, "filter": filt,
        "relevan": sorted(set(relevan)), "sumber_label": sumber_label,
        "catatan": catatan, "bobot_traffic": bobot,
    }


def bangun(conn, wilayah: str) -> list[dict]:
    per_zona = _muat_zona_korpus(conn, wilayah)
    bobot = _bobot_zona_nyata()
    topik: list[dict] = []

    # ---- A. Sub-zona PRESISI (BE mengirim rdtr_subzone) -> filter zona exact ----
    for zona, lamp in sorted(per_zona.items()):
        w = bobot.get(zona, 0)
        if lamp["vi"]:
            topik.append(_topik(
                f"intensitas-kdb-{zona}", "search", "intensitas",
                _QUERY_FALLBACK_PER_POIN["intensitas"], {"zona": zona}, lamp["vi"], "aturan",
                f"Lampiran VI zona {zona} = sumber otoritatif ambang KDB/KLB/KDH sub-zona itu.", w))
            topik.append(_topik(
                f"intensitas-ambang-{zona}", "search", "intensitas",
                _QUERY_INTENSITAS_TAJAM, {"zona": zona}, lamp["vi"], "aturan",
                "Query tajam produksi, filter exact sub-zona.", w))
        if lamp["vb"]:
            topik.append(_topik(
                f"itbx-kegiatan-{zona}", "search", "itbx",
                _QUERY_FALLBACK_PER_POIN["itbx"], {"zona": zona}, lamp["vb"], "aturan",
                f"Lampiran V.B zona {zona} = ketentuan kegiatan T/B sub-zona itu.", w))
            topik.append(_topik(
                f"itbx-anchor-{zona}", "reference", "itbx",
                _RUJUKAN_ITBX, {"zona": zona}, lamp["vb"], "aturan",
                "Jalur get_by_reference — menangani 100% poin itbx di produksi.", w))

    # ---- B. Hanya zona INDUK (BE tak mengirim sub-zona; mayoritas traffic) -> filter keluarga ----
    # Tanpa sub-zona presisi, SELURUH tabel satu keluarga zona sah dianggap relevan — itu memang
    # batas informasi yang tersedia sistem, bukan kelonggaran penilaian.
    keluarga: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"vi": [], "vb": []})
    for zona, lamp in per_zona.items():
        k = _keluarga(zona)
        keluarga[k]["vi"].extend(lamp["vi"])
        keluarga[k]["vb"].extend(lamp["vb"])

    for k, lamp in sorted(keluarga.items()):
        w = sum(v for kk, v in bobot.items() if kk.startswith("induk:")) if k == "R" else 0
        if lamp["vi"]:
            topik.append(_topik(
                f"intensitas-kdb-keluarga-{k}", "search", "intensitas",
                _QUERY_FALLBACK_PER_POIN["intensitas"], {"zona_prefix": k}, lamp["vi"], "aturan",
                "Sub-zona tak diketahui, query pendek. Sejak gating query dibuka ini jadi "
                "KONTRAFAKTUAL (produksi memakai query tajam) — dipertahankan sebagai pembanding "
                f"& penjaga regresi. Seluruh Lampiran VI keluarga {k} sah.", w))
        if lamp["vb"]:
            topik.append(_topik(
                f"itbx-kegiatan-keluarga-{k}", "search", "itbx",
                _QUERY_FALLBACK_PER_POIN["itbx"], {"zona_prefix": k}, lamp["vb"], "aturan",
                f"Sub-zona tak diketahui: seluruh Lampiran V.B keluarga {k} sah.", w))
            topik.append(_topik(
                f"itbx-anchor-keluarga-{k}", "reference", "itbx",
                _RUJUKAN_ITBX, {"zona_prefix": k}, lamp["vb"], "aturan",
                "Jalur rujukan tanpa sub-zona presisi — kasus paling sering di produksi.", w))

    # ---- C. Kontrol tanpa filter zona ----
    # Sebelumnya bagian ini mewarisi 5 topik dampak dari eval set lama. EMPAT di antaranya memakai
    # query yang produksi TIDAK PERNAH terbitkan ("Resapan", "Banjir", "kawasan resapan air",
    # "rawan bencana banjir lahar") — peninggalan desain 8-indikator. Menguji query yang tak pernah
    # dipakai hanya menambah angka tanpa menambah informasi tentang sistem yang berjalan, dan
    # membuat metrik `dampak` tercampur antara jalur produksi dan jalur yang sudah mati.
    #
    # Yang tersisa satu: query dampak produksi TANPA filter zona. Dipertahankan sebagai KONTROL —
    # topik `dampak-zona-*` dan `dampak-keluarga-*` memakai query yang sama dengan filter, jadi
    # selisihnya mengisolasi sumbangan filter itu sendiri. Labelnya kini diturunkan dari aturan
    # (ayat Pasal 53 yang menyebut kode zona), bukan diwarisi, sehingga eval set tak lagi
    # bergantung pada berkas lama sama sekali.
    ayat_resapan = sorted({c for z, ids in _muat_ayat_dampak(conn, wilayah).items()
                           for c in ids if "-p53-" in c})
    if ayat_resapan:
        topik.append(_topik(
            "dampak-query-produksi", "search", "dampak",
            _QUERY_FALLBACK_PER_POIN["dampak"], {}, ayat_resapan, "aturan",
            "KONTROL: query dampak produksi TANPA filter zona. Pasangannya dampak-zona-* dan "
            "dampak-keluarga-* memakai query sama dengan filter, jadi selisihnya mengukur "
            "sumbangan filter. Label = ayat Pasal 53 (ketentuan kawasan resapan air), yaitu tema "
            "yang memang dijangkau query ini.", 0))

    # ---- D. KONTRAFAKTUAL: filter keluarga + query TAJAM ----
    # Produksi tidak pernah menerbitkan kombinasi ini: query tajam dipakai HANYA saat sub-zona
    # presisi diketahui. Gating itu kini DIBUKA di produksi justru karena hasil ablasi ini —
    # topiknya tetap dipertahankan sebagai pembanding historis & penjaga regresi.
    # Topik ini mengisolasi efek STRING QUERY: filternya identik dengan topik "-kdb-keluarga-"
    # di bagian B, hanya teks querinya berbeda, sehingga perbandingannya berpasangan sempurna.
    # Ditaruh PALING AKHIR agar urutan topik sebelumnya tidak bergeser — checkpoint evaluasi
    # yang sudah berjalan tetap sah dan tidak perlu dibayar ulang.
    for t in [x for x in topik if x["id"].startswith("intensitas-kdb-keluarga-")]:
        topik.append(_topik(
            t["id"].replace("intensitas-kdb-keluarga-", "intensitas-tajam-keluarga-"),
            "search", "intensitas", _QUERY_INTENSITAS_TAJAM,
            dict(t["filter"]), t["relevan"], "aturan",
            "Filter keluarga + query tajam = perilaku PRODUKSI sejak gating dibuka. "
            f'Pasangan dari {t["id"]} — hanya string query yang berbeda.', t["bobot_traffic"]))

    # ---- E. Dampak berlabel ATURAN, dari Pasal 50 & 53 ----
    # Menggantikan ketergantungan pada 5 topik warisan di bagian C (4 di antaranya memakai query
    # yang produksi tak pernah terbitkan, semuanya berlabel `seeded`). Topik lama TETAP ada
    # sebagai pembanding historis, dan tandanya tetap `seeded` supaya pembaca laporan bisa
    # memisahkan mana yang objektif.
    #
    # BATAS LABEL YANG HARUS IKUT TERBAWA: Pasal 53 ayat (1) membatasi keberlakuan ke SWP/Blok
    # tertentu, dan dari KODE ZONA saja kita tidak bisa tahu apakah lokasi pemohon ada di blok
    # resapan itu. Jadi label di sini bermakna "KALAU lokasi berada di kawasan resapan air /
    # rawan bencana, inilah ayat yang mengatur" — bersyarat, lebih lemah daripada label
    # intensitas/itbx yang tabelnya tanpa syarat milik pemohon. Dinyatakan di `catatan` tiap
    # topik, bukan cuma di komentar kode, supaya batas itu ikut pindah bersama datanya.
    ayat_dampak = _muat_ayat_dampak(conn, wilayah)
    _CATATAN_DAMPAK = (
        "Label turunan aturan: Pasal 50/53 menyebut kode zona ini eksplisit. BERSYARAT — "
        "Pasal 53 ayat (1) membatasi keberlakuan ke SWP/Blok tertentu dan itu tak bisa "
        "ditentukan dari kode zona, jadi ayat ini otoritatif HANYA bila lokasi memang berada "
        "di kawasan resapan air/rawan bencana."
    )
    for zona in sorted(per_zona):
        relevan = ayat_dampak.get(_kode_dasar(zona)) or []
        if not relevan:
            continue
        topik.append(_topik(
            f"dampak-zona-{zona}", "search", "dampak",
            _QUERY_FALLBACK_PER_POIN["dampak"], {"zona": zona}, relevan, "aturan",
            f"{_CATATAN_DAMPAK} Sub-zona presisi diketahui -> filter exact.",
            bobot.get(zona, 0)))

    # Cabang mayoritas: sub-zona tak dikonfirmasi -> filter keluarga, seluruh ayat yang mengatur
    # anggota keluarga itu sah (sejalan dgn perlakuan Lampiran VI/V.B di bagian B).
    kel_dampak: dict[str, set[str]] = defaultdict(set)
    for zona in per_zona:
        kel_dampak[_keluarga(zona)] |= set(ayat_dampak.get(_kode_dasar(zona)) or [])
    for k, relevan in sorted(kel_dampak.items()):
        if not relevan:
            continue
        topik.append(_topik(
            f"dampak-keluarga-{k}", "search", "dampak",
            _QUERY_FALLBACK_PER_POIN["dampak"], {"zona_prefix": k}, sorted(relevan), "aturan",
            f"{_CATATAN_DAMPAK} Sub-zona tak diketahui -> filter keluarga {k}.",
            0))

    return topik


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangkitkan eval set titik operasi produksi.")
    ap.add_argument("--out", type=Path, default=_AKAR / "tests" / "eval_set_operasi.jsonl")
    args = ap.parse_args()

    import psycopg

    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"')
    conn = psycopg.connect(os.getenv("DATABASE_URL"))
    topik = bangun(conn, wilayah)
    conn.close()

    kosong = [t["id"] for t in topik if not t["relevan"]]
    if kosong:
        raise SystemExit(f"[gagal] {len(kosong)} topik tanpa label relevan: {kosong[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in topik) + "\n", encoding="utf-8")

    per_jalur = Counter(t["jalur"] for t in topik)
    per_poin = Counter(t["poin"] for t in topik)
    per_label = Counter(t["sumber_label"] for t in topik)
    print(f"[eval-set] wilayah={wilayah}  topik={len(topik)}  -> {args.out}")
    print(f"  jalur : {dict(per_jalur)}")
    print(f"  poin  : {dict(per_poin)}")
    print(f"  label : {dict(per_label)}")
    print(f"  rata-rata chunk relevan/topik: "
          f"{sum(len(t['relevan']) for t in topik)/len(topik):.2f}")


if __name__ == "__main__":
    main()
