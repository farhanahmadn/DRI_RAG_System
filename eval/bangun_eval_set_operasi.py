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
_EVAL_LAMA = _AKAR / "tests" / "eval_set.jsonl"
_LOG_PRECHECK = _AKAR / "logs" / "precheck.jsonl"

# String query PERSIS seperti yang diterbitkan app/reasoning/generator.py. Diimpor, bukan disalin,
# supaya eval set ikut basi kalau produksi mengganti querinya (ketimbang diam-diam menguji yang lama).
from app.reasoning.generator import (  # noqa: E402
    _QUERY_FALLBACK_INTENSITAS_DGN_SUBZONA,
    _QUERY_FALLBACK_PER_POIN,
)

# Rujukan generik yang SELALU dikirim BE utk poin itbx (terhitung 349/349 payload di log).
_RUJUKAN_ITBX = "RDTR Sleman Matriks ITBX"


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
                _QUERY_FALLBACK_INTENSITAS_DGN_SUBZONA, {"zona": zona}, lamp["vi"], "aturan",
                "Query tajam yang dipakai produksi saat sub-zona presisi diketahui.", w))
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
                f"Sub-zona tak diketahui: seluruh Lampiran VI keluarga {k} sah.", w))
        if lamp["vb"]:
            topik.append(_topik(
                f"itbx-kegiatan-keluarga-{k}", "search", "itbx",
                _QUERY_FALLBACK_PER_POIN["itbx"], {"zona_prefix": k}, lamp["vb"], "aturan",
                f"Sub-zona tak diketahui: seluruh Lampiran V.B keluarga {k} sah.", w))
            topik.append(_topik(
                f"itbx-anchor-keluarga-{k}", "reference", "itbx",
                _RUJUKAN_ITBX, {"zona_prefix": k}, lamp["vb"], "aturan",
                "Jalur rujukan tanpa sub-zona presisi — kasus paling sering di produksi.", w))

    # ---- C. Dampak hidrologi — TIDAK ada tabel per-zona, labelnya memang penilaian ----
    # Diwarisi dari eval set lama & ditandai seeded. Jujur lebih berguna daripada memaksakan
    # label "aturan" untuk sesuatu yang tak punya dasar struktural.
    if _EVAL_LAMA.exists():
        kunci_dampak = ("resapan", "banjir", "dampak")
        for baris in _EVAL_LAMA.read_text(encoding="utf-8").splitlines():
            if not baris.strip():
                continue
            lama = json.loads(baris)
            if any(k in lama["query"].lower() for k in kunci_dampak):
                topik.append(_topik(
                    "dampak-" + re.sub(r"\W+", "-", lama["query"].lower()).strip("-"),
                    "search", "dampak", lama["query"], {}, lama["relevan"], "seeded",
                    f"Diwarisi eval set lama; tak ada tabel per-zona utk dampak. {lama.get('catatan','')}".strip(),
                    0))
        topik.append(_topik(
            "dampak-query-produksi", "search", "dampak",
            _QUERY_FALLBACK_PER_POIN["dampak"], {},
            [t["relevan"] for t in topik if t["poin"] == "dampak"][0] if any(
                t["poin"] == "dampak" for t in topik) else [],
            "seeded", "Query dampak yang PERSIS diterbitkan produksi.", 0))

    # ---- D. KONTRAFAKTUAL: filter keluarga + query TAJAM ----
    # Produksi tidak pernah menerbitkan kombinasi ini: query tajam dipakai HANYA saat sub-zona
    # presisi diketahui (generator.py, _QUERY_FALLBACK_INTENSITAS_DGN_SUBZONA), dengan alasan
    # eksplisit bahwa tanpa filter exact sistem bisa percaya diri menyitasi sub-zona yang salah.
    # Topik ini mengisolasi efek STRING QUERY: filternya identik dengan topik "-kdb-keluarga-"
    # di bagian B, hanya teks querinya berbeda, sehingga perbandingannya berpasangan sempurna.
    # Ditaruh PALING AKHIR agar urutan topik sebelumnya tidak bergeser — checkpoint evaluasi
    # yang sudah berjalan tetap sah dan tidak perlu dibayar ulang.
    for t in [x for x in topik if x["id"].startswith("intensitas-kdb-keluarga-")]:
        topik.append(_topik(
            t["id"].replace("intensitas-kdb-keluarga-", "intensitas-tajam-keluarga-"),
            "search", "intensitas", _QUERY_FALLBACK_INTENSITAS_DGN_SUBZONA,
            dict(t["filter"]), t["relevan"], "aturan",
            "Kontrafaktual: filter keluarga + query tajam. Pasangan dari "
            f'{t["id"]} — hanya string query yang berbeda.', t["bobot_traffic"]))

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
