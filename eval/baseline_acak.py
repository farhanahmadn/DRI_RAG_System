"""eval/baseline_acak.py — lantai dan langit-langit metrik untuk korpus INI.

Kenapa modul ini ada. Pertanyaan "skor 0.33 itu bagus atau buruk?" tidak bisa dijawab oleh ambang
yang dipinjam dari luar: nDCG bergantung pada koleksinya — berapa kandidat yang lolos filter, dan
berapa chunk yang dilabeli relevan. Skor yang sama bisa berarti nyaris sempurna di satu koleksi dan
nyaris acak di koleksi lain. Satu-satunya pembanding absolut yang sah adalah pembanding yang
dihitung DARI koleksi yang sama:

  * LANTAI   — harapan metrik bila sistem mengembalikan k chunk ACAK dari kandidat yang lolos
               filter topik itu. Inilah "tanpa kepintaran apa pun, tapi dengan filter yang sama".
  * LANGIT   — nilai maksimum yang MUNGKIN dicapai mengingat struktur label. Penting karena saat
               satu topik punya 3 chunk relevan sedangkan sistem hanya mengirim k=3, Recall@3 masih
               bisa 100%, TAPI kalau punya 4 relevan maka langit-langitnya 3/4 = 75%, bukan 100%.
               Membandingkan skor dengan 1.0 di kasus begitu menghukum sistem atas batas yang
               diciptakan desain, bukan atas kegagalannya.

Lantai acak dihitung dengan simulasi, bukan rumus tertutup: nDCG mendiskon menurut posisi, jadi
bentuk analitiknya menjadi berantakan sementara simulasi langsung memakai `_metrik_satu_query` yang
SAMA dengan yang dipakai laporan — menghapus peluang lantai dan skor sistem dihitung dengan dua
definisi metrik yang sedikit berbeda.

Tidak memanggil API embedding/rerank apa pun; hanya DB untuk menghitung besar kandidat per topik.

CLI:
  python -m eval.baseline_acak
  python -m eval.baseline_acak --eval-set tests/eval_set_operasi.jsonl --n-simulasi 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_OUT = Path(__file__).parent / "baseline.json"
_SEED = 20261005          # tetap -> laporan bisa diproduksi ulang persis


def _kandidat_per_topik(conn, topik: list[dict], wilayah: str) -> dict[str, list[str]]:
    """id chunk yang LOLOS FILTER tiap topik — semesta tempat jalur acak mengundi.

    Memakai `db._where` yang sama dengan jalur retrieval produksi, bukan WHERE tiruan: kalau
    semantik filter berubah (mis. chunk ber-zona NULL ikut lolos, yang memang berlaku sekarang),
    lantai acaknya ikut berubah dan tidak diam-diam salah.
    """
    from app.retrieval import db
    from app.retrieval.base import RetrievalFilters

    keluar: dict[str, list[str]] = {}
    for row in topik:
        filters = RetrievalFilters(dokumen=wilayah, **(row.get("filter") or {}))
        klausa, params = db._where(filters)
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM chunks WHERE TRUE{klausa} ORDER BY id", params)
            keluar[row["id"]] = [r[0] for r in cur.fetchall()]
    return keluar


def lantai_acak(topik: list[dict], kandidat: dict[str, list[str]], top_n: int,
                n_simulasi: int) -> dict:
    """Harapan metrik untuk urutan ACAK dari kandidat yang lolos filter."""
    from eval.eval_rag import _metrik_satu_query

    rng = random.Random(_SEED)
    kumpul: dict[str, list[float]] = {}
    per_topik: dict[str, float] = {}
    for row in topik:
        pool = kandidat.get(row["id"]) or []
        relevan = set(row["relevan"])
        if not pool:
            continue
        nilai_topik: dict[str, list[float]] = {}
        for _ in range(n_simulasi):
            k = min(top_n, len(pool))
            urutan = rng.sample(pool, k)
            m = _metrik_satu_query(urutan, relevan)
            for kunci, v in m.items():
                if kunci == "peringkat_pertama":
                    continue
                nilai_topik.setdefault(kunci, []).append(float(v))
        for kunci, vs in nilai_topik.items():
            kumpul.setdefault(kunci, []).append(statistics.fmean(vs))
        per_topik[row["id"]] = statistics.fmean(nilai_topik.get("ndcg@3", [0.0]))
    return {
        "metrik": {k: statistics.fmean(v) for k, v in kumpul.items()},
        "n_topik": len(per_topik),
        "besar_kandidat": {
            "min": min((len(v) for v in kandidat.values() if v), default=0),
            "median": statistics.median([len(v) for v in kandidat.values() if v] or [0]),
            "maks": max((len(v) for v in kandidat.values()), default=0),
        },
    }


def langit_langit(topik: list[dict], k_list: tuple[int, ...], k_operasi: int) -> dict:
    """Nilai TERTINGGI yang mungkin dicapai tiap metrik, mengingat struktur label.

    Ini bukan hiasan. Dua metrik punya langit-langit di bawah 1.0, dan membandingkannya dengan 1.0
    berarti menghukum sistem atas batas yang dibuat desain, bukan atas kegagalannya:

      * Recall@k    — kalau satu topik punya R > k chunk relevan, mustahil mengambil semuanya dalam
                      k slot. Langit-langitnya min(k,R)/R.
      * Precision@k — kalau R < k, sebagian slot PASTI terisi chunk tak relevan walau sistem
                      sempurna. Langit-langitnya min(k,R)/k. Untuk eval set yang mayoritas topiknya
                      berlabel tunggal, Precision@5 tertingginya hanya sekitar 0.2 — jadi angka P@5
                      yang "kelihatan buruk" sebenarnya nyaris mentok.

    nDCG@k, MRR, dan MAP TIDAK terkena: idealnya ikut dibatasi min(R,k), jadi 1.0 tetap bisa
    dicapai. Hit@k juga tidak — cukup satu chunk relevan yang masuk.
    """
    relevan_n = [len(set(row["relevan"])) for row in topik if row.get("relevan")]
    maks: dict[str, float] = {}
    for k in k_list:
        maks[f"recall@{k}"] = statistics.fmean(min(k, r) / r for r in relevan_n)
        maks[f"precision@{k}"] = statistics.fmean(min(k, r) / k for r in relevan_n)
        maks[f"hit@{k}"] = 1.0
        maks[f"ndcg@{k}"] = 1.0
    maks["rr"] = 1.0
    maks["ap"] = 1.0

    lebih = sum(1 for r in relevan_n if r > k_operasi)
    kurang = sum(1 for r in relevan_n if r < k_operasi)
    return {
        "maks": maks,
        "n_relevan_rata": statistics.fmean(relevan_n) if relevan_n else None,
        "topik_relevan_lebih_dari_k": lebih,
        "topik_relevan_kurang_dari_k": kurang,
        "catatan": (
            f"Recall@{k_operasi} tertinggi yang mungkin {maks[f'recall@{k_operasi}']:.3f} — "
            f"{lebih} topik punya lebih dari {k_operasi} chunk relevan. "
            f"Precision@{k_operasi} tertinggi hanya {maks[f'precision@{k_operasi}']:.3f} — "
            f"{kurang} topik punya KURANG dari {k_operasi} chunk relevan, sehingga sebagian slot "
            f"pasti terisi chunk tak relevan walau sistem sempurna. nDCG, MRR, dan MAP tidak "
            f"terkena batas ini dan tetap bisa mencapai 1.0."
        ),
    }


def main() -> None:
    import psycopg

    from eval.eval_rag import _K_LIST, _K_OPERASI, _TOP_N, _muat_topik

    ap = argparse.ArgumentParser(description="Lantai acak & langit-langit metrik utk korpus ini.")
    ap.add_argument("--eval-set", type=Path, default=_AKAR / "tests" / "eval_set_operasi.jsonl")
    ap.add_argument("--n-simulasi", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    semua = _muat_topik(args.eval_set)
    cari = [t for t in semua if t["jalur"] == "search"]
    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"')

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    try:
        kandidat = _kandidat_per_topik(conn, cari, wilayah)
    finally:
        conn.close()

    lantai = lantai_acak(cari, kandidat, _TOP_N, args.n_simulasi)
    langit = langit_langit(cari, _K_LIST, _K_OPERASI)

    hasil = {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "berkas_eval": args.eval_set.name,
        "n_simulasi": args.n_simulasi,
        "seed": _SEED,
        "k_operasi": _K_OPERASI,
        "lantai_acak": lantai,
        "langit_langit": langit,
    }
    args.out.write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")

    m = lantai["metrik"]
    bk = lantai["besar_kandidat"]
    print(f"\n=== LANTAI ACAK (n={lantai['n_topik']} topik, {args.n_simulasi} simulasi/topik) ===")
    print(f"  besar kandidat per topik: min={bk['min']} median={bk['median']:.0f} maks={bk['maks']}")
    for kunci in (f"ndcg@{_K_OPERASI}", f"recall@{_K_OPERASI}", f"hit@{_K_OPERASI}", "rr", "ap"):
        if kunci in m:
            print(f"  {kunci:12s} {m[kunci]:.4f}")
    print("\n=== LANGIT-LANGIT (tertinggi yang MUNGKIN, bukan 1.0 utk semua) ===")
    print(f"  rata-rata chunk relevan/topik: {langit['n_relevan_rata']:.2f}")
    for kunci in (f"recall@{_K_OPERASI}", f"precision@{_K_OPERASI}", "recall@5", "precision@5",
                  f"ndcg@{_K_OPERASI}", "rr", "ap"):
        if kunci in langit["maks"]:
            print(f"  {kunci:12s} maks={langit['maks'][kunci]:.3f}")
    print(f"  {langit['catatan']}")
    print(f"\n[baseline] -> {args.out}")


if __name__ == "__main__":
    main()
