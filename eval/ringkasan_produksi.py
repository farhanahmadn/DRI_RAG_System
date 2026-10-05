"""eval/ringkasan_produksi.py — satu tabel yang boleh dibaca sebagai KINERJA PRODUKSI.

Kenapa modul ini ada. Tabel titik operasi di laporan merata-ratakan seluruh topik dengan bobot
sama, dan itu BUKAN angka produksi: 52 dari 214 topik menguji jalur `itbx` lewat `search` yang di
produksi hampir tak pernah menyala, dan puluhan topik lain adalah lengan KONTRAFAKTUAL yang
sengaja dibuat untuk ablasi (query pendek `kdb`, topik dampak warisan berlabel seeded). Semuanya
perlu ada untuk membuktikan sebab-akibat, tapi tak satu pun mewakili perilaku produksi.

Modul ini menyusun bacaan per-poin: untuk tiap poin, cabang mana yang BENAR-BENAR dipakai
produksi, berapa skornya, dan — yang paling menentukan — berapa persen dari LANGIT-LANGIT yang
mungkin dicapai pada subset itu. Lantai acak dan langit-langit dihitung per subset, bukan dipinjam
dari angka global, karena keduanya bergantung pada jumlah label dan besar kandidat subset itu
sendiri.

Jalur produksi tiap poin diturunkan dari kode (`generator._TANPA_LEXICAL_PER_POIN`), bukan ditulis
ulang di sini, sehingga tabel ini tak bisa menyimpang dari apa yang benar-benar dijalankan.

Bobot trafik dihitung HANYA atas era desain 3-poin yang berjalan sekarang. Pernah keliru sekali:
menghitung seluruh log mencampur 309 permohonan era 8-indikator lama yang semuanya tanpa zona
induk, menghasilkan 73% padahal angka sebenarnya 96.4%.

Tidak memanggil API embedding/rerank; hanya DB untuk besar kandidat per topik.

CLI:
  python -m eval.ringkasan_produksi
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_OUT = Path(__file__).parent / "ringkasan_produksi.json"
_METRIK = ("ndcg@3", "recall@3", "hit@3", "mrr", "map")
_POIN_SEKARANG = ("itbx", "intensitas", "dampak")


def bobot_trafik(log: Path) -> dict:
    """Proporsi permohonan yang membawa sub-zona presisi vs tidak, era 3-poin saja.

    Inilah yang menentukan cabang mana yang dominan. Era lama dibuang karena jalur reasoning-nya
    sudah tidak ada — memasukkannya hanya mengaburkan bobot (lihat catatan di docstring modul).
    """
    ada = tanpa = 0
    if not log.exists():
        return {"ada_subzona": None, "tanpa_subzona": None, "n": 0, "sumber": "log tidak ada"}
    for baris in log.read_text(encoding="utf-8").splitlines():
        if not baris.strip():
            continue
        try:
            d = json.loads(baris)
        except Exception:
            continue
        poin = [p for p in ((d.get("response") or {}).get("poin") or []) if isinstance(p, dict)]
        if not any(p.get("poin_id") in _POIN_SEKARANG for p in poin):
            continue
        lok = ((d.get("request") or {}).get("lokasi") or {})
        if (lok.get("rdtr_subzone") or "").strip():
            ada += 1
        else:
            tanpa += 1
    n = ada + tanpa
    return {
        "ada_subzona": (ada / n) if n else None,
        "tanpa_subzona": (tanpa / n) if n else None,
        "n": n,
        "sumber": f"{log.name}, era 3-poin saja",
    }


def _cabang_produksi() -> list[dict]:
    """Cabang yang dipakai produksi, dipetakan ke awalan id topik di eval set.

    Awalan id mengikuti konvensi penamaan `eval/bangun_eval_set_operasi.py`. Topik yang SENGAJA
    tidak dipakai di sini beserta alasannya:
      * `intensitas-kdb-*`      — query pendek, kontrafaktual sejak gating query dibuka
      * `dampak-resapan/banjir/...` — warisan eval set lama, berlabel seeded & query non-produksi
      * `itbx-kegiatan-*`       — jalur search `itbx`, cadangan yang hampir tak pernah menyala
    """
    from app.reasoning.generator import _TANPA_LEXICAL_PER_POIN
    from eval.eval_rag import _CABANG_PRODUKSI

    def _kfg(poin: str) -> str:
        return "dense+rerank" if poin in _TANPA_LEXICAL_PER_POIN else "rrf+rerank"

    return [{"poin": poin, "cabang": cabang, "bobot": bobot,
             "konfigurasi": _kfg(poin), "awalan": awalan}
            for awalan, poin, cabang, bobot in _CABANG_PRODUKSI]


def susun(hasil: dict, eval_set: Path, log: Path, n_simulasi: int) -> dict:
    import psycopg

    from eval.baseline_acak import _kandidat_per_topik, langit_langit, lantai_acak
    from eval.eval_rag import _K_LIST, _K_OPERASI, _muat_topik

    detail, pq = hasil["detail_per_query"], hasil["skor_per_query"]
    peta_topik = {t["id"]: t for t in _muat_topik(eval_set) if t["jalur"] == "search"}
    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"')
    bobot = bobot_trafik(log)

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    baris: list[dict] = []
    try:
        for cab in _cabang_produksi():
            idx = [i for i, d in enumerate(detail) if d["query"].startswith(cab["awalan"])]
            topik = [peta_topik[detail[i]["query"]] for i in idx if detail[i]["query"] in peta_topik]
            if not topik:
                continue
            kand = _kandidat_per_topik(conn, topik, wilayah)
            lantai = lantai_acak(topik, kand, max(_K_LIST), n_simulasi)["metrik"]
            langit = langit_langit(topik, _K_LIST, _K_OPERASI)["maks"]
            kfg = cab["konfigurasi"]
            metrik = {}
            for m in _METRIK:
                base = {"mrr": "rr", "map": "ap"}.get(m, m)
                # `skor_per_query` memakai nama KLAIM (mrr/map), bukan nama per-topik (rr/ap).
                ukur = statistics.fmean(pq[kfg][i][m] for i in idx)
                lg = langit.get(base)
                metrik[m] = {
                    "ukur": ukur,
                    "lantai": lantai.get(base),
                    "langit": lg,
                    "dari_langit": (ukur / lg) if lg else None,
                }
            baris.append({**{k: v for k, v in cab.items() if k != "awalan"},
                          "n": len(idx), "bobot_nilai": bobot.get(cab["bobot"]),
                          "metrik": metrik})
    finally:
        conn.close()

    # itbx: jalur get_by_reference, murni SQL. Tak ada lantai acak yang sebanding — semestanya
    # bukan seluruh korpus yang lolos filter, melainkan daftar kandidat yang dikembalikan rujukan.
    anc = hasil.get("anchor") or {}
    if anc.get("n_query"):
        a = anc["agregat"][anc["produksi"]]
        baris.insert(0, {
            "poin": "itbx", "cabang": "get_by_reference (anchor dari back-end)",
            "bobot": None, "konfigurasi": anc["produksi"], "n": anc["n_query"],
            "bobot_nilai": 1.0,
            "metrik": {m: {"ukur": a.get(m), "lantai": None, "langit": 1.0,
                           "dari_langit": a.get(m)} for m in _METRIK},
            "catatan": "Jalur SQL; tidak melewati embedding/rerank sehingga kebal pilihan fusi.",
        })

    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "berkas_eval": eval_set.name,
        "n_simulasi_lantai": n_simulasi,
        "bobot_trafik": bobot,
        "baris": baris,
    }


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def cetak(r: dict) -> None:
    b = r["bobot_trafik"]
    print(f"\nbobot trafik ({b['sumber']}, n={b['n']}): "
          f"sub-zona ada {_pct(b['ada_subzona'])} | tanpa sub-zona {_pct(b['tanpa_subzona'])}")
    print(f"\n{'poin · cabang':52s} {'n':>4s} {'metrik':10s} {'lantai':>8s} {'ukur':>7s} "
          f"{'langit':>7s} {'%langit':>8s}")
    for row in r["baris"]:
        judul = f"{row['poin']} · {row['cabang']}"
        for i, (m, e) in enumerate(row["metrik"].items()):
            lt = "—" if e["lantai"] is None else f"{e['lantai']:.4f}"
            print(f"{judul if i == 0 else '':52s} {row['n'] if i == 0 else '':>4} "
                  f"{m:10s} {lt:>8s} {e['ukur']:7.3f} {e['langit']:7.3f} "
                  f"{_pct(e['dari_langit']):>8s}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ringkasan kinerja per-poin pada jalur produksi.")
    ap.add_argument("--hasil", type=Path, default=Path(__file__).parent / "laporan_rag.json")
    ap.add_argument("--eval-set", type=Path, default=_AKAR / "tests" / "eval_set_operasi.jsonl")
    ap.add_argument("--log", type=Path, default=_AKAR / "logs" / "precheck.jsonl")
    ap.add_argument("--n-simulasi", type=int, default=400)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    if not args.hasil.exists():
        raise SystemExit(f"hasil evaluasi tidak ada: {args.hasil} — jalankan eval_rag dulu")
    hasil = json.loads(args.hasil.read_text(encoding="utf-8"))
    r = susun(hasil, args.eval_set, args.log, args.n_simulasi)
    cetak(r)
    args.out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ringkasan] -> {args.out}")


if __name__ == "__main__":
    main()
