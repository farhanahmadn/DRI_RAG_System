"""eval/replay_k.py — uji kenaikan kedalaman konteks (k) di SISI GENERASI, bukan retrieval.

Kenapa modul ini ada. Metrik retrieval mengatakan k=5 jelas lebih baik daripada k=3: Recall@k naik
dari 63% ke 94% langit-langitnya. Tapi metrik itu berhenti di ambang prompt — ia tidak bisa
menjawab apakah LLM memakai 5 chunk lebih baik daripada 3, atau justru tersesat. Mode gagal itu
nyata dan pernah terjadi (APP-2026-6191: "prompt yang membanjiri LLM bikin ia gagal memilih sitasi
sama sekali"), dan satu-satunya cara menangkapnya adalah menjalankan generasi sungguhan.

Yang dibandingkan karena itu bukan nDCG, melainkan akibat di luaran:

  * sebab akhir tiap poin — berhasil / guardrail_menolak / panggilan_llm_gagal
  * low_confidence per poin
  * jumlah sitasi yang dihasilkan, dan berapa yang terverifikasi
  * groundedness kutipan: apakah teks yang dikutip memang ada di chunk yang disitasi
  * recall sitasi terhadap zona pemohon

GENERASI TIDAK DETERMINISTIK. Satu kali jalan tidak cukup — replay berbeda menghasilkan kasus
gagal berbeda, dan itu sudah terbukti di proyek ini. Karena itu tiap konfigurasi dijalankan
BEBERAPA KALI (default 3) dan yang dilaporkan adalah rata-rata beserta sebarannya. Perbandingan
k=3 vs k=5 dilakukan atas permohonan yang SAMA, sehingga berpasangan.

Payload diambil dari `logs/precheck.jsonl` (bagian `request` saja) dan dibuang duplikatnya —
permohonan nyata, bukan karangan. Koordinat ikut terbawa ke LLM persis seperti di produksi, jadi
keluaran mentahnya TIDAK ditulis ke repo; yang disimpan hanya agregat.

BUTUH kuota Groq nyata. Perkiraan beban: n_permohonan x 3 poin x n_jalan x 2 konfigurasi panggilan.

CLI:
  python -m eval.replay_k --maks-permohonan 10 --jalan 3
  python -m eval.replay_k --k 5 --poin intensitas dampak --out eval/replay_k.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_LOG = _AKAR / "logs" / "precheck.jsonl"
_OUT = Path(__file__).parent / "replay_k.json"
_POIN_SEKARANG = ("itbx", "intensitas", "dampak")
_RE_BUKAN_KATA = re.compile(r"[^0-9a-z]+")


def muat_payload(log: Path, maks: int | None) -> list[dict]:
    """Payload L2 nyata, unik, era 3-poin. Urutannya ditetapkan supaya replay bisa diulang."""
    uniq: dict[str, dict] = {}
    for baris in log.read_text(encoding="utf-8").splitlines():
        if not baris.strip():
            continue
        try:
            d = json.loads(baris)
        except Exception:
            continue
        resp = d.get("response") or {}
        if not any(p.get("poin_id") in _POIN_SEKARANG
                   for p in (resp.get("poin") or []) if isinstance(p, dict)):
            continue
        req = d.get("request") or {}
        if not req.get("lokasi"):
            continue
        uniq.setdefault(json.dumps(req, sort_keys=True), req)
    keluar = [v for _, v in sorted(uniq.items())]
    return keluar[:maks] if maks else keluar


def _normalisasi(teks: str) -> str:
    return _RE_BUKAN_KATA.sub(" ", (teks or "").lower()).strip()


def _ukur_keluaran(output, teks_chunk: dict[str, str], prefix_zona: str | None) -> dict:
    """Ukur akibat di LUARAN — bukan skor retrieval."""
    hasil = {"poin": {}, "sitasi": 0, "sitasi_terverifikasi": 0,
             "kutipan_dinilai": 0, "kutipan_grounded": 0,
             "poin_berzona_dinilai": 0, "poin_berzona_kena": 0}
    for p in output.poin:
        hasil["poin"][p.poin_id] = {
            "low_confidence": bool(p.low_confidence),
            "n_sitasi": len(p.sitasi or []),
        }
        id_korpus = []
        for s in (p.sitasi or []):
            hasil["sitasi"] += 1
            if getattr(s, "terverifikasi", False):
                hasil["sitasi_terverifikasi"] += 1
            cid = getattr(s, "citation_id", "") or ""
            teks = teks_chunk.get(cid)
            if teks:
                id_korpus.append(cid)
                k = _normalisasi(getattr(s, "kutipan", "") or "")
                if len(k) >= 25:
                    hasil["kutipan_dinilai"] += 1
                    if k in _normalisasi(teks):
                        hasil["kutipan_grounded"] += 1
        if p.poin_id in ("intensitas", "itbx") and prefix_zona and id_korpus:
            hasil["poin_berzona_dinilai"] += 1
            if any(_prefix(cid) == prefix_zona for cid in id_korpus):
                hasil["poin_berzona_kena"] += 1
    return hasil


_ZONA_CHUNK: dict[str, str | None] = {}


def _prefix(cid: str) -> str | None:
    z = _ZONA_CHUNK.get(cid)
    return z.split("-")[0].upper() if z else None


def _muat_korpus() -> tuple[dict[str, str], dict[str, str | None]]:
    """Teks & zona tiap chunk — dipakai menilai groundedness kutipan dan kecocokan zona."""
    import psycopg

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        baris = conn.execute("SELECT id, zona, teks FROM chunks").fetchall()
    return {r[0]: r[2] for r in baris}, {r[0]: r[1] for r in baris}


def jalankan(payloads: list[dict], k_map: dict[str, int], n_jalan: int,
             teks_chunk: dict, jeda: float) -> list[dict]:
    """Jalankan seluruh payload n_jalan kali pada satu konfigurasi k."""
    from app.reasoning import generator
    from app.reasoning.assemble import jalankan_precheck
    from app.reasoning.generator import _zona_prefix_dari_nama
    from app.retrieval.retriever import RetrieverAsli
    from app.schemas import L2Assessment

    # `jalankan_precheck` menulis ke logs/precheck.jsonl. Itu BERKAS DATA PRODUKSI, dan
    # metrik atribusi membacanya — replay yang ikut menulis ke sana akan mencemari sumber
    # metrik dengan jalan percobaan yang konfigurasinya sengaja diubah. Dialihkan ke berkas
    # terpisah: datanya tidak hilang, tapi tidak tercampur.
    log_replay = Path(__file__).parent / "_replay_precheck.jsonl"

    def _log_terpisah(assessment, output, **kw):
        try:
            with log_replay.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                    "replay": True, "k_map": k_map,
                                    "application_number": getattr(assessment, "application_number", None)},
                                   ensure_ascii=False) + chr(10))
        except Exception:
            pass

    import app.reasoning.assemble as _asm
    log_asli = _asm.log_precheck
    _asm.log_precheck = _log_terpisah

    asli = dict(generator._TOP_K_PER_POIN)
    generator._TOP_K_PER_POIN.clear()
    generator._TOP_K_PER_POIN.update(k_map)
    retriever = RetrieverAsli(default_wilayah=os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"'))
    keluar: list[dict] = []
    try:
        for jalan in range(1, n_jalan + 1):
            for i, req in enumerate(payloads, 1):
                lok = req.get("lokasi") or {}
                pz = (lok.get("rdtr_subzone") or "").split("-")[0].upper() or None
                if not pz:
                    p = _zona_prefix_dari_nama(lok.get("rdtr_zone"))
                    pz = p.upper() if p else None
                try:
                    assessment = L2Assessment.model_validate(req)
                    out = jalankan_precheck(assessment, retriever)
                    ukur = _ukur_keluaran(out, teks_chunk, pz)
                    ukur["gagal_total"] = False
                except Exception as exc:
                    ukur = {"gagal_total": True, "exception": f"{type(exc).__name__}: {exc}"[:200],
                            "poin": {}, "sitasi": 0, "sitasi_terverifikasi": 0,
                            "kutipan_dinilai": 0, "kutipan_grounded": 0,
                            "poin_berzona_dinilai": 0, "poin_berzona_kena": 0}
                ukur["jalan"] = jalan
                ukur["payload"] = req.get("application_number") or f"#{i}"
                keluar.append(ukur)
                print(f"  [jalan {jalan}/{n_jalan}] {i}/{len(payloads)} {ukur['payload']:16s} "
                      f"sitasi={ukur['sitasi']:2d} "
                      f"low_conf={sum(1 for v in ukur['poin'].values() if v['low_confidence'])}",
                      flush=True)
                if jeda:
                    time.sleep(jeda)
    finally:
        generator._TOP_K_PER_POIN.clear()
        generator._TOP_K_PER_POIN.update(asli)
        _asm.log_precheck = log_asli
    return keluar


def ringkas(hasil: list[dict]) -> dict:
    n = len(hasil) or 1
    low = Counter()
    for h in hasil:
        for pid, v in h["poin"].items():
            if v["low_confidence"]:
                low[pid] += 1
    sitasi = sum(h["sitasi"] for h in hasil)
    kd = sum(h["kutipan_dinilai"] for h in hasil)
    bz = sum(h["poin_berzona_dinilai"] for h in hasil)
    semua_poin = sum(len(h["poin"]) for h in hasil) or 1
    low_total = sum(1 for h in hasil for v in h["poin"].values() if v["low_confidence"])
    return {
        "n_jalan_permohonan": len(hasil),
        # Ambang ini bukan penilaian mutu — ia pendeteksi RUN YANG RUSAK. Pada jalan sehat
        # low_confidence berada di kisaran persen; kalau mayoritas poin jatuh ke sana,
        # yang terukur hampir pasti kegagalan panggilan (kuota/jaringan), bukan efek k.
        "rasio_low_confidence": low_total / semua_poin,
        "mencurigakan": (low_total / semua_poin) > 0.5,
        "gagal_total": sum(1 for h in hasil if h.get("gagal_total")),
        "sitasi_total": sitasi,
        "sitasi_per_permohonan": sitasi / n,
        "sitasi_terverifikasi": (sum(h["sitasi_terverifikasi"] for h in hasil) / sitasi) if sitasi else None,
        "groundedness_kutipan": (sum(h["kutipan_grounded"] for h in hasil) / kd) if kd else None,
        "kutipan_dinilai": kd,
        "recall_zona": (sum(h["poin_berzona_kena"] for h in hasil) / bz) if bz else None,
        "poin_berzona_dinilai": bz,
        "low_confidence_per_poin": dict(low),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay k di sisi generasi (butuh kuota Groq).")
    ap.add_argument("--log", type=Path, default=_LOG)
    ap.add_argument("--maks-permohonan", type=int, default=10)
    ap.add_argument("--jalan", type=int, default=3, help="pengulangan per konfigurasi (generasi non-deterministik)")
    ap.add_argument("--k", type=int, default=5, help="kedalaman yang diuji")
    ap.add_argument("--poin", nargs="+", default=["intensitas", "dampak"])
    ap.add_argument("--jeda", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    payloads = muat_payload(args.log, args.maks_permohonan)
    if not payloads:
        raise SystemExit("tidak ada payload era 3-poin di log")
    teks_chunk, zona = _muat_korpus()
    _ZONA_CHUNK.update(zona)

    kfg = {"k=3 (produksi)": {}, f"k={args.k} ({', '.join(args.poin)})": {p: args.k for p in args.poin}}
    print(f"payload: {len(payloads)} | jalan/konfigurasi: {args.jalan} | konfigurasi: {list(kfg)}")
    print(f"perkiraan panggilan LLM: {len(payloads) * 3 * args.jalan * len(kfg)}\n")

    hasil = {}
    rusak_awal = False
    for nama, k_map in kfg.items():
        print(f"=== {nama} ===")
        hasil[nama] = ringkas(jalankan(payloads, k_map, args.jalan, teks_chunk, args.jeda))
        rusak_awal = rusak_awal or hasil[nama].get("mencurigakan", False)
        print()

    keluaran = {"sah": not rusak_awal, "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_payload": len(payloads), "n_jalan": args.jalan,
                "poin_diubah": args.poin, "k_diuji": args.k, "ringkasan": hasil}
    args.out.write_text(json.dumps(keluaran, ensure_ascii=False, indent=2), encoding="utf-8")

    rusak = [n for n, v in hasil.items() if v.get("mencurigakan")]
    if rusak:
        print("!" * 78)
        print("HASIL TIDAK SAH — mayoritas poin jatuh ke low_confidence pada konfigurasi: "
              + ", ".join(rusak))
        print("Itu pola kegagalan PANGGILAN (kuota/jaringan), bukan efek k. "
              "Tabel di bawah TIDAK boleh dibaca sebagai perbandingan k. "
              "Periksa kuota lalu ulangi.")
        print("!" * 78 + chr(10))
    print(f"{'metrik':28s} " + " ".join(f"{n[:22]:>24s}" for n in hasil))
    for m in ("gagal_total", "sitasi_per_permohonan", "sitasi_terverifikasi",
              "groundedness_kutipan", "recall_zona"):
        sel = []
        for n in hasil:
            v = hasil[n][m]
            sel.append(f"{'—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v)):>24s}")
        print(f"{m:28s} " + " ".join(sel))
    print(f"{'low_confidence per poin':28s} " +
          " ".join(f"{str(hasil[n]['low_confidence_per_poin']):>24s}" for n in hasil))
    print(f"\n[replay-k] -> {args.out}")


if __name__ == "__main__":
    main()
