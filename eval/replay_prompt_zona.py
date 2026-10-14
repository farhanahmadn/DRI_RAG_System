"""eval/replay_prompt_zona.py — uji berpasangan: apakah prompt baru membuat narasi intensitas
benar-benar menyebut zona pemohon?

KENAPA MODUL INI ADA. `eval/metrik_generasi.py::kekhususan` melaporkan jangkar `zona_pemohon` pada
poin intensitas hanya kena 27,3% (3 dari 11), sedangkan pada `itbx` 100% (11 dari 11). Sebabnya
ditelusuri ke prompt: `PoinKonteks.zona` terisi adapter untuk ketiga poin, tapi `prompts.py` tak
pernah membacanya di blok fakta intensitas, dan 3 narasi yang kebetulan kena hanya mencocoki KODE
sub-zona ("P-1") yang sumbernya teks chunk RAG — bukan fakta permohonannya. `_baris_zona_intensitas`
menutup itu. Modul ini mengukur apakah penutupannya berhasil.

PERBANDINGAN BERPASANGAN, BUKAN AGREGAT CAMPURAN. Mengukur ulang seluruh log setelah prompt berubah
menghasilkan angka campuran — sebagian narasi dari prompt lama, sebagian dari prompt baru — dan
angka seperti itu tak bisa dipakai menyimpulkan apa pun tentang perubahannya. Di sini tiap fixture
dibandingkan dengan DIRINYA SENDIRI: lengan "sebelum" dibaca dari `logs/precheck.jsonl` (narasi yang
benar-benar pernah diterbitkan prompt lama untuk permohonan itu), lengan "sesudah" dijalankan di
sini. Hanya permohonan yang punya kedua lengan masuk ke angka berpasangan; sisanya dilaporkan
terpisah sebagai "tanpa pembanding", bukan dicampur.

INPUTNYA FIXTURE YANG SUDAH DI-COMMIT, bukan payload dari log. Dua alasan, keduanya penting:
  * Reproduksibilitas — siapa pun bisa menjalankan ulang modul ini dan mendapat himpunan input yang
    sama. Klaim dari replay yang datanya hanya ada di direktori temporer tidak bisa diperiksa, dan
    karena itu tidak boleh dikutip.
  * Privasi — payload nyata di `logs/precheck.jsonl` memuat koordinat pemohon. `eval/replay_k.py`
    sudah menetapkan prinsipnya: keluaran mentah replay TIDAK ditulis ke repo. Modul ini mematuhi
    itu — yang disimpan ke `eval/replay_zona.json` hanya HASIL UKUR per fixture (zona, status, kena
    atau tidak, token yang cocok), tanpa narasi utuh dan tanpa koordinat.

TIDAK MENULIS KE LOG PRODUKSI. `generate_poin_terdiagnosis` adalah jalur produksi untuk SATU poin
dan tak pernah memanggil `log_precheck` (itu tugas `jalankan_precheck`), jadi replay di sini tak
bisa mencemari sumber metrik — tak perlu pengalihan seperti di `eval/replay_k.py`.

GENERASI TIDAK DETERMINISTIK, jadi `--jalan` > 1 dan yang dilaporkan adalah berapa kali dari n jalan
jangkarnya kena. Satu jalan yang kebetulan bagus bukan bukti.

HANYA POIN INTENSITAS dijalankan, bukan precheck penuh — hemat kuota. Beban: n_fixture x n_jalan
panggilan Groq (TPM 8.000/kunci, TPD 200.000/kunci).

CLI:
  python -m eval.replay_prompt_zona --jalan 1          # validasi harness dulu
  python -m eval.replay_prompt_zona --jalan 3 --out eval/replay_zona.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_AKAR = Path(__file__).parent.parent
_FIXTURE = _AKAR / "tests" / "fixtures"
_LOG = _AKAR / "logs" / "precheck.jsonl"
_OUT = Path(__file__).parent / "replay_zona.json"

# Kode sub-zona dikarang: pola huruf-angka seperti "R-2"/"P-1"/"SPU-3". Dicari HANYA saat back-end
# tak mengirim `rdtr_subzone` — di situ model tak punya sumber sah untuk kode apa pun, dan
# `_baris_zona_intensitas` secara eksplisit melarangnya. Kalau cek ini tak nol, kewajiban menyebut
# zona telah berubah jadi tekanan mengarang presisi, yaitu kelas bug yang
# CAVEAT_SUBZONA_TAK_TERKONFIRMASI ada untuk mencegah.
_RE_KODE_SUBZONA = re.compile(r"\b[A-Z]{1,4}-\d\b")


def _muat_fixture() -> list[tuple[str, dict]]:
    """(nama berkas, payload L2) tiap fixture yang punya amplop back-end `{"data": ...}`."""
    keluar = []
    for p in sorted(_FIXTURE.glob("*.json")):
        try:
            isi = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload = isi.get("data") if isinstance(isi, dict) and "data" in isi else isi
        if isinstance(payload, dict) and payload.get("gate_hukum"):
            keluar.append((p.name, payload))
    return keluar


def _lengan_sebelum() -> dict[str, dict]:
    """Narasi intensitas prompt LAMA per permohonan, dari log produksi.

    Memakai `muat_kasus` apa adanya — dedup per (permohonan, poin), filter stub/mock/era-lama, dan
    deteksi jalur template ikut terpakai. Menulis ulang penyaringan itu di sini berarti dua
    definisi yang bisa menyimpang.
    """
    from eval.metrik_generasi import _jangkar_zona, _normalisasi, muat_kasus

    if not _LOG.exists():
        return {}
    kasus, _ = muat_kasus(_LOG)
    keluar: dict[str, dict] = {}
    for k in kasus:
        if k.poin_id != "intensitas":
            continue
        jangkar = _jangkar_zona(k)
        teks = _normalisasi(k.teks_narasi)
        cocok = next((a for a in jangkar if a and a in teks), None)
        keluar[k.permohonan] = {
            "jalur": k.jalur,
            "kena": bool(cocok) if k.jalur == "llm" else None,
            "token_cocok": cocok,
            "timestamp": k.timestamp,
        }
    return keluar


def _ukur(poin, keluaran, zona: str | None, subzona: str | None) -> dict:
    """Hasil ukur satu narasi — TANPA narasi utuh (lihat catatan privasi di docstring modul)."""
    from eval.metrik_generasi import (
        Kasus,
        _jalur_narasi,
        _jangkar_zona,
        _normalisasi,
    )

    kasus = Kasus(permohonan="-", timestamp="-", poin_id=poin.poin_id, konteks=poin,
                  keluaran=keluaran, zona=zona, zona_subzone=subzona,
                  jalur=_jalur_narasi(poin, keluaran))
    jangkar = _jangkar_zona(kasus)
    teks = _normalisasi(kasus.teks_narasi)
    cocok = next((a for a in jangkar if a and a in teks), None)

    # Kode sub-zona dikarang hanya bermakna saat back-end tak mengirimnya.
    mentah = f"{keluaran.reasoning_pendek} {keluaran.reasoning_panjang} {keluaran.rekomendasi.saran}"
    dikarang = sorted(set(_RE_KODE_SUBZONA.findall(mentah))) if not subzona else []

    return {
        "jalur": kasus.jalur,
        "kena": bool(cocok) if kasus.jalur == "llm" else None,
        "token_cocok": cocok,
        "jangkar_tersedia": jangkar,
        "kode_subzona_dikarang": dikarang,
        "low_confidence": keluaran.low_confidence,
        "n_sitasi": len(keluaran.sitasi),
    }


def jalankan(fixtures: list[tuple[str, dict]], n_jalan: int, jeda: float) -> list[dict]:
    from app.adapter import adaptasi
    from app.reasoning.guardrail import generate_poin_terdiagnosis
    from app.retrieval.retriever import RetrieverAsli
    from app.schemas import L2Assessment

    retriever = RetrieverAsli(
        default_wilayah=os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"'))
    keluar: list[dict] = []

    for jalan in range(1, n_jalan + 1):
        for nama, payload in fixtures:
            baris: dict = {"fixture": nama, "jalan": jalan}
            try:
                assessment = L2Assessment.model_validate(payload)
                poin = next(p for p in adaptasi(assessment).poin if p.poin_id == "intensitas")
                baris.update({
                    "permohonan": getattr(assessment, "application_number", None) or "?",
                    "zona": poin.zona, "zona_subzone": poin.zona_subzone,
                    "status": poin.status,
                })
                keluaran, diag = generate_poin_terdiagnosis(poin, retriever, assessment)
                baris.update(_ukur(poin, keluaran, poin.zona, poin.zona_subzone))
                baris["sebab"] = diag.sebab() if hasattr(diag, "sebab") else None
                baris["percobaan"] = getattr(diag, "percobaan", None)
            except Exception as exc:
                baris["gagal"] = f"{type(exc).__name__}: {exc}"[:200]
            keluar.append(baris)
            tanda = ("KENA" if baris.get("kena") else
                     "miss" if baris.get("kena") is False else
                     baris.get("jalur") or baris.get("gagal", "?"))
            print(f"  [jalan {jalan}/{n_jalan}] {nama:40s} {str(baris.get('zona')):28s} "
                  f"sub={str(baris.get('zona_subzone')):6s} -> {tanda}", flush=True)
            if jeda:
                time.sleep(jeda)
    return keluar


def ringkas(hasil: list[dict], sebelum: dict[str, dict], n_jalan: int) -> dict:
    """Agregat + pasangan per permohonan. Penyebut selalu ikut dicetak."""
    per_permohonan: dict[str, dict] = {}
    for b in hasil:
        pid = b.get("permohonan") or "?"
        e = per_permohonan.setdefault(pid, {
            "fixture": b.get("fixture"), "zona": b.get("zona"),
            "zona_subzone": b.get("zona_subzone"), "status": b.get("status"),
            "jalan_llm": 0, "jalan_kena": 0, "jalur": Counter(), "dikarang": [],
        })
        e["jalur"][b.get("jalur") or b.get("gagal", "gagal")] += 1
        if b.get("jalur") == "llm":
            e["jalan_llm"] += 1
            e["jalan_kena"] += int(bool(b.get("kena")))
        if b.get("kode_subzona_dikarang"):
            e["dikarang"].extend(b["kode_subzona_dikarang"])

    pasangan, tanpa_pembanding = [], []
    for pid, e in sorted(per_permohonan.items()):
        lama = sebelum.get(pid)
        baris = {
            "permohonan": pid, "fixture": e["fixture"], "zona": e["zona"],
            "zona_subzone": e["zona_subzone"], "status": e["status"],
            "sesudah": {"jalan_llm": e["jalan_llm"], "jalan_kena": e["jalan_kena"],
                        "jalur": dict(e["jalur"])},
            "kode_subzona_dikarang": sorted(set(e["dikarang"])),
        }
        if lama is not None:
            baris["sebelum"] = lama
            (pasangan if lama.get("kena") is not None and e["jalan_llm"]
             else tanpa_pembanding).append(baris)
        else:
            tanpa_pembanding.append(baris)

    # Angka berpasangan: hanya permohonan yang KEDUA lengannya lewat jalur LLM.
    n = len(pasangan)
    kena_lama = sum(1 for b in pasangan if b["sebelum"]["kena"])
    # "sesudah" dihitung kena kalau SELURUH jalan LLM-nya kena — ambang paling ketat, karena
    # generasi tak deterministik dan satu jalan yang kebetulan bagus bukan bukti.
    kena_baru = sum(1 for b in pasangan if b["sesudah"]["jalan_kena"] == b["sesudah"]["jalan_llm"])
    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_jalan": n_jalan,
        "berpasangan": {
            "n": n,
            "kena_prompt_lama": kena_lama,
            "kena_prompt_baru": kena_baru,
            "nilai_lama": (kena_lama / n) if n else None,
            "nilai_baru": (kena_baru / n) if n else None,
            "membaik": [b["permohonan"] for b in pasangan
                        if not b["sebelum"]["kena"]
                        and b["sesudah"]["jalan_kena"] == b["sesudah"]["jalan_llm"]],
            "memburuk": [b["permohonan"] for b in pasangan
                         if b["sebelum"]["kena"] and b["sesudah"]["jalan_kena"] == 0],
        },
        "kode_subzona_dikarang": sorted({k for b in pasangan + tanpa_pembanding
                                         for k in b["kode_subzona_dikarang"]}),
        "pasangan": pasangan,
        "tanpa_pembanding": tanpa_pembanding,
    }


def cetak(h: dict) -> None:
    b = h["berpasangan"]
    print("\n=== PERBANDINGAN BERPASANGAN (prompt lama vs baru, permohonan yang SAMA) ===")
    if not b["n"]:
        print("  TAK TERUKUR — tak ada permohonan yang kedua lengannya lewat jalur LLM")
    else:
        print(f"  n permohonan berpasangan   : {b['n']}")
        print(f"  jangkar zona, prompt LAMA  : {b['kena_prompt_lama']}/{b['n']} "
              f"({b['nilai_lama']:.1%})")
        print(f"  jangkar zona, prompt BARU  : {b['kena_prompt_baru']}/{b['n']} "
              f"({b['nilai_baru']:.1%})   (kena = SELURUH jalan LLM menyebut zonanya)")
        if b["membaik"]:
            print(f"  membaik  : {', '.join(b['membaik'])}")
        if b["memburuk"]:
            print(f"  MEMBURUK : {', '.join(b['memburuk'])}")
    dk = h["kode_subzona_dikarang"]
    print(f"\n  kode sub-zona dikarang (saat BE tak mengirimnya): "
          f"{', '.join(dk) if dk else 'TIDAK ADA'}")

    print(f"\n=== TANPA PEMBANDING ({len(h['tanpa_pembanding'])}) ===")
    for e in h["tanpa_pembanding"]:
        lama = (e.get("sebelum") or {}).get("jalur", "tak ada di log")
        print(f"  {e['permohonan']:18s} {str(e['zona']):26s} status={str(e['status']):16s} "
              f"sesudah={dict(e['sesudah']['jalur'])}  sebelum={lama}")

    print("\n=== RINCI PER PERMOHONAN BERPASANGAN ===")
    for e in h["pasangan"]:
        s = e["sesudah"]
        print(f"  {e['permohonan']:18s} {str(e['zona']):26s} sub={str(e['zona_subzone']):6s} "
              f"status={str(e['status']):16s} "
              f"lama={'KENA' if e['sebelum']['kena'] else 'miss':4s} "
              f"baru={s['jalan_kena']}/{s['jalan_llm']} jalan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Uji berpasangan jangkar zona pada narasi intensitas (prompt lama vs baru).")
    ap.add_argument("--jalan", type=int, default=1,
                    help="berapa kali tiap fixture dijalankan (generasi tak deterministik)")
    ap.add_argument("--jeda", type=float, default=2.0, help="jeda detik antar panggilan (rate limit)")
    ap.add_argument("--maks", type=int, default=None, help="batasi jumlah fixture")
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    fixtures = _muat_fixture()
    if args.maks:
        fixtures = fixtures[:args.maks]
    if not fixtures:
        raise SystemExit(f"tak ada fixture ber-payload L2 di {_FIXTURE}")
    print(f"[replay-zona] {len(fixtures)} fixture x {args.jalan} jalan "
          f"= {len(fixtures) * args.jalan} panggilan Groq")

    sebelum = _lengan_sebelum()
    print(f"[replay-zona] lengan 'sebelum' dari log: {len(sebelum)} permohonan")

    hasil = jalankan(fixtures, args.jalan, args.jeda)
    ringkasan = ringkas(hasil, sebelum, args.jalan)
    cetak(ringkasan)
    args.out.write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[replay-zona] -> {args.out}")


if __name__ == "__main__":
    main()
