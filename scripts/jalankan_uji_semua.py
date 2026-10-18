"""Jalankan semua payload uji lewat alur SEBENARNYA (retriever asli + LLM asli), simpan hasil tiap permintaan.

Tiap payload dijalankan berurutan (hemat rate limit Groq) lewat `jalankan_precheck`, fungsi yang sama dengan
endpoint /reasoning. Hasil disimpan sebagai berkas JSON terpisah (isi = respons OutputL3 apa adanya) di folder
keluaran, ditambah `_ringkasan.json` berisi durasi dan status tiap permintaan. Setiap permintaan juga
tercatat otomatis di logs/precheck.jsonl seperti biasa.

Jalankan dari root repo:  python -m scripts.jalankan_uji_semua [folder_keluaran]
Butuh: Postgres menyala (RETRIEVER=asli), kunci Groq dan Jina sehat (cek: python -m scripts.check_api_keys).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.api.dependencies import get_retriever  # noqa: E402
from app.reasoning.assemble import jalankan_precheck  # noqa: E402
from app.schemas import L2Envelope  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "tests" / "payload_uji"
URUTAN = [
    ROOT / "tests" / "fixtures" / "l2_sample_amplop_0014.json",
    *sorted(PAYLOAD.glob("hidrologi_[1-4]_*.json")),
    *sorted(PAYLOAD.glob("hidrologi_set2_*.json")),
]


def main() -> None:
    keluar = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "hasil_uji" / date.today().isoformat()
    keluar.mkdir(parents=True, exist_ok=True)
    retriever = get_retriever()
    ringkasan = []
    saring = os.getenv("UJI_FILTER", "")  # potongan nama berkas, dipisah koma; kosong = semua
    for berkas in [b for b in URUTAN if any(t in b.name for t in saring.split(","))]:
        envelope = L2Envelope.model_validate(json.loads(berkas.read_text(encoding="utf-8")))
        nomor = envelope.data.application_number
        if ringkasan and os.getenv("UJI_JEDA_S"):
            time.sleep(float(os.environ["UJI_JEDA_S"]))  # beri jendela token menit Groq pulih antar payload
        mulai = time.perf_counter()
        status, hasil = "ok", None
        try:
            hasil = jalankan_precheck(envelope.data, retriever).model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 — pelari uji: catat apa pun lalu lanjut
            status = f"error: {type(exc).__name__}: {exc}"
        durasi = round(time.perf_counter() - mulai, 1)
        if hasil is not None:
            (keluar / f"{berkas.stem}.json").write_text(
                json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")
        baris = {"berkas": berkas.name, "application_number": nomor, "durasi_detik": durasi, "status": status,
                 "low_confidence_keseluruhan": None if hasil is None else hasil["low_confidence_keseluruhan"],
                 "poin_low_confidence": [] if hasil is None else
                 [p["poin_id"] for p in hasil["poin"] if p["low_confidence"]]}
        ringkasan.append(baris)
        print(json.dumps(baris, ensure_ascii=False), flush=True)
    (keluar / "_ringkasan.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selesai. Hasil di: {keluar}", flush=True)


if __name__ == "__main__":
    main()
