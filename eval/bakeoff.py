"""Bandingkan beberapa model Groq atas gold set yang sama — skor 4 metrik + latensi per model.

MAHAL: menjalankan seluruh gold set untuk setiap model (panggilan Groq nyata berkali-kali).
Jalankan manual saat butuh (mis. sebelum memutuskan model dev), TIDAK masuk pytest/CI rutin.

Tidak menyimpulkan model "terbaik" secara otomatis — sajikan angka, keputusan model dibuat manusia
berdasarkan bukti tabel ini (CLAUDE.md: "Model dipilih via eval").

Jalankan: `python -m eval.bakeoff` dari root repo.
"""

import os
import time

from dotenv import load_dotenv

from app.retrieval.mock import MockRetriever
from eval.run_eval import HasilKasus, jalankan_gold_set, muat_gold_set

load_dotenv()

MODEL_KANDIDAT = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
# Katalog model Groq sering berubah — "gemma2-9b-it" dipakai sebelumnya sudah DECOMMISSIONED
# per klien.models.list() (400 model_decommissioned). Cek `client.models.list()` kalau kandidat
# di atas suatu saat gagal serupa, jangan asumsikan itu otomatis rate limit.
#
# 2026-08-15: "llama-3.3-70b-versatile" (default lama, lihat docs/STATUS.md baseline 100%/2.21s)
# di-decommission Groq per 2026-08-16. "qwen/qwen3.6-27b" (rekomendasi resmi Groq) DICORET dari
# daftar setelah dites: model reasoning ber-<think> yang gagal total (generation kosong,
# json_validate_failed) begitu dipaksa response_format json_object/json_schema — bukan soal
# kualitas, memang tak kompatibel dgn pipeline JSON-schema-constrained kita, butuh rework besar.
# "openai/gpt-oss-120b" lolos tes mentah (isolated call) tapi skor bakeoff pertama masih di bawah
# baseline lama (67% overall vs 100%) — sebelum diputuskan sbg default, dibandingkan dulu vs
# varian lebih kecil/cepat/murah (gpt-oss-20b, llama-3.1-8b-instant) yang justru mungkin CUKUP
# utk tugas ini (LLM di sistem cuma pelapis narasi dlm skema sempit, bukan reasoning bebas —
# lihat CLAUDE.md, "model dipilih via eval" bukan asumsi 'lebih besar = lebih perlu').

JEDA_ANTAR_KASUS_DETIK = 3.0
JEDA_ANTAR_MODEL_DETIK = 8.0


def _ringkas(hasil: list[HasilKasus]) -> dict:
    total = len(hasil)
    lulus_semua = sum(1 for h in hasil if h.lulus)

    def rate(key: str) -> float:
        n = sum(1 for h in hasil if h.metrik is not None and getattr(h.metrik, key))
        return n / total if total else 0.0

    latensi = [h.latensi_detik for h in hasil if h.error is None]
    avg_latensi = sum(latensi) / len(latensi) if latensi else 0.0
    n_error = sum(1 for h in hasil if h.error is not None)
    n_low_confidence = sum(1 for h in hasil if h.low_confidence)

    return {
        "overall": lulus_semua / total if total else 0.0,
        "faithfulness": rate("faithfulness"),
        "sitasi_grounded": rate("sitasi_grounded"),
        "numerik_benar": rate("numerik_benar"),
        "json_valid": rate("json_valid"),
        "avg_latensi_s": avg_latensi,
        "n_error": n_error,
        "n_low_confidence": n_low_confidence,
    }


def main() -> None:
    gold_set = muat_gold_set()
    retriever = MockRetriever()
    model_asli = os.getenv("LLM_MODEL")

    ringkasan_per_model: dict[str, dict] = {}

    try:
        for idx, model in enumerate(MODEL_KANDIDAT):
            if idx > 0:
                print(f"\n(jeda {JEDA_ANTAR_MODEL_DETIK:.0f}s sebelum model berikutnya, hindari rate limit RPM)")
                time.sleep(JEDA_ANTAR_MODEL_DETIK)

            print(f"\n=== Menjalankan gold set dengan model: {model} ===")
            os.environ["LLM_MODEL"] = model
            hasil = jalankan_gold_set(retriever, gold_set, jeda_detik=JEDA_ANTAR_KASUS_DETIK)
            ringkasan_per_model[model] = _ringkas(hasil)
            r = ringkasan_per_model[model]
            print(
                f"  overall={r['overall']:.0%} faithfulness={r['faithfulness']:.0%} "
                f"grounded={r['sitasi_grounded']:.0%} numerik={r['numerik_benar']:.0%} "
                f"json={r['json_valid']:.0%} avg_latensi={r['avg_latensi_s']:.2f}s "
                f"errors={r['n_error']} low_confidence={r['n_low_confidence']}"
            )
            if r["n_low_confidence"] or r["n_error"]:
                print(
                    "  [!] low_confidence/error terdeteksi — bisa berarti model memang lemah "
                    "ATAU rate limit/kuota API. Periksa sebelum menyimpulkan kualitas model."
                )
    finally:
        if model_asli is not None:
            os.environ["LLM_MODEL"] = model_asli
        else:
            os.environ.pop("LLM_MODEL", None)

    print("\n" + "=" * 100)
    header = (
        f"{'Model':<28} {'Overall':<9} {'Faith':<8} {'Ground':<8} {'Numerik':<9} "
        f"{'JSON':<7} {'AvgLat(s)':<10} {'Errors':<7} {'LowConf':<8}"
    )
    print(header)
    print("-" * len(header))
    for model, r in ringkasan_per_model.items():
        print(
            f"{model:<28} {r['overall']:<9.0%} {r['faithfulness']:<8.0%} "
            f"{r['sitasi_grounded']:<8.0%} {r['numerik_benar']:<9.0%} {r['json_valid']:<7.0%} "
            f"{r['avg_latensi_s']:<10.2f} {r['n_error']:<7} {r['n_low_confidence']:<8}"
        )

    if any(r["n_low_confidence"] or r["n_error"] for r in ringkasan_per_model.values()):
        print(
            "\n[!] Ada kasus low_confidence/error di atas — sebelum menyimpulkan model mana yang "
            "lebih baik, pastikan itu bukan kuota API habis (cek pesan RateLimitError di log run) "
            "dan bukan kelemahan model asli."
        )


if __name__ == "__main__":
    main()
