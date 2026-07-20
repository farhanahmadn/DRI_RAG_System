"""Bandingkan beberapa model Groq atas gold set yang sama — skor 4 metrik + latensi per model.

MAHAL: menjalankan seluruh gold set untuk setiap model (panggilan Groq nyata berkali-kali).
Jalankan manual saat butuh (mis. sebelum memutuskan model dev), TIDAK masuk pytest/CI rutin.

Tidak menyimpulkan model "terbaik" secara otomatis — sajikan angka, keputusan model dibuat manusia
berdasarkan bukti tabel ini (CLAUDE.md: "Model dipilih via eval").

Jalankan: `python -m eval.bakeoff` dari root repo.
"""

import os

from dotenv import load_dotenv

from app.retrieval.mock import MockRetriever
from eval.run_eval import HasilKasus, jalankan_gold_set, muat_gold_set

load_dotenv()

MODEL_KANDIDAT = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


def _ringkas(hasil: list[HasilKasus]) -> dict:
    total = len(hasil)
    lulus_semua = sum(1 for h in hasil if h.lulus)

    def rate(key: str) -> float:
        n = sum(1 for h in hasil if h.metrik is not None and getattr(h.metrik, key))
        return n / total if total else 0.0

    latensi = [h.latensi_detik for h in hasil if h.error is None]
    avg_latensi = sum(latensi) / len(latensi) if latensi else 0.0
    n_error = sum(1 for h in hasil if h.error is not None)

    return {
        "overall": lulus_semua / total if total else 0.0,
        "faithfulness": rate("faithfulness"),
        "sitasi_grounded": rate("sitasi_grounded"),
        "numerik_benar": rate("numerik_benar"),
        "json_valid": rate("json_valid"),
        "avg_latensi_s": avg_latensi,
        "n_error": n_error,
    }


def main() -> None:
    gold_set = muat_gold_set()
    retriever = MockRetriever()
    model_asli = os.getenv("LLM_MODEL")

    ringkasan_per_model: dict[str, dict] = {}

    try:
        for model in MODEL_KANDIDAT:
            print(f"\n=== Menjalankan gold set dengan model: {model} ===")
            os.environ["LLM_MODEL"] = model
            hasil = jalankan_gold_set(retriever, gold_set)
            ringkasan_per_model[model] = _ringkas(hasil)
            r = ringkasan_per_model[model]
            print(
                f"  overall={r['overall']:.0%} faithfulness={r['faithfulness']:.0%} "
                f"grounded={r['sitasi_grounded']:.0%} numerik={r['numerik_benar']:.0%} "
                f"json={r['json_valid']:.0%} avg_latensi={r['avg_latensi_s']:.2f}s "
                f"errors={r['n_error']}"
            )
    finally:
        if model_asli is not None:
            os.environ["LLM_MODEL"] = model_asli
        else:
            os.environ.pop("LLM_MODEL", None)

    print("\n" + "=" * 100)
    header = (
        f"{'Model':<28} {'Overall':<9} {'Faith':<8} {'Ground':<8} {'Numerik':<9} "
        f"{'JSON':<7} {'AvgLat(s)':<10} {'Errors':<7}"
    )
    print(header)
    print("-" * len(header))
    for model, r in ringkasan_per_model.items():
        print(
            f"{model:<28} {r['overall']:<9.0%} {r['faithfulness']:<8.0%} "
            f"{r['sitasi_grounded']:<8.0%} {r['numerik_benar']:<9.0%} {r['json_valid']:<7.0%} "
            f"{r['avg_latensi_s']:<10.2f} {r['n_error']:<7}"
        )


if __name__ == "__main__":
    main()
