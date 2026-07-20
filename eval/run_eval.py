"""Jalankan gold set lewat jalankan_precheck (MockRetriever) & cetak tabel hasil 4 metrik.

BUTUH GROQ_API_KEY — memanggil Groq beneran (berbeda dari pytest yang mostly-mock). Ini tempat
kita mengevaluasi KUALITAS OUTPUT LLM, bukan cuma kebenaran kode.

Jalankan: `python -m eval.run_eval` dari root repo.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.reasoning.assemble import jalankan_precheck
from app.reasoning.generator import ambil_chunks_pendukung
from app.retrieval.base import Retriever
from app.retrieval.mock import MockRetriever
from app.schemas import JejakAturanRequest
from eval.metrics import HasilMetrik, evaluasi_kasus

load_dotenv()

GOLD_SET_PATH = Path(__file__).parent / "gold_set.jsonl"


@dataclass
class HasilKasus:
    nama: str
    metrik: HasilMetrik | None
    latensi_detik: float
    error: str | None
    low_confidence: bool

    @property
    def lulus(self) -> bool:
        return self.error is None and self.metrik is not None and self.metrik.lulus_semua


def muat_gold_set(path: Path = GOLD_SET_PATH) -> list[dict]:
    kasus_list = []
    with open(path, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if baris:
                kasus_list.append(json.loads(baris))
    return kasus_list


def jalankan_gold_set(retriever: Retriever, gold_set: list[dict]) -> list[HasilKasus]:
    """Jalankan setiap kasus gold set lewat jalankan_precheck, kembalikan hasil per kasus."""
    hasil: list[HasilKasus] = []

    for kasus in gold_set:
        nama = kasus["nama"]
        expected = kasus["expected"]
        try:
            request = JejakAturanRequest.model_validate(kasus["request"])
            indikator = request.indikator[0]
            chunks = ambil_chunks_pendukung(indikator, retriever)

            diharapkan = set(expected.get("citation_ids_diharapkan") or [])
            tersedia = {c.id for c in chunks}
            if diharapkan and not diharapkan <= tersedia:
                print(
                    f"  [WARNING] {nama}: citation_ids_diharapkan {sorted(diharapkan - tersedia)} "
                    "tidak pernah diretrieve — cek label gold, bukan masalah kualitas LLM."
                )

            mulai = time.perf_counter()
            output = jalankan_precheck(request, retriever)
            latensi = time.perf_counter() - mulai

            poin = output.poin[0]
            metrik = evaluasi_kasus(poin, indikator, chunks, output, expected)
            hasil.append(
                HasilKasus(
                    nama=nama,
                    metrik=metrik,
                    latensi_detik=latensi,
                    error=None,
                    low_confidence=poin.low_confidence,
                )
            )
        except Exception as exc:
            hasil.append(
                HasilKasus(nama=nama, metrik=None, latensi_detik=0.0, error=str(exc), low_confidence=False)
            )

    return hasil


def _tanda(nilai: bool) -> str:
    return "OK" if nilai else "GAGAL"


def cetak_tabel(hasil: list[HasilKasus]) -> bool:
    header = f"{'Kasus':<26} {'Faith':<7} {'Ground':<7} {'Numerik':<8} {'JSON':<6} {'LowConf':<8} {'Hasil':<6}"
    print(header)
    print("-" * len(header))

    for h in hasil:
        if h.error is not None:
            print(f"{h.nama:<26} ERROR: {h.error}")
            continue
        m = h.metrik
        low_conf = "!" if h.low_confidence else ""
        print(
            f"{h.nama:<26} {_tanda(m.faithfulness):<7} {_tanda(m.sitasi_grounded):<7} "
            f"{_tanda(m.numerik_benar):<8} {_tanda(m.json_valid):<6} {low_conf:<8} "
            f"{'LULUS' if h.lulus else 'GAGAL':<6}"
        )
        if not h.lulus:
            for label, ok, detail in [
                ("faithfulness", m.faithfulness, m.faithfulness_detail),
                ("grounded", m.sitasi_grounded, m.sitasi_grounded_detail),
                ("numerik", m.numerik_benar, m.numerik_benar_detail),
                ("json_valid", m.json_valid, m.json_valid_detail),
            ]:
                if not ok:
                    print(f"    - {label}: {detail}")

    print("-" * len(header))
    total = len(hasil)
    lulus_semua = sum(1 for h in hasil if h.lulus)
    print(f"Total: {lulus_semua}/{total} kasus lulus semua metrik.")

    for label, key in [
        ("faithfulness", "faithfulness"),
        ("sitasi_grounded", "sitasi_grounded"),
        ("numerik_benar", "numerik_benar"),
        ("json_valid", "json_valid"),
    ]:
        n = sum(1 for h in hasil if h.metrik is not None and getattr(h.metrik, key))
        print(f"  {label}: {n}/{total}")

    n_low_conf = sum(1 for h in hasil if h.low_confidence)
    if n_low_conf:
        print(f"  [!] {n_low_conf} kasus jatuh ke low_confidence (guardrail fallback).")

    return lulus_semua == total


def main() -> None:
    gold_set = muat_gold_set()
    retriever = MockRetriever()
    hasil = jalankan_gold_set(retriever, gold_set)
    semua_lulus = cetak_tabel(hasil)
    sys.exit(0 if semua_lulus else 1)


if __name__ == "__main__":
    main()
