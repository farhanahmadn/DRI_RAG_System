"""Pemeriksa eval deterministik — 4 metrik murah & andal (metrik struktural, bukan LLM-as-judge).

"Kejelasan Bahasa Indonesia" via LLM-as-judge DITUNDA (CLAUDE.md: mulai ramping, tambah pengaman
saat eval membuktikan perlu) — modul ini hanya properti yang bisa diverifikasi deterministik.
"""

import re
from dataclasses import dataclass

from app.reasoning.calculator import hitung_target_rekomendasi, pilih_target_utama
from app.retrieval.base import Chunk
from app.schemas import IndikatorJejak, OutputPreCheck, PoinOutput


@dataclass
class HasilMetrik:
    faithfulness: bool
    faithfulness_detail: str
    sitasi_grounded: bool
    sitasi_grounded_detail: str
    numerik_benar: bool
    numerik_benar_detail: str
    json_valid: bool
    json_valid_detail: str

    @property
    def lulus_semua(self) -> bool:
        return self.faithfulness and self.sitasi_grounded and self.numerik_benar and self.json_valid


def cek_faithfulness(poin: PoinOutput, expected: dict) -> tuple[bool, str]:
    """Status output harus sama dengan ekspektasi; kata arah terlarang tak boleh muncul di narasi."""
    if poin.status != expected["status"]:
        return False, f"status={poin.status!r}, diharapkan={expected['status']!r}"

    arah_terlarang = expected.get("arah_terlarang")
    if arah_terlarang:
        # Hanya reasoning (klaim status), BUKAN saran — saran boleh pakai kata kerja arah
        # berlawanan yang justru benar (mis. "kurangi luas bangunan" utk kasus melebihi).
        # Word-boundary: "kurang" tidak boleh cocok di dalam "kurangi"/"dikurangi"/"pengurangan".
        teks = " ".join([poin.reasoning_pendek, poin.reasoning_panjang]).lower()
        if re.search(rf"\b{re.escape(arah_terlarang.lower())}\b", teks):
            return False, f"kata terlarang {arah_terlarang!r} muncul di reasoning (arah salah)"

    return True, "status & arah narasi sesuai"


def cek_sitasi_grounded(
    poin: PoinOutput, chunks_diretrieve: list[Chunk], expected: dict
) -> tuple[bool, str]:
    """Sitasi harus grounded ke chunk yang benar-benar diretrieve; RAG kosong -> sitasi kosong."""
    id_chunk_diretrieve = {c.id for c in chunks_diretrieve}

    if not chunks_diretrieve:
        if poin.sitasi:
            return False, "chunks_diretrieve kosong tapi sitasi tidak kosong (RAG kosong dilanggar)"
        return True, "RAG kosong, sitasi kosong sesuai ekspektasi"

    for s in poin.sitasi:
        if s.citation_id not in id_chunk_diretrieve:
            return False, f"citation_id={s.citation_id!r} tidak ada di chunk yang diretrieve (karangan)"

    citation_ids_diharapkan = expected.get("citation_ids_diharapkan") or []
    if citation_ids_diharapkan:
        id_disitasi = {s.citation_id for s in poin.sitasi}
        if not (id_disitasi & set(citation_ids_diharapkan)):
            return False, (
                f"tidak ada citation_ids_diharapkan={citation_ids_diharapkan} yang disitasi "
                f"(disitasi={sorted(id_disitasi)})"
            )

    return True, "semua sitasi grounded, minimal satu sumber relevan disitasi"


def cek_numerik(poin: PoinOutput, indikator: IndikatorJejak) -> tuple[bool, str]:
    """rekomendasi.target harus persis sama dengan hasil calculator (berlaku juga utk non-numerik: None==None)."""
    target_benar = pilih_target_utama(hitung_target_rekomendasi(indikator))
    if poin.rekomendasi.target != target_benar:
        return False, f"target={poin.rekomendasi.target!r}, seharusnya={target_benar!r} (dari calculator)"
    return True, "target sesuai calculator"


def cek_json_valid(output: OutputPreCheck) -> tuple[bool, str]:
    """Roundtrip serialisasi JSON — output harus tetap valid setelah dump & parse ulang."""
    try:
        OutputPreCheck.model_validate_json(output.model_dump_json())
    except Exception as exc:
        return False, f"roundtrip JSON gagal: {exc}"
    return True, "roundtrip JSON valid"


def evaluasi_kasus(
    poin: PoinOutput,
    indikator: IndikatorJejak,
    chunks_diretrieve: list[Chunk],
    output: OutputPreCheck,
    expected: dict,
) -> HasilMetrik:
    faithfulness, faithfulness_detail = cek_faithfulness(poin, expected)
    grounded, grounded_detail = cek_sitasi_grounded(poin, chunks_diretrieve, expected)
    numerik, numerik_detail = cek_numerik(poin, indikator)
    json_valid, json_valid_detail = cek_json_valid(output)

    return HasilMetrik(
        faithfulness=faithfulness,
        faithfulness_detail=faithfulness_detail,
        sitasi_grounded=grounded,
        sitasi_grounded_detail=grounded_detail,
        numerik_benar=numerik,
        numerik_benar_detail=numerik_detail,
        json_valid=json_valid,
        json_valid_detail=json_valid_detail,
    )
