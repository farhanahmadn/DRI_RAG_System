"""Pemeriksa eval deterministik — 4 metrik murah & andal (metrik struktural, bukan LLM-as-judge).

"Kejelasan Bahasa Indonesia" via LLM-as-judge DITUNDA (CLAUDE.md: mulai ramping, tambah pengaman
saat eval membuktikan perlu) — modul ini hanya properti yang bisa diverifikasi deterministik.
"""

import re
from dataclasses import dataclass

from app.reasoning.calculator import hitung_target_intensitas, pilih_target_utama_intensitas
from app.retrieval.base import Chunk
from app.schemas import L2Assessment, OutputL3, PoinOutput

_KATA_MITIGASI = ("kdb", "kdh", "rth", "resapan", "retensi", "mitigasi")


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


def _cari_poin(output: OutputL3, poin_id: str) -> PoinOutput:
    return next(p for p in output.poin if p.poin_id == poin_id)


def cek_faithfulness(output: OutputL3, expected: dict) -> tuple[bool, str]:
    """Kesetiaan ke ground truth: rekomendasi_sistem, status poin fokus, larangan kata arah salah,
    dan (bila relevan utk kasus) syarat/mitigasi/caveat/low_confidence WAJIB muncul sesuai ekspektasi.
    """
    if output.rekomendasi_sistem != expected["rekomendasi_sistem"]:
        return False, f"rekomendasi_sistem={output.rekomendasi_sistem!r}, diharapkan={expected['rekomendasi_sistem']!r}"

    poin = _cari_poin(output, expected["poin_id_fokus"])

    status_diharapkan = expected.get("status_diharapkan")
    if status_diharapkan is not None and poin.status != status_diharapkan:
        return False, f"poin[{poin.poin_id}].status={poin.status!r}, diharapkan={status_diharapkan!r}"

    teks = " ".join([poin.reasoning_pendek, poin.reasoning_panjang]).lower()

    for frasa in expected.get("arah_terlarang") or []:
        if re.search(rf"\b{re.escape(frasa.lower())}\b", teks):
            return False, f"frasa terlarang {frasa!r} muncul di reasoning poin[{poin.poin_id}]"

    if expected.get("harus_ada_syarat"):
        if "tidak diperlukan tindakan khusus" in poin.rekomendasi.saran.lower():
            return False, f"poin[{poin.poin_id}] diharapkan menyebut syarat konkret, saran masih generik (template aman)"

    if expected.get("harus_ada_mitigasi"):
        gabungan = f"{teks} {poin.rekomendasi.saran.lower()}"
        if not any(kata in gabungan for kata in _KATA_MITIGASI):
            return False, f"poin[{poin.poin_id}] diharapkan menyebut arah mitigasi, tidak ditemukan kata kunci {_KATA_MITIGASI}"

    if "low_confidence_diharapkan" in expected and poin.low_confidence != expected["low_confidence_diharapkan"]:
        return False, (
            f"poin[{poin.poin_id}].low_confidence={poin.low_confidence}, "
            f"diharapkan={expected['low_confidence_diharapkan']}"
        )

    if expected.get("caveat_diharapkan") and not poin.rekomendasi.disclaimer:
        return False, f"poin[{poin.poin_id}] diharapkan ada caveat di disclaimer, tapi kosong"

    return True, "rekomendasi_sistem, status, & narasi sesuai ekspektasi"


def cek_sitasi_grounded(output: OutputL3, chunks_diretrieve: list[Chunk], expected: dict) -> tuple[bool, str]:
    """Sitasi harus grounded: anchor-N (dari dasar_hukum asli via adapter) SELALU dianggap valid;
    selain itu harus ada di chunk yang benar-benar diretrieve (bukan karangan)."""
    poin = _cari_poin(output, expected["poin_id_fokus"])
    id_chunk_diretrieve = {c.id for c in chunks_diretrieve}

    for s in poin.sitasi:
        if s.citation_id.startswith("anchor-"):
            continue
        if s.citation_id not in id_chunk_diretrieve:
            return False, f"citation_id={s.citation_id!r} tidak ada di chunk yang diretrieve atau anchor (karangan)"

    citation_ids_diharapkan = expected.get("citation_ids_diharapkan") or []
    if citation_ids_diharapkan:
        id_disitasi = {s.citation_id for s in poin.sitasi}
        if not (id_disitasi & set(citation_ids_diharapkan)):
            return False, (
                f"tidak ada citation_ids_diharapkan={citation_ids_diharapkan} yang disitasi "
                f"(disitasi={sorted(id_disitasi)})"
            )

    return True, "semua sitasi grounded (anchor asli atau chunk RAG nyata)"


def cek_numerik(output: OutputL3, expected: dict, assessment: L2Assessment) -> tuple[bool, str]:
    """rekomendasi.target poin fokus harus persis sama dgn hasil calculator (dihitung ULANG di sini
    dari assessment, bukan dihardcode di expected — tetap benar kalau logika calculator berubah)."""
    poin = _cari_poin(output, expected["poin_id_fokus"])

    if poin.rekomendasi.tipe != "numerik":
        if poin.rekomendasi.target is not None:
            return False, f"tipe={poin.rekomendasi.tipe!r} bukan numerik tapi target={poin.rekomendasi.target!r} (seharusnya None)"
        return True, "non-numerik, target None sesuai"

    target_benar = pilih_target_utama_intensitas(
        hitung_target_intensitas(assessment.gate_hukum.tahapan.intensitas, assessment.lokasi.luas_lahan_m2)
    )
    if poin.rekomendasi.target != target_benar:
        return False, f"target={poin.rekomendasi.target!r}, seharusnya={target_benar!r} (dari calculator)"
    return True, "target sesuai calculator"


def cek_json_valid(output: OutputL3) -> tuple[bool, str]:
    """Roundtrip serialisasi JSON — output harus tetap valid setelah dump & parse ulang."""
    try:
        OutputL3.model_validate_json(output.model_dump_json())
    except Exception as exc:
        return False, f"roundtrip JSON gagal: {exc}"
    return True, "roundtrip JSON valid"


def evaluasi_kasus(
    output: OutputL3,
    assessment: L2Assessment,
    chunks_diretrieve: list[Chunk],
    expected: dict,
) -> HasilMetrik:
    faithfulness, faithfulness_detail = cek_faithfulness(output, expected)
    grounded, grounded_detail = cek_sitasi_grounded(output, chunks_diretrieve, expected)
    numerik, numerik_detail = cek_numerik(output, expected, assessment)
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
