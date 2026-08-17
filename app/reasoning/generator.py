"""Inti reasoning — generate_poin() untuk satu PoinKonteks (dari app/adapter.py).

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua metadata (poin_id, kategori, status, tipe rekomendasi, target) dan
metadata sitasi (dokumen/pasal/halaman) dirakit DI SINI dari poin/calculator/chunk yang benar-benar
diretrieve — bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

from app.reasoning import llm_client
from app.reasoning.calculator import pilih_target_mitigasi_dampak, pilih_target_utama_intensitas
from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import Chunk, RetrievalFilters, Retriever
from app.schemas import MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

# APP-2026-3468: saran deterministik utk poin aman/lolos — reasoning_pendek/panjang & sitasi TETAP
# dari LLM (poin aman WAJIB tetap dijelaskan KENAPA lolos, bukan reasoning generik tanpa sitasi),
# hanya bagian "tidak ada tindakan lanjut" ini yang aman ditemplate (tak ada apa pun utk direkomendasikan).
_SARAN_AMAN = "Tidak diperlukan tindakan khusus; poin ini telah memenuhi ketentuan."

# APP-2026-003: "Tidak Dinilai" (intensitas tanpa data krn gate berhenti di ITBX, atau
# impact_assessment.dinilai=False) BUKAN "memenuhi ketentuan" — poin ini memang belum pernah
# dievaluasi sama sekali. Pakai `_SARAN_AMAN` di sini akan kontradiktif dgn reasoning LLM yang
# (benar) menjelaskan poin tidak dinilai (pola sama dgn bug kesimpulan APP-2026-8376 yg baru
# diperbaiki: saran & narasi harus SATU sumber kebenaran, bukan dua kalimat beda makna).
_SARAN_TIDAK_DINILAI = (
    "Tidak dievaluasi karena proses pemeriksaan berhenti pada tahapan sebelumnya; "
    "tidak ada rekomendasi untuk poin ini."
)

_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_pendek": {"type": "string"},
        "reasoning_panjang": {"type": "string"},
        "sitasi": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_id": {"type": "string"},
                    "kutipan": {"type": "string"},
                },
                "required": ["citation_id", "kutipan"],
                "additionalProperties": False,
            },
        },
        "saran": {"type": "string"},
        "disclaimer": {"type": ["string", "null"]},
    },
    "required": ["reasoning_pendek", "reasoning_panjang", "sitasi", "saran", "disclaimer"],
    "additionalProperties": False,
}


# Kata kunci PENDEK utk jalur search fallback — selaras dgn `_EXPANSION` di
# app/retrieval/retriever.py (RAG asli teman: exact-match lowercase key -> istilah regulasi,
# TIDAK fuzzy/substring). `poin.kategori` sekarang string deskriptif panjang ("Klasifikasi
# Kegiatan (ITBX)" dst) yang TIDAK match kunci `_EXPANSION` manapun persis — pakai kata kunci
# pendek di sini supaya RetrieverAsli tetap dapat manfaat ekspansi query.
_QUERY_FALLBACK_PER_POIN = {
    "itbx": "kegiatan",
    "intensitas": "kdb",
    # Fix #5: entri "dampak tata guna lahan" ditambahkan ke _EXPANSION retriever.py (limpasan/
    # runoff/sumur resapan/kolam retensi/zero delta Q/RTH) — kata kunci pendek di sini persis
    # match key itu, jangan pakai poin.kategori panjang ("Dampak Tata Guna Lahan" apa adanya
    # kebetulan lowercase-match juga, tapi eksplisit lebih tahan kalau kategori berubah nanti).
    "dampak": "dampak tata guna lahan",
}

# Nama zona INDUK (persis spt `assessment.lokasi.rdtr_zone` dari back-end) -> kode prefix, sesuai
# Pasal 17 (Zona Lindung) & Pasal 23 (Zona Budi Daya), "RDTR Kawasan Sleman Tengah 2023-2043.md"
# (data/parsed/v1/) — diverifikasi thd `data/raw/*.pdf` langsung, BUKAN ditebak. Dipakai
# `ambil_chunks_pendukung` sbg `RetrievalFilters.zona_prefix` saat search() fallback, supaya
# Lampiran V.B/VI yang dikutip TIDAK lintas KELUARGA zona (bug nyata: APP-2026-6191, pemohon
# "Zona Perumahan" tapi Lampiran VI Zona Perkantoran "KT" ikut terkutip krn search() sebelumnya
# tanpa filter zona sama sekali). CATATAN: back-end cuma kasih nama zona INDUK, bukan kode
# sub-zona presisi (mis. "Zona Perumahan" tanpa tahu R-2/R-3/R-4 yang mana) — prefix ini cegah
# kontaminasi ANTAR-keluarga, TIDAK menjamin sub-zona presisi di DALAM satu keluarga.
_ZONA_KODE_PREFIX = {
    "zona badan air": "BA",
    "zona perlindungan setempat": "PS",
    "zona ruang terbuka hijau": "RTH",
    "zona konservasi": "KS",
    "zona cagar budaya": "CB",
    "zona badan jalan": "BJ",
    "zona pertanian": "P",
    "zona pembangkitan tenaga listrik": "PTL",
    "zona kawasan peruntukan industri": "KPI",
    "zona pariwisata": "W",
    "zona perumahan": "R",
    "zona sarana pelayanan umum": "SPU",
    "zona ruang terbuka non hijau": "RTNH",
    "zona campuran": "C",
    "zona perdagangan dan jasa": "K",
    "zona perkantoran": "KT",
    "zona peruntukan lainnya": "PL",
    "zona pengelolaan persampahan": "PP",
    "zona transportasi": "TR",
    "zona pertahanan dan keamanan": "HK",
}


def _zona_prefix_dari_nama(nama_zona: str | None) -> str | None:
    """"Zona Perumahan" -> "R" dst (lihat `_ZONA_KODE_PREFIX`). None kalau nama tak dikenal —
    JANGAN menebak, biarkan filter kosong (search tanpa filter zona, seperti perilaku lama) drpd
    salah filter berdasar tebakan."""
    if not nama_zona:
        return None
    return _ZONA_KODE_PREFIX.get(nama_zona.strip().lower())


def ambil_chunks_pendukung(
    poin: PoinKonteks,
    retriever: Retriever,
    top_k_dukungan: int = 3,
) -> list[Chunk]:
    """Retrieval chunk pendukung utk satu poin: dasar_hukum dulu, fallback search kata kunci pendek."""
    referensi = [f"{d.dokumen} {d.pasal}" for d in poin.dasar_hukum if d.pasal]
    chunks = retriever.get_by_reference(referensi) if referensi else []
    # Investigasi ITBX APP-2026-6191 (live thd DB nyata): `dasar_hukum` back-end kadang berupa label
    # non-pasal ("Matriks ITBX", tanpa nomor) + dokumen generik ("RDTR Sleman") — get_by_reference
    # (tidak bisa parse nomor pasal, jatuh ke fallback dokumen-level) bisa mengembalikan RATUSAN chunk
    # TAK TERBATAS (terbukti live: seluruh korpus, bukan cuma 6 chunk kecil di MockRetriever). Prompt
    # yang membanjiri LLM bikin ia gagal memilih sitasi sama sekali. Batasi ke top_k_dukungan di sini
    # (bukan di retriever.py — bukan file saya) — anchor (dasar_hukum asli) tetap utuh dikirim terpisah
    # ke prompt (lihat build_user_prompt), jadi pembatasan ini TIDAK mengurangi sitasi yang faithful.
    if len(chunks) > top_k_dukungan:
        chunks = chunks[:top_k_dukungan]
    if not chunks:
        query_fallback = _QUERY_FALLBACK_PER_POIN.get(poin.poin_id, poin.kategori)
        # APP-2026-8090: filter EXACT ke sub-zona presisi (mis. "P-1") kalau BE mengirim &
        # yakin (poin.zona_subzone) — jauh lebih presisi drpd filter KELUARGA (zona_prefix, cuma
        # bisa saring "P" tanpa beda P-1/P-2). Fallback ke zona_prefix (APP-2026-6191, cegah
        # Lampiran V.B/VI zona lain ikut terkutip) kalau sub-zona kosong/BE tak yakin — TIDAK
        # PERNAH menebak sub-zona sendiri di sini.
        if poin.zona_subzone:
            filters = RetrievalFilters(zona=poin.zona_subzone)
        else:
            filters = RetrievalFilters(zona_prefix=_zona_prefix_dari_nama(poin.zona))
        chunks = retriever.search(query_fallback, filters, top_k=top_k_dukungan)
    return chunks


def apakah_aman(poin: PoinKonteks) -> bool:
    """Poin jelas aman/lolos -> boleh template tanpa LLM (hemat kuota). REUSE fakta yang sudah
    dihitung adapter/calculator, tidak menurunkan ulang di sini.
    """
    if poin.poin_id == "itbx":
        return (
            poin.status == "I"
            and bool(poin.fakta.get("lolos"))
            and not poin.fakta.get("fallback_data_kosong", False)
        )
    if poin.poin_id == "intensitas":
        return not poin.fakta.get("target")
    if poin.poin_id == "dampak":
        mitigasi = poin.fakta.get("mitigasi") or {}
        return not mitigasi.get("perlu_mitigasi")
    return False


def generate_poin(
    poin: PoinKonteks,
    retriever: Retriever,
    meta: MetaL2 | None = None,
    *,
    top_k_dukungan: int = 3,
    catatan_perbaikan: str | None = None,
    temperature: float = 0.0,
) -> PoinOutput:
    """Hasilkan PoinOutput untuk satu poin — SELALU lewat retrieval + LLM, termasuk poin aman/lolos
    (APP-2026-3468: poin aman WAJIB tetap dijelaskan KENAPA lolos + sitasi pasal, bukan reasoning
    generik tanpa sitasi seperti sebelumnya). Poin aman hanya dapat template pada
    `rekomendasi.saran`/`target` (lihat apakah_aman di bawah) — reasoning & sitasi TETAP dari LLM.
    """
    chunks = ambil_chunks_pendukung(poin, retriever, top_k_dukungan)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    anchor_by_id = {f"anchor-{i}": d for i, d in enumerate(poin.dasar_hukum)}

    prompt = build_user_prompt(poin, chunks, meta, catatan_perbaikan)

    llm_out = llm_client.generate(
        prompt,
        _LLM_RESPONSE_SCHEMA,
        schema_name="poin_reasoning",
        system=SYSTEM_PROMPT,
        temperature=temperature,
    )

    sitasi: list[SitasiOutput] = []
    for item in llm_out.get("sitasi", []):
        citation_id = item.get("citation_id")
        kutipan = item.get("kutipan", "")

        anchor = anchor_by_id.get(citation_id)
        if anchor is not None:
            sitasi.append(
                SitasiOutput(
                    citation_id=citation_id,
                    dokumen=anchor.dokumen,
                    pasal=anchor.pasal or "",
                    halaman=0,
                    kutipan=kutipan or anchor.kutipan,
                    terverifikasi=True,
                )
            )
            continue

        chunk = chunk_by_id.get(citation_id)
        if chunk is None:
            continue  # citation_id tak dikenal (halusinasi) -> dibuang
        sitasi.append(
            SitasiOutput(
                citation_id=chunk.id,
                dokumen=chunk.dokumen,
                pasal=chunk.pasal or "",
                halaman=chunk.halaman or 0,
                # SELALU chunk.teks apa adanya (BUKAN `kutipan or chunk.teks`) — kutipan dari LLM di
                # sini cuma dipakai utk MEMILIH chunk mana yang relevan, isi teksnya sendiri wajib
                # verbatim dari hasil retrieval, LLM tak boleh menulis ulang/mengarang kutipan pasal.
                kutipan=chunk.teks,
                terverifikasi=True,
            )
        )

    target: float | str | None = None
    if poin.tipe_rekomendasi == "numerik":
        target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})
    elif poin.tipe_rekomendasi == "numerik-mitigasi":
        target = pilih_target_mitigasi_dampak(poin.fakta.get("target_mitigasi") or {})

    saran = llm_out["saran"]
    if poin.status == "Tidak Dinilai":
        # Belum pernah dievaluasi (bukan "sudah memenuhi ketentuan") — lihat _SARAN_TIDAK_DINILAI.
        saran = _SARAN_TIDAK_DINILAI
        target = None
    elif apakah_aman(poin):
        # Tidak ada pelanggaran utk ditindaklanjuti -> saran & target deterministik, TAPI
        # reasoning_pendek/panjang & sitasi di atas tetap murni dari LLM (lihat docstring).
        saran = _SARAN_AMAN
        target = None

    return PoinOutput(
        poin_id=poin.poin_id,
        kategori=poin.kategori,
        status=poin.status,
        reasoning_pendek=llm_out["reasoning_pendek"],
        reasoning_panjang=llm_out["reasoning_panjang"],
        sitasi=sitasi,
        rekomendasi=RekomendasiOutput(
            tipe=poin.tipe_rekomendasi,
            target=target,
            saran=saran,
            disclaimer=llm_out.get("disclaimer"),
        ),
        low_confidence=False,
    )
