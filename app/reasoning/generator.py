"""Inti reasoning — generate_poin() untuk satu PoinKonteks (dari app/adapter.py).

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua metadata (poin_id, kategori, status, tipe rekomendasi, target) dan
metadata sitasi (dokumen/pasal/halaman) dirakit DI SINI dari poin/calculator/chunk yang benar-benar
diretrieve — bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

import logging

from app.reasoning import llm_client
from app.reasoning.calculator import (
    bangun_langkah_konkret_dampak,
    bangun_langkah_konkret_intensitas,
    pilih_target_mitigasi_dampak,
    pilih_target_utama_intensitas,
)
from app.reasoning.prompts import (
    CAVEAT_SUBZONA_TAK_TERKONFIRMASI,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.retrieval.base import Chunk, RetrievalFilters, Retriever
from app.schemas import MetaL2, PoinKonteks, PoinOutput, RekomendasiOutput, SitasiOutput

logger = logging.getLogger(__name__)

# APP-2026-3468: saran deterministik utk poin aman/lolos — reasoning_pendek/panjang & sitasi TETAP
# dari LLM (poin aman WAJIB tetap dijelaskan KENAPA lolos, bukan reasoning generik tanpa sitasi),
# hanya bagian "tidak ada tindakan lanjut" ini yang aman ditemplate (tak ada apa pun utk direkomendasikan).
_SARAN_AMAN = "Tidak diperlukan tindakan khusus. Poin ini telah memenuhi ketentuan."

# APP-2026-003: "Tidak Dinilai" (intensitas tanpa data krn gate berhenti di ITBX, atau
# impact_assessment.dinilai=False) BUKAN "memenuhi ketentuan" — poin ini memang belum pernah
# dievaluasi sama sekali. Pakai `_SARAN_AMAN` di sini akan kontradiktif dgn reasoning LLM yang
# (benar) menjelaskan poin tidak dinilai (pola sama dgn bug kesimpulan APP-2026-8376 yg baru
# diperbaiki: saran & narasi harus SATU sumber kebenaran, bukan dua kalimat beda makna).
_SARAN_TIDAK_DINILAI = (
    "Tidak dievaluasi karena proses pemeriksaan berhenti pada tahapan sebelumnya; "
    "tidak ada rekomendasi untuk poin ini."
)


def _saran_tidak_dinilai(poin: PoinKonteks) -> str:
    """APP-2026-2428: status "Tidak Dinilai" sekarang punya >1 penyebab beda makna —
    `_SARAN_TIDAK_DINILAI` generik di atas (gate berhenti di tahap sebelumnya) HANYA benar utk
    kasus lama (mis. intensitas tanpa data krn ITBX gagal). Kasus baru: impact_assessment.dinilai=
    False krn back-end SENGAJA menolak menghitung (mis. poligon bersinggungan >1 persil,
    `poin.fakta['limitations']` terisi) — pesan generik "gate berhenti" jadi SALAH/menyesatkan di
    sini (ITBX & intensitas bisa saja lolos normal, cuma dampak spesifik yg gagal, sebab spasial
    bukan prosedural). Echo `limitations` VERBATIM (FAITHFUL, ground truth back-end, tak dikarang)
    kalau tersedia — generalisasi otomatis ke alasan apa pun yg BE kirim nanti, bukan di-hardcode
    ke "multi persil" doang. Fallback ke pesan generik lama kalau `limitations` kosong (kasus lama
    tetap jalan spt sebelumnya, backward-compatible)."""
    limitations = (poin.fakta or {}).get("limitations")
    if limitations:
        return (
            f"Asesmen dampak terhadap lingkungan (hidrologi) tidak dapat dilakukan: {limitations} "
            "Pemohon perlu menindaklanjuti catatan tersebut sebelum penilaian dampak dapat diproses."
        )
    return _SARAN_TIDAK_DINILAI


def _saran_mitigasi_dampak(poin: PoinKonteks) -> str | None:
    """APP-2026-8025/-5067: BE kini (kadang) kirim `impact_assessment.rekomendasi_mitigasi.saran` —
    narasi mitigasi SUDAH DIHITUNG PENUH (angka fisik m²/KDB%/KDH% konkret + dimensi sumur/kolam
    resapan, bukan cuma ambang indeks abstrak). Echo VERBATIM (FAITHFUL, sama prinsip dgn
    `_saran_tidak_dinilai` di atas) alih-alih biarkan LLM menulis ulang/improvisasi dari arah
    kualitatif generik (`sarankan_arah_mitigasi_dampak`, calculator.py) — BE lebih otoritatif krn
    rumus C/index & konversi ke luasan fisik ada di sisi mereka, bukan kita.

    Diteruskan lewat `poin.fakta['target_mitigasi']['saran_be']` (dirakit
    `calculator.py::hitung_target_mitigasi_dampak`), BUKAN baca `rekomendasi_mitigasi` mentah
    langsung di sini — satu jalur perakitan fakta dampak, satu sumber kebenaran. None kalau BE tak
    kirim (fixture lama / kategori dampak tak butuh mitigasi) -> caller pakai saran LLM apa adanya
    (perilaku lama, backward-compatible)."""
    if poin.poin_id != "dampak":
        return None
    target_mitigasi = (poin.fakta or {}).get("target_mitigasi") or {}
    return target_mitigasi.get("saran_be") or None


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

# APP-2026-8090 (live, verifikasi manual thd DB): query "kdb" polos kalah oleh pasal definisional
# umum (Pasal 1/44/61 — skor lebih tinggi drpd tabel ambang Lampiran VI spesifik-zona, walau
# filter zona SUDAH benar) — tabel ambang cuma naik ke rank #1 dgn query lebih spesifik ini.
#
# Dulu query ini DIBATASI hanya untuk kasus `zona_subzone` presisi, karena dgn filter KELUARGA
# saja beberapa sub-zona (mis. R-2/R-3/R-4) skornya berdekatan & tak bisa dibedakan — dinilai
# "lebih baik aman tapi kurang presisi". Pengukuran (eval/laporan_rag.html bagian 6, 21 keluarga
# zona, filter & label IDENTIK, hanya teks query berbeda) menunjukkan ongkos "aman" itu jauh
# lebih besar dari dugaan: dgn "kdb" polos, nDCG@3 0.157 & Recall@3 30.2%, dan tabel ambang yang
# benar TIDAK PERNAH sampai peringkat 1 (0/21). Artinya pada cabang ini sistem sebagian besar
# bukan "aman", melainkan menjawab TANPA tabel ambang sama sekali. Dgn query tajam:
# nDCG@3 0.856, Recall@3 90.5%, peringkat 1 pada 16/21 (p=5.3e-05).
#
# Cabang ini menguasai 96.4% permohonan (425 dari 441) pada desain 3-poin yang berjalan
# sekarang — dihitung HANYA atas era itu. Angka 73%-74% yang sempat tertulis di sini keliru:
# ia mencampur 309 permohonan era 8-indikator lama (2026-07-20..23, seluruhnya tanpa zona
# induk) yang tak lagi mewakili produksi.
#
# Kekhawatiran aslinya TIDAK terbantah oleh angka itu — label eval menganggap seluruh tabel satu
# keluarga sah, jadi ia tak bisa memutuskan apakah sub-zona yang DIKUTIP tepat. Karena itu
# pembukaan gating ini WAJIB berpasangan dgn CAVEAT_SUBZONA_TAK_TERKONFIRMASI (dipasang di dua
# tempat: prompt, supaya narasi tak mengklaim lebih dari yang diketahui; dan catatan_global di
# assemble.py, supaya pembaca tetap melihatnya walau LLM gagal). Jangan hapus salah satunya
# tanpa menutup kembali gating di bawah.
_QUERY_INTENSITAS_TAJAM = "ambang KDB KLB KDH maksimal minimal"

# Poin yang memakai kandidat dense-saja (leg lexical dimatikan) sebelum rerank.
#
# Diukur pada titik operasi k=3 SETELAH query tajam dipakai di kedua cabang
# (eval/laporan_rag.html). Pada cabang mayoritas — filter keluarga, 96.4% permohonan desain
# 3-poin — dense+rerank unggul nDCG@3 0.978 vs 0.856 (p=0.016, efek +1.00) dan
# menempatkan tabel ambang yang benar di peringkat 1 pada 21/21 keluarga zona, lawan 16/21.
# Sebabnya terbaca: leg lexical praktis mati untuk query intensitas (Hit@10 hanya 3%, karena
# teks tabel Lampiran VI miskin lexeme yang cocok), sehingga RRF mengencerkan peringkat
# dense dgn daftar yang tak membawa informasi.
#
# `dampak` SENGAJA tidak masuk: di sana arahnya justru sebaliknya (0.786 vs 0.719) tapi n=5
# dan p=0.625 — tak terbaca. Mempertahankan perilaku lama di tempat yang belum terukur
# adalah pilihan sadar, bukan kelupaan. `itbx` juga tidak: ia lewat get_by_reference dan
# tak menyentuh fusi sama sekali.
_TANPA_LEXICAL_PER_POIN = frozenset({"intensitas"})

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


def _keluarga_zona(kode: str | None) -> str | None:
    """Kode sub-zona -> keluarga zonanya: "P-1"/"P-1 LP2B" -> "P", "RTH-2" -> "RTH", "CA" -> "CA".
    Sejalan dgn `_ZONA_KODE_PREFIX` di atas (nilai dict itu = keluarga zona yang sama)."""
    if not kode:
        return None
    return kode.strip().upper().split("-")[0].split()[0] or None


def _skor_kecocokan_zona(chunk_zona: str | None, subzona: str | None, prefix: str | None) -> int | None:
    """Seberapa cocok satu chunk dgn zona pemohon. 3 = sub-zona persis, 2 = satu keluarga zona,
    1 = chunk tak terikat zona apa pun (prosa umum spt Pasal 43 — sah dikutip zona mana pun).
    None = chunk milik KELUARGA ZONA LAIN, harus DIBUANG (inilah sumber salah-sitasi)."""
    if not chunk_zona:
        return 1
    kode = chunk_zona.strip().upper()
    if subzona and kode == subzona.strip().upper():
        return 3
    if prefix and _keluarga_zona(kode) == prefix.strip().upper():
        return 2
    return None


def _pilih_chunks_referensi(chunks: list[Chunk], poin: PoinKonteks, top_k: int) -> list[Chunk]:
    """Pilih chunk hasil `get_by_reference` menurut KECOCOKAN ZONA pemohon, bukan urutan DB.

    Bug nyata (APP-2026-2428, terverifikasi thd DB & logs/precheck.jsonl 2026-09-07): rujukan
    generik back-end ("RDTR Sleman" + "Matriks ITBX") membuat `get_by_reference` mengembalikan
    ribuan chunk seluruh korpus, lalu dipotong `[:top_k]` menurut urutan DB apa adanya. Untuk
    pemohon Zona Pertanian (P-1), 3 chunk teratas urutan DB adalah Lampiran V.B zona CA/R-2/R-3 —
    sehingga permohonan pertanian dijelaskan memakai Lampiran **Cagar Alam**, lolos guardrail dgn
    `terverifikasi=True` (guardrail cuma mengecek citation_id ADA di daftar kandidat, bukan bahwa
    kandidatnya relevan). Tabel zona pemohon sendiri (`vb-p-1`) ada jauh di urutan 47 — tak pernah
    terkirim. Urutan DB memang bukan sinyal relevansi apa pun; di sinilah relevansi ditentukan.

    Chunk keluarga zona LAIN dibuang, bukan cuma diturunkan peringkatnya — kalau semua kandidat
    ternyata salah zona, lebih baik daftar kosong (pemanggil jatuh ke `search()` yang SUDAH
    berfilter zona) drpd menyodorkan Lampiran zona lain ke LLM. Zona pemohon tak dikenal -> JANGAN
    menebak, pertahankan perilaku lama (potong menurut urutan apa adanya).
    """
    subzona = poin.zona_subzone
    prefix = _zona_prefix_dari_nama(poin.zona)
    if not subzona and not prefix:
        return chunks[:top_k]

    berskor: list[tuple[int, int, Chunk]] = []
    for urutan_asli, chunk in enumerate(chunks):
        skor = _skor_kecocokan_zona(chunk.zona, subzona, prefix)
        if skor is None:
            continue
        # -skor supaya skor tertinggi dulu; `urutan_asli` menjaga urutan semula utk skor yang sama.
        berskor.append((-skor, urutan_asli, chunk))
    berskor.sort()
    return [chunk for _, _, chunk in berskor[:top_k]]


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
    # APP-2026-2428: pemotongan itu TIDAK BOLEH menurut urutan DB — pilih menurut kecocokan zona
    # pemohon dulu (lihat _pilih_chunks_referensi utk bukti & alasan lengkapnya).
    chunks = _pilih_chunks_referensi(chunks, poin, top_k_dukungan)
    if not chunks:
        query_fallback = _QUERY_FALLBACK_PER_POIN.get(poin.poin_id, poin.kategori)
        if poin.poin_id == "intensitas":
            # Query tajam dipakai apa pun tingkat filternya — lihat pengukuran di
            # _QUERY_INTENSITAS_TAJAM. Ketidakpastian sub-zona ditangani lewat caveat, bukan
            # dgn melemahkan query sampai tabel ambangnya tidak ketemu sama sekali.
            query_fallback = _QUERY_INTENSITAS_TAJAM
        # APP-2026-8090: filter EXACT ke sub-zona presisi (mis. "P-1") kalau BE mengirim &
        # yakin (poin.zona_subzone) — jauh lebih presisi drpd filter KELUARGA (zona_prefix, cuma
        # bisa saring "P" tanpa beda P-1/P-2). Fallback ke zona_prefix (APP-2026-6191, cegah
        # Lampiran V.B/VI zona lain ikut terkutip) kalau sub-zona kosong/BE tak yakin — TIDAK
        # PERNAH menebak sub-zona sendiri di sini.
        if poin.zona_subzone:
            filters = RetrievalFilters(zona=poin.zona_subzone)
        else:
            filters = RetrievalFilters(zona_prefix=_zona_prefix_dari_nama(poin.zona))
        chunks = retriever.search(query_fallback, filters, top_k=top_k_dukungan,
                                  tanpa_lexical=poin.poin_id in _TANPA_LEXICAL_PER_POIN)
    return chunks


# Batas ekspansi small-to-big. Diukur dari korpus nyata (2026-09-08): 137 pasal induk, median 1.449
# char, p90 3.214, tapi ada ekor ekstrem sampai 30.748 — tanpa batas, satu pasal saja bisa memicu
# ulang 413/429 "Request too large" (APP-2026-9461). Ambil induk hanya utk chunk paling relevan.
_MAKS_CHUNK_DGN_INDUK = 2
_MAKS_CHAR_INDUK = 2500

# Penanda bahwa sebuah ayat MERUJUK bagian lain sehingga tak bermakna sendirian. Substring biasa,
# bukan regex — cukup, dan tak ada escaping yang bisa salah. Terukur di korpus nyata: 65% chunk ayat
# ter-embed memuat salah satunya.
_PENANDA_RUJUKAN_SILANG = ("sebagaimana dimaksud", "ayat (", "huruf ")


def _butuh_konteks_induk(teks: str) -> bool:
    """Hanya ayat yang MERUJUK bagian lain yang perlu induknya.

    Tanpa saringan ini, definisi di Pasal 1 (mis. "Koefisien Dasar Bangunan ... adalah angka
    persentase...") ikut ditempeli induknya — padahal definisi sudah mandiri, DAN induknya adalah
    pasal definisi terpanjang di korpus (30.748 char). Terbukti live: chunk `p1-a117` menarik 2.511
    char berisi 126 definisi tak terkait ke prompt. Menyaring di sini menghemat prompt sekaligus
    mengurangi derau — bukan cuma soal ukuran.
    """
    rendah = teks.lower()
    return any(penanda in rendah for penanda in _PENANDA_RUJUKAN_SILANG)


def caveat_subzona(poin: PoinKonteks) -> list[str]:
    """Caveat sub-zona untuk poin intensitas yang disitasi dari tabel tingkat KELUARGA zona.

    Hanya untuk `intensitas`: poin inilah yang ambangnya (KDB/KLB/KDH) berbeda antar sub-zona
    dalam satu keluarga. ITBX memakai Lampiran V.B yang juga per-sub-zona, tapi jalurnya
    `get_by_reference` dgn anchor dari back-end, bukan query tajam yang baru dibuka di sini —
    menambahkan caveat ke sana akan memperingatkan hal yang tidak berubah.
    """
    if poin.poin_id != "intensitas" or poin.zona_subzone:
        return []
    if poin.status == "Tidak Dinilai":
        # BE tak mengirim penilaian intensitas -> tak ada ambang yang dikutip sama sekali.
        # Lihat assemble._intensitas_dinilai; syaratnya harus sama di kedua lapis, kalau
        # tidak narasi memperingatkan hal yang tak disebut catatan_global (atau sebaliknya).
        return []
    return [CAVEAT_SUBZONA_TAK_TERKONFIRMASI]


def ambil_konteks_induk(chunks: list[Chunk], retriever: Retriever) -> dict[str, str]:
    """Ambil teks pasal INDUK untuk chunk ayat — melengkapi separuh desain small-to-big yang selama
    ini terbangun tapi tak pernah dipakai.

    Bukti kenapa perlu (korpus nyata, 2026-09-08): 62% dari 895 chunk ayat yang di-embed memuat
    "sebagaimana dimaksud", yaitu merujuk ayat lain yang TIDAK ikut terkirim ke LLM. Contohnya
    "(3) Lokasi sebagaimana dimaksud pada ayat (1) huruf b terdapat di blok dalam SWP." — verbatim
    benar, tapi tak bermakna sendirian. Chunker sudah menyimpan pasal induk khusus untuk ini
    (`to_embed=False`, lihat app/ingest/chunk.py) dan `get_parent` sudah ada di seluruh lapis
    retrieval, tapi tak pernah dipanggil dari app/reasoning/ — jadi konteksnya menganggur di DB.

    Return: {chunk_id: teks_induk}. Induk yang SAMA hanya dilampirkan sekali (beberapa ayat dari
    pasal yang sama tak perlu mengulang teks induk yang identik). Kegagalan retrieval di sini
    TIDAK PERNAH menggagalkan generasi — konteks ini penyempurna, bukan syarat.
    """
    hasil: dict[str, str] = {}
    induk_terpakai: set[str] = set()
    for chunk in chunks[:_MAKS_CHUNK_DGN_INDUK]:
        if chunk.level != "ayat" or not chunk.parent_id or chunk.parent_id in induk_terpakai:
            continue
        if not _butuh_konteks_induk(chunk.teks):
            continue
        try:
            induk = retriever.get_parent(chunk.id)
        except Exception:
            logger.warning("get_parent gagal utk chunk %r — lanjut tanpa konteks induk.", chunk.id, exc_info=True)
            continue
        if induk is None or not induk.teks:
            continue
        induk_terpakai.add(chunk.parent_id)
        teks = induk.teks
        if len(teks) > _MAKS_CHAR_INDUK:
            teks = teks[:_MAKS_CHAR_INDUK].rstrip() + " […dipotong]"
        hasil[chunk.id] = teks
    return hasil


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
    konteks_induk = ambil_konteks_induk(chunks, retriever)

    prompt = build_user_prompt(poin, chunks, meta, catatan_perbaikan, konteks_induk=konteks_induk,
                               catatan_tambahan=caveat_subzona(poin))

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
    # langkah_konkret (additive, 2026-08-18): SEMUA parameter yang melanggar sekaligus, dirakit
    # deterministik dari calculator.py — beda dari `target` di atas (cuma 1 angka representatif,
    # lihat pilih_target_utama_intensitas). Field tambahan di RekomendasiOutput, tak mengubah
    # `target`/`saran` yang sudah ada.
    langkah_konkret: list[dict] = []
    if poin.tipe_rekomendasi == "numerik":
        target = pilih_target_utama_intensitas(poin.fakta.get("target") or {})
        langkah_konkret = bangun_langkah_konkret_intensitas(
            poin.fakta.get("parameter") or {}, poin.fakta.get("target") or {}
        )
    elif poin.tipe_rekomendasi == "numerik-mitigasi":
        target = pilih_target_mitigasi_dampak(poin.fakta.get("target_mitigasi") or {})
        langkah_konkret = bangun_langkah_konkret_dampak(poin.fakta.get("target_mitigasi") or {})

    saran = llm_out["saran"]
    if poin.status == "Tidak Dinilai":
        # Belum pernah dievaluasi (bukan "sudah memenuhi ketentuan") — lihat _saran_tidak_dinilai().
        saran = _saran_tidak_dinilai(poin)
        target = None
        langkah_konkret = []
    elif apakah_aman(poin):
        # Tidak ada pelanggaran utk ditindaklanjuti -> saran & target deterministik, TAPI
        # reasoning_pendek/panjang & sitasi di atas tetap murni dari LLM (lihat docstring).
        saran = _SARAN_AMAN
        target = None
        langkah_konkret = []
    else:
        saran_mitigasi = _saran_mitigasi_dampak(poin)
        if saran_mitigasi:
            # target/langkah_konkret SUDAH dirakit dari fakta['target_mitigasi'] di atas (yg,
            # via calculator.py, JUGA sudah prioritaskan rekomendasi_mitigasi BE) — cuma saran yg
            # ditimpa verbatim di sini, satu sumber kebenaran dgn target/langkah_konkret di atas.
            saran = saran_mitigasi

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
            langkah_konkret=langkah_konkret,
        ),
        low_confidence=False,
    )
