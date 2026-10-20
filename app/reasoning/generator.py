"""Inti reasoning — generate_poin() untuk satu PoinKonteks (dari app/adapter.py).

Neuro-simbolik (CLAUDE.md): LLM diberi skema SEMPIT — hanya bagian bahasa (reasoning, kutipan,
saran, disclaimer). Semua metadata (poin_id, kategori, status, tipe rekomendasi, target) dan
metadata sitasi (dokumen/pasal/halaman) dirakit DI SINI dari poin/calculator/chunk yang benar-benar
diretrieve — bukan dipercaya dari output LLM. citation_id yang tidak dikenal (halusinasi) dibuang.
"""

import logging

from app.reasoning import llm_client
from app.reasoning.calculator import (
    bangun_langkah_dampak_poin,
    bangun_saran_mitigasi_dampak,
    pilih_target_dampak_poin,
    bangun_langkah_konkret_intensitas,
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


_ID_PASAL_53_AYAT_3 = "rdtr-sleman-tengah-p53-a3"


def _lengkapi_sitasi_dampak(sitasi: list, poin: PoinKonteks, chunk_by_id: dict,
                            retriever: Retriever | None = None) -> list:
    """Setiap luaran harus bersitasi (PM 2026-10-18), jangan bergantung pada LLM memilihnya.
    - Kawasan resapan: Pasal 53 ayat (3) (zero delta Q huruf a, sumur/kolam huruf d).
    - Bukan resapan dengan mitigasi: Lampiran VI (batas KDB/KDH), dasar langkah menurunkan KDB dan menaikkan KDH.
    Chunk dicari dulu di hasil retrieval poin, lalu lewat satu pencarian tambahan (chunk pengisi Lampiran VI
    sengaja disaring dari konteks LLM dampak non-resapan, jadi tidak ada di `chunk_by_id`). Hanya chunk asli
    dari database yang dipakai, tak pernah mengarang."""
    if poin.poin_id != "dampak" or poin.status == "Tidak Dinilai":
        return sitasi
    ada = {x.citation_id for x in sitasi}
    tambah = None
    if poin.kawasan_resapan is True:
        tambah = chunk_by_id.get(_ID_PASAL_53_AYAT_3)
        cocok, query = chunk_ketentuan_resapan, _QUERY_KETENTUAN_RESAPAN
    elif poin.kawasan_resapan is False and (poin.fakta or {}).get("target_mitigasi"):
        tambah = next((c for cid, c in chunk_by_id.items() if "-vi-" in cid), None)
        cocok, query = (lambda c: "-vi-" in (c.id or "")), _QUERY_INTENSITAS_TAJAM
    else:
        return sitasi
    if tambah is None and retriever is not None:
        try:
            filters = (RetrievalFilters(zona=poin.zona_subzone) if poin.zona_subzone
                       else RetrievalFilters(zona_prefix=_zona_prefix_dari_nama(poin.zona)))
            tambah = next((c for c in retriever.search(query, filters, top_k=6)
                           if cocok(c) and _berlaku_utk_subzona(c, poin.zona_subzone)), None)
        except Exception:  # noqa: BLE001 — sitasi tambahan bersifat opsional, jangan menggagalkan poin
            logger.warning("Pencarian sitasi tambahan untuk dampak gagal.", exc_info=True)
    if tambah is None or tambah.id in ada:
        return sitasi
    return [*sitasi, SitasiOutput(
        citation_id=tambah.id, dokumen=tambah.dokumen, pasal=tambah.pasal or "",
        halaman=tambah.halaman or 0, kutipan=tambah.teks, terverifikasi=True)]


_MAKS_SYARAT_ITBX_DALAM_SARAN = 5
_DAFTAR_PANJANG_ITBX = 8


def saran_itbx_deterministik(poin: PoinKonteks) -> str | None:
    """Saran ITBX bersyarat/terbatas dari kode, bukan LLM: syarat diambil VERBATIM dari
    `keterangan_ketentuan` kegiatan pemohon. Hasil uji 2026-10-19: penolakan guardrail terbesar di ITBX adalah
    saran yang meminjam syarat kelompok lain, mengarang kewajiban dokumen/bukti, atau menyatakan syarat
    "sudah terpenuhi". Menyalin syarat apa adanya menutup ketiganya dan menghemat panggilan LLM.

    Daftar pendek (sampai 8 item) dianggap sudah relevan dari BE. Daftar panjang (satu dokumen zona memuat
    semua kelompok kegiatan) hanya diambil item yang berbagi kata kunci dengan kegiatan pemohon. Tidak ada
    yang relevan -> None, dan saran tetap dari LLM seperti sebelumnya."""
    if poin.poin_id != "itbx" or poin.status not in ("T", "B", "TB"):
        return None
    fakta = poin.fakta or {}
    ketentuan = [str(x).strip() for x in (fakta.get("keterangan_ketentuan") or []) if str(x).strip()]
    if not ketentuan:
        return None
    if len(ketentuan) > _DAFTAR_PANJANG_ITBX:
        kata = _kata_kunci_itbx(fakta.get("kegiatan_diusulkan") or "")
        ketentuan = [k for k in ketentuan if kata and kata & _kata_kunci_itbx(k)]
        if not ketentuan:
            return None
    terpilih = ketentuan[:_MAKS_SYARAT_ITBX_DALAM_SARAN]
    butir = []
    for i, item in enumerate(terpilih, 1):
        teks = item.replace(";", ",").rstrip()
        if teks and teks[-1] not in ".!?":
            teks += "."
        butir.append(f"{i}. Syarat yang harus dipenuhi pemohon: {teks}")
    return "\n".join(butir)


def _kata_kunci_itbx(teks: str) -> set[str]:
    from app.reasoning.prompts import _kata_kunci
    return _kata_kunci(teks)


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
    # Aturan PM 2026-10-18: saran dirakit dari angka BE (wajib, sumur hanya di kawasan resapan), bukan
    # menyalin kalimat BE yang menulis sumur sebagai alternatif.
    saran_pm = bangun_saran_mitigasi_dampak(target_mitigasi, poin.kawasan_resapan)
    if saran_pm:
        return saran_pm
    saran = target_mitigasi.get("saran_be") or None
    if saran and poin.kawasan_resapan is True:
        saran = (saran.rstrip() + " Lokasi berada di kawasan resapan air, sehingga ketentuan khusus "
                 "kawasan resapan air berlaku, antara lain memastikan air tidak melimpas (zero delta Q).")
    elif saran and poin.kawasan_resapan is False:
        saran = (saran.rstrip() + " Lokasi bukan kawasan resapan air, sehingga sumur atau kolam resapan "
                 "di sini merupakan rekomendasi mitigasi, bukan kewajiban dari ketentuan kawasan resapan air.")
    return saran


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
# `dampak` SENGAJA tidak masuk. Saat keputusan ini diambil alasannya masih lemah (n=5,
# p=0.625 — tak terbaca, jadi perilaku lama dipertahankan). Setelah topik dampak dibangkitkan
# berlabel aturan dari Pasal 50 & 53, n jadi 53 dan arahnya TERBUKTI berlawanan: dense-saja
# KALAH, nDCG@3 0.251 vs 0.330 (p<0.001, efek -0.66). Belakangan ketahuan label itu mencakup
# dua tema sementara query produksi hanya satu, jadi kesimpulannya diuji ulang dengan label
# yang setema: selisihnya justru MELEBAR (0.707 vs 0.924, p<0.001), bukan hilang. Keputusan
# ini karena itu tidak bergantung pada cacat label tersebut. `itbx` juga tidak masuk: ia lewat
# get_by_reference dan tak menyentuh fusi.
_TANPA_LEXICAL_PER_POIN = frozenset({"intensitas"})

# Kedalaman konteks per poin. KOSONG di produksi — seluruh poin memakai `top_k_dukungan`
# (3). Dict ini ada supaya harness evaluasi bisa menjalankan eksperimen k lain TANPA
# mengubah perilaku default: ablasi diaktifkan lewat parameter di sisi eksperimen, bukan
# dengan menggeser produksi lalu lupa menggesernya kembali.
#
# Catatan ukuran, supaya keputusan menaikkannya tidak diambil dari metrik retrieval saja:
# k=5 menaikkan Recall@k dari 63% ke 94% langit-langitnya, tapi menambah ~485-1.042 token
# per poin, dan tiga poin berjalan paralel. Risiko yang TIDAK terlihat di metrik retrieval
# adalah membanjiri LLM sampai ia gagal memilih sitasi sama sekali (APP-2026-6191), jadi
# kenaikan k wajib diuji di sisi generasi lebih dulu — lihat eval/replay_k.py.
_TOP_K_PER_POIN: dict[str, int] = {}


def top_k_utk(poin_id: str, default: int) -> int:
    """Kedalaman konteks utk satu poin. Produksi selalu memakai `default`."""
    return _TOP_K_PER_POIN.get(poin_id, default)

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


_QUERY_KETENTUAN_RESAPAN = "ketentuan khusus kawasan resapan air zero delta Q sumur resapan"


def chunk_ketentuan_resapan(chunk: Chunk) -> bool:
    """Chunk Pasal 53 (ketentuan khusus kawasan resapan air). Chunk Lampiran VI yang cuma menyebut
    "resapan/non-resapan" sebagai kolom tabel BUKAN ketentuan khusus, jadi tak ikut."""
    return (chunk.pasal or "").strip() == "53" and "resapan" in (chunk.teks or "").lower()


def chunk_rawan_kekeringan(chunk: Chunk) -> bool:
    """Chunk Pasal 50 yang mengatur kawasan rawan bencana kekeringan (ayat 10 dan 11).

    Berlaku HANYA di SWP E Blok I.E.1 dan I.E.2, dan tak ada hubungannya dgn status resapan air
    (keduanya kebetulan memuat "zero delta Q"). Payload belum membawa ID blok, jadi berlaku-tidaknya
    pasal ini tak bisa dipastikan -> tak dikutip sebagai dasar poin dampak (APP-2026-UJI-4 mengutipnya
    di lokasi yang bukan kawasan rawan kekeringan)."""
    return (chunk.pasal or "").strip() == "50" and "kekeringan" in (chunk.teks or "").lower()


def chunk_pengisi_dampak(chunk: Chunk) -> bool:
    """Chunk yang bukan dasar hukum hidrologi: definisi Pasal 1 (mis. "Jaringan Drainase Lokal") dan
    tabel intensitas Lampiran VI (dibahas poin intensitas). Pada APP-2026-UJI-3/-4 keduanya jadi satu-
    satunya sitasi dampak non-resapan: sekadar pengisi, bukan dasar."""
    teks = (chunk.teks or "").lstrip()
    return ((chunk.pasal or "").strip() in ("1", "44") or "-vi-" in (chunk.id or "")
            or teks.startswith("Lampiran VI"))


def _saring_pengisi_poin(chunks: list[Chunk], poin: PoinKonteks) -> list[Chunk]:
    """ITBX tak butuh tabel intensitas (Lampiran VI), intensitas tak butuh syarat kegiatan (Lampiran V.B)
    — masing-masing hanya pengisi di poin yang lain (APP-2026-UJI-1..4), dan sumber kontradiksi antar poin
    (UJI-2: ITBX menyuruh KDB 60% dari rentang "50-60%" tabel intensitas). Bila penyaringan mengosongkan
    daftar, daftar asli dipertahankan: lebih baik sitasi kurang tepat drpd tanpa sitasi sama sekali."""
    if poin.poin_id == "itbx":
        sisa = [c for c in chunks if "-vi-" not in (c.id or "") and not (c.teks or "").lstrip().startswith("Lampiran VI")]
        if poin.status in ("I", "X"):
            # Lampiran V.B memuat SYARAT kegiatan bersyarat/terbatas (T/B). Kegiatan berstatus I tidak punya syarat
            # itu, dan LLM yang membacanya meminjam syarat kelompok lain (APP-2026-0022 zona R: dokumen lingkungan,
            # pengelolaan limbah) lalu ditolak guardrail tiga kali. Dikosongkan walau tak ada pengganti: anchor BE
            # (Matriks ITBX) tetap tersedia sebagai sitasi.
            return [c for c in sisa if "-vb-" not in (c.id or "") and not (c.teks or "").lstrip().startswith("Lampiran V.B")]
    elif poin.poin_id == "intensitas":
        sisa = [c for c in chunks if "-vb-" not in (c.id or "") and not (c.teks or "").lstrip().startswith("Lampiran V.B")]
    else:
        return chunks
    return sisa or chunks


def _berlaku_utk_subzona(chunk: Chunk, subzona: str | None) -> bool:
    """Pasal 53 ayat (2) dan (3) memuat daftar kode zona yang berbeda. Hanya ayat yang menyebut kode
    sub-zona pemohon yang berlaku (APP-2026-UJI-2 mengutip ayat 2 untuk R-3 padahal ayat 2 hanya untuk
    BA, PS, RTH-3, RTH-4, RTH-7)."""
    if not subzona or (chunk.pasal or "").strip() != "53":
        return True
    teks = (chunk.teks or "").lstrip()
    if not (teks.startswith("(2)") or teks.startswith("(3)")):
        return True
    return f"kode {subzona.strip().lower()}" in teks.lower()


def _sesuaikan_dgn_status_resapan(chunks: list[Chunk], poin: PoinKonteks, retriever: Retriever,
                                  top_k: int) -> list[Chunk]:
    """Poin dampak: sitasi ketentuan resapan air mengikuti `kawasan_resapan` dari BE.

    False -> buang (APP-2026-UJI-4: lokasi non-resapan tetap mengutip Pasal 53 ayat 3 hanya karena
    R-3 ada di daftar zonanya; pasal itu berlaku per blok resapan, bukan per zona).
    True  -> pastikan ada (tak boleh bergantung pada untung-untungan retrieval).
    None  -> perilaku lama."""
    if poin.poin_id != "dampak":
        return chunks
    if poin.rawan_kekeringan is not True:
        chunks = [c for c in chunks if not chunk_rawan_kekeringan(c)]
    chunks = [c for c in chunks if not chunk_pengisi_dampak(c) and _berlaku_utk_subzona(c, poin.zona_subzone)]
    if poin.kawasan_resapan is None:
        return chunks[:top_k] if len(chunks) > top_k else chunks
    if poin.kawasan_resapan is False:
        # Pasal 53 bukan dasar kewajiban utk lokasi non-resapan. (LLM tetap boleh MENYEBUT bahwa
        # pasal itu tidak berlaku — lihat guardrail._cek_klaim_resapan.)
        return [c for c in chunks if not chunk_ketentuan_resapan(c)][:top_k]
    if any(chunk_ketentuan_resapan(c) for c in chunks):
        return chunks
    filters = (RetrievalFilters(zona=poin.zona_subzone) if poin.zona_subzone
               else RetrievalFilters(zona_prefix=_zona_prefix_dari_nama(poin.zona)))
    tambahan = [c for c in retriever.search(_QUERY_KETENTUAN_RESAPAN, filters, top_k=4)
                if chunk_ketentuan_resapan(c) and _berlaku_utk_subzona(c, poin.zona_subzone)][:1]
    return (tambahan + chunks)[: max(top_k, len(tambahan))]


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
    top_k = top_k_utk(poin.poin_id, top_k_dukungan)
    chunks = _pilih_chunks_referensi(chunks, poin, top_k)
    # Over-fetch sedikit kalau chunk resapan akan dibuang, supaya jatah top_k tak habis percuma.
    top_k_cari = top_k + 2 if poin.poin_id == "dampak" else top_k
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
        chunks = retriever.search(query_fallback, filters, top_k=top_k_cari,
                                  tanpa_lexical=poin.poin_id in _TANPA_LEXICAL_PER_POIN)
    return _saring_pengisi_poin(_sesuaikan_dgn_status_resapan(chunks, poin, retriever, top_k), poin)


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
        # Permintaan user 2026-10-14: dampak yang DINILAI (Rendah/Sedang/Tinggi...) tak pernah
        # "aman tanpa saran" lagi — selalu lewat LLM + saran/langkah konkret. Hanya "Tidak Dinilai"
        # yang tetap tanpa saran (ditangani cabang tersendiri di generate_poin/perbaiki_poin).
        return poin.status == "Tidak Dinilai"
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
                    # SELALU teks BE apa adanya; kutipan dari LLM hanya cadangan bila BE kosong
                    # (APP-2026-UJI-1: LLM menambah awalan ke kutipan jangkar).
                    kutipan=anchor.kutipan or kutipan,
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

    sitasi = _lengkapi_sitasi_dampak(sitasi, poin, chunk_by_id, retriever)

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
        target = pilih_target_dampak_poin(poin.fakta)
        langkah_konkret = bangun_langkah_dampak_poin(poin.fakta)

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
        saran_itbx = saran_itbx_deterministik(poin)
        if saran_itbx:
            saran = saran_itbx
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
