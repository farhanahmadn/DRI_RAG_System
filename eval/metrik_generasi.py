"""eval/metrik_generasi.py — metrik SISI GENERASI, offline dari `logs/precheck.jsonl` + korpus.

Kenapa modul ini ada. `eval/eval_rag.py` mengukur leg retrieval (apakah chunk benar masuk top-k);
`eval/metrik_atribusi.py` mengukur atribusi sitasi (apakah id & kutipannya nyata). Keduanya diam
soal dua pertanyaan yang paling sering ditanyakan tentang RAG: apakah LLM PATUH pada konteks
(faithfulness), dan apakah jawabannya RELEVAN dengan permohonan yang ditanyakan (answer relevance).

Yang membedakan modul ini dari kerangka LLM-as-judge: seluruh empat metrik di bawah **mekanis** —
tidak ada model yang menilai model, tidak ada anotator, tidak ada panggilan API. Semuanya dihitung
ulang dari `request` + `response` yang sudah tersimpan di log, sehingga bisa dijalankan kapan pun
tanpa kuota dan hasilnya identik tiap kali dijalankan atas log yang sama.

EMPAT METRIK

1. `faithfulness_numerik` — setiap angka >=2 digit di narasi harus terlacak ke fakta sumber poin itu,
   ke pasal/dokumen yang benar-benar disitasi, atau ke teks chunk yang benar-benar disitasi. Ini
   TIDAK menulis ulang logikanya: ia memanggil `guardrail._angka_terlacak_ke_sumber` — cek yang sama
   yang berjalan di produksi sebagai GERBANG — lalu melaporkan laju lolosnya sebagai ANGKA. Selama
   ini cek itu hanya menolak/meloloskan tanpa pernah dilaporkan.

   Replay-nya setara dengan produksi, bukan pendekatan: `guardrail._teks_chunk_disitasi` membatasi
   diri pada chunk yang id-nya muncul di `sitasi[]`, jadi memasok chunk hasil sitasi dari DB
   memberikan `sumber` yang identik dengan memasok seluruh hasil retrieval saat itu.

2. `kekhususan` — answer relevance, diadaptasi ke sistem ini. Definisi RAGAS (rekonstruksi
   pertanyaan dari jawaban, lalu ukur kesamaannya) hampir DEGENERATE di sini: "query"-nya template
   tetap per poin dan skema jawabannya dipaksa Pydantic, jadi skornya selalu tinggi tanpa memberi
   informasi. Yang bermakna adalah apakah narasi membahas permohonan INI — menyebut zona pemohon,
   kegiatan yang diusulkan, parameter yang benar-benar gagal — atau boilerplate aman yang cocok
   untuk permohonan apa pun. Jangkarnya diambil dari fakta adapter, jadi tiap jangkar punya nilai
   kebenaran yang tak bisa diperdebatkan.

3. `ketepatan_arah` — faithfulness terhadap verdict, bukan terhadap teks. Tiga sub-cek, masing-masing
   diikat ke fakta deterministik back-end, bukan ke selera bahasa:
     a. per-parameter intensitas: `parameter.<kdb|klb|kdh>.memenuhi` (boolean BE) vs klaim arah di
        narasi pada jendela sekitar nama parameter itu.
     b. kebutuhan mitigasi dampak: `mitigasi.perlu_mitigasi` (True hanya utk Tinggi/Sangat Tinggi,
        lihat `rekomendasi.KATEGORI_DAMPAK_BERSYARAT`) vs klaim perlu/tak perlu mitigasi.
     c. klaim kategori ITBX: narasi yang MENGKLASIFIKASIKAN permohonan ini ke huruf ITBX tertentu
        harus menyebut huruf yang sama dengan `status`.
   Kosakata arahnya sengaja dibatasi ke empat kata kerja yang NETRAL terhadap arah ambang
   (`memenuhi` vs `melampaui`/`melebihi`/`melanggar`). Kata posisi seperti "di bawah" DILARANG masuk
   lexicon: ia menandakan patuh untuk KDB/KLB (ambang maksimum) tapi menandakan pelanggaran untuk
   KDH (ambang minimum), jadi memakainya akan menghasilkan temuan palsu yang arahnya terbalik.

4. `boilerplate` — jawaban yang nyaris identik untuk dua permohonan BERBEDA bukan jawaban tentang
   permohonannya. Diukur sebagai kemiripan 5-gram kata ke tetangga terdekat, dalam poin yang sama,
   antar permohonan yang berbeda. Dipilah: pasangan ber-status BERBEDA (kemiripan tinggi di sini
   adalah cacat nyata) vs ber-status SAMA (kemiripan wajar — dua pemohon dengan verdict & zona sama
   memang pantas dijelaskan dengan cara mirip). Tanpa label, tanpa ambang yang disetel ke data:
   `_AMBANG_MIRIP` ditetapkan dari konvensi deteksi near-duplicate sebelum angkanya dilihat.

CARA MEMBACA ANGKANYA

Sama seperti `metrik_atribusi`: log BUKAN trafik produksi bersih. Baris stub dari pytest, baris
`MockRetriever`, dan baris era 8-indikator lama dibuang, dan penyebutnya selalu dicetak. Angka tanpa
penyebut tidak boleh dikutip.

WAKTU ADALAH BAGIAN DARI METRIK. Log merentang beberapa versi kode, jadi `--sejak` wajib dipakai saat
angkanya dikutip — kalau tidak, yang terukur adalah campuran perilaku sebelum dan sesudah perbaikan.

BATAS YANG TIDAK BISA DIHILANGKAN MODUL INI. Faithfulness di sini berhenti di angka, arah verdict,
dan kekhususan. Entailment tingkat-KLAIM — apakah kalimat penafsiran di `reasoning_panjang` benar
tersirat dari ayat yang disitasi — TIDAK terukur di sini dan tidak bisa diukur secara mekanis. LLM
bisa mengutip Pasal 53 dengan benar lalu menyimpulkan sesuatu yang ayat itu tidak katakan, dan
seluruh metrik di berkas ini akan tetap hijau.

CLI:
  python -m eval.metrik_generasi
  python -m eval.metrik_generasi --sejak 2026-09-07 --out eval/generasi.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.adapter import adaptasi
from app.reasoning.guardrail import _angka_terlacak_ke_sumber
from app.reasoning.guardrail import _RE_ANGKA_MENCURIGAKAN
from app.reasoning.templates import template_aman, template_low_confidence
from app.retrieval.base import Chunk
from app.schemas import L2Assessment, PoinKonteks, PoinOutput

load_dotenv()

_AKAR = Path(__file__).parent.parent
_LOG = _AKAR / "logs" / "precheck.jsonl"
_OUT = Path(__file__).parent / "generasi.json"

_POIN_SEKARANG = ("itbx", "intensitas", "dampak")

# --- parameter pencocokan, SELURUHNYA ditetapkan sebelum melihat hasil -------------------------

# Lingkup negasi ditentukan BATAS KLAUSA, bukan jumlah karakter. Pendekatan negasi berlingkup-klausa
# ini standar (NegEx). Jendela karakter tetap terbukti salah dua kali selama penyetelan modul ini:
#   - 30 char: "namun TIDAK mencapai tingkat tinggi yang memerlukan mitigasi" (negator 35 char di
#     depan) ditandai menyimpang, padahal kalimatnya benar.
#   - 80 char: "TIDAK ada indikasi bahwa kegiatan ... atau memerlukan mitigasi tambahan" (88 char)
#     ikut ditandai.
# Di dalam satu klausa bahasa Indonesia, negator memang berlingkup ke seluruh klausa, jadi batas
# klausa sudah merupakan pembatas yang benar. Cap di bawah tinggal pengaman terhadap teks yang
# sama sekali tak bertanda baca — bukan bagian dari definisi lingkupnya.
_JENDELA_NEGASI_MAKS = 400

# Penanda batas klausa. "dan"/"atau"/"serta" SENGAJA TIDAK termasuk: dalam bahasa Indonesia negasi
# menembus konjungsi aditif ("tidak melampaui KDB dan KLB" menegasikan keduanya), jadi
# memperlakukannya sebagai batas akan menghilangkan negasi yang masih berlaku.
_RE_BATAS_KLAUSA = re.compile(
    r"[.,;:()\[\]]"
    r"|\b(?:namun|tetapi|tapi|sehingga|karena|sedangkan|meskipun|walaupun|jika|apabila)\b"
)

# Jendela DI DEPAN nama parameter (kdb/klb/kdh) tempat klaim arah dicari. Bahasa Indonesia menaruh
# kata kerja SESUDAH subjeknya ("KDB melampaui ambang"), jadi jendelanya asimetris: sejauh ini ke
# depan, dan ke belakang hanya sampai batas klausa (lihat `_jendela_parameter`).
_JENDELA_PARAMETER = 90

# Kemiripan 5-gram di atas ini dihitung sebagai near-duplicate. Konvensi umum deteksi
# near-duplicate; DITETAPKAN DI SINI sebelum distribusinya dilihat, supaya bukan hasil penyetelan.
_AMBANG_MIRIP = 0.5
_N_GRAM = 5

_NEGATOR = re.compile(r"\b(tidak|belum|bukan|tanpa|non|kurang)\b")

# Kata kerja arah yang NETRAL terhadap jenis ambang (maksimum vs minimum). Lihat docstring modul
# kenapa kata posisi ("di bawah", "melebihi batas bawah", ...) sengaja TIDAK ada di sini.
_V_PATUH = ("memenuhi",)
_V_LANGGAR = ("melampaui", "melebihi", "melanggar")

# Nama panjang parameter intensitas, supaya narasi yang menulis "koefisien dasar bangunan" alih-alih
# "KDB" tetap terbaca.
_ALIAS_PARAMETER = {
    "kdb": ("kdb", "koefisien dasar bangunan"),
    "klb": ("klb", "koefisien lantai bangunan"),
    "kdh": ("kdh", "koefisien daerah hijau"),
}

# Klaim perlu/tak perlu mitigasi pada poin dampak.
_FRASA_PERLU_MITIGASI = ("memerlukan mitigasi", "perlu mitigasi", "wajib mitigasi",
                         "memerlukan tindakan mitigasi", "perlu tindakan mitigasi",
                         "diperlukan mitigasi", "diperlukan tindakan mitigasi")

# Narasi yang MENGKLASIFIKASIKAN permohonan ini ke suatu huruf ITBX. Hanya pola self-classification
# yang ditangkap — penyebutan kategori lain sebagai konteks ("kegiatan Terbatas di zona ini
# meliputi ...") sengaja dilewatkan supaya tak jadi temuan palsu.
_RE_KLAIM_ITBX = re.compile(
    r"(?:termasuk|masuk|tergolong|diklasifikasikan|klasifikasi|merupakan|berada)[^.]{0,80}?"
    r"itbx\s+([itbx])\b"
    r"|itbx\s+([itbx])\b[\s*_]*\(\s*\**\s*(?:diizinkan|terbatas|bersyarat|tidak\s+diizinkan)"
)

_RE_BUKAN_KATA = re.compile(r"[^0-9a-z]+")
_RE_SPASI = re.compile(r"\s+")


def _normalisasi(teks: str) -> str:
    """Huruf kecil, tanda baca -> spasi, spasi tunggal. Sama seperti metrik_atribusi.

    Dipakai di tempat yang tanda bacanya tak bermakna: pencocokan jangkar (`kekhususan`) dan
    perbandingan n-gram (`boilerplate`).
    """
    return _RE_BUKAN_KATA.sub(" ", (teks or "").lower()).strip()


def _teks_klausa(teks: str) -> str:
    """Huruf kecil, spasi tunggal, TANDA BACA DIPERTAHANKAN.

    Dipakai di jalur `ketepatan_arah`, yang bersandar pada tanda baca & konjungsi sebagai batas
    klausa untuk menentukan lingkup negasi. Menormalisasi tanda baca lebih dulu akan menghapus
    justru penanda yang dibutuhkan.
    """
    return _RE_SPASI.sub(" ", (teks or "").lower()).strip()


def _dokumen_mock() -> frozenset[str]:
    """Nama dokumen MockRetriever — penanda baris log hasil replay/test. Dibaca dari mock-nya
    sendiri, bukan disalin sebagai string, supaya ikut kalau fixture-nya berganti nama."""
    try:
        from app.retrieval.mock import _build_mock_chunks

        return frozenset(c.dokumen for c in _build_mock_chunks() if c.dokumen)
    except Exception:
        return frozenset()


def _ternegasi(teks: str, awal: int) -> bool:
    """Apakah ada negator di dalam KLAUSA yang sama, sebelum posisi `awal`.

    `teks` harus hasil `_teks_klausa` (tanda baca utuh) — dengan teks ternormalisasi, batas
    klausanya hilang dan fungsi ini melebar sampai 80 karakter tanpa pembatas.
    """
    mulai = max(0, awal - _JENDELA_NEGASI_MAKS)
    span = teks[mulai:awal]
    batas = None
    for m in _RE_BATAS_KLAUSA.finditer(span):
        batas = m.end()
    if batas is not None:
        span = span[batas:]
    return _NEGATOR.search(span) is not None


def _klaim_arah(teks_norm: str) -> set[str]:
    """Klaim arah yang terbaca di sepotong teks (hasil `_teks_klausa`): {'patuh'}, {'langgar'},
    keduanya, atau kosong.

    "memenuhi" afirmatif dan "tidak melampaui" sama-sama mengklaim PATUH; "tidak memenuhi" dan
    "melampaui" afirmatif sama-sama mengklaim LANGGAR. Memperlakukan negasi sebagai pembalik
    (bukan sebagai kata terlarang) itulah yang membuat cek ini tidak menghukum kalimat yang benar.
    """
    klaim: set[str] = set()
    for kata in _V_PATUH:
        for m in re.finditer(rf"\b{kata}\b", teks_norm):
            klaim.add("langgar" if _ternegasi(teks_norm, m.start()) else "patuh")
    for kata in _V_LANGGAR:
        for m in re.finditer(rf"\b{kata}\b", teks_norm):
            klaim.add("patuh" if _ternegasi(teks_norm, m.start()) else "langgar")
    return klaim


def _n_gram(teks_norm: str, n: int = _N_GRAM) -> frozenset[tuple[str, ...]]:
    t = teks_norm.split()
    if len(t) < n:
        return frozenset()
    return frozenset(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _proporsi(pembilang: int, penyebut: int) -> float | None:
    """None kalau penyebutnya nol — 0.0 akan terbaca sebagai 'gagal total', bukan 'tak terukur'."""
    return (pembilang / penyebut) if penyebut else None


# ---------------------------------------------------------------------------
# Pemuatan
# ---------------------------------------------------------------------------


def _jalur_narasi(konteks: PoinKonteks, keluaran: PoinOutput) -> str:
    """"llm" | "template_aman" | "template_low_confidence".

    KENAPA INI WAJIB ADA. Tidak semua narasi di log berasal dari LLM. `app/reasoning/templates.py`
    menghasilkan dua teks DETERMINISTIK tanpa memanggil model sama sekali: `template_aman` untuk
    poin yang jelas lolos (hemat kuota) dan `template_low_confidence` saat retry guardrail habis.
    Keduanya identik lintas permohonan dan tak pernah menyebut partikular apa pun — jadi
    memasukkannya ke metrik mutu LLM menghasilkan angka yang mengukur TEMPLATE, bukan model.
    Terbukti saat modul ini pertama dijalankan: satu teks `template_low_confidence` muncul di 11
    permohonan berbeda, dan `kekhususan` poin dampak keluar 0.0%.

    Deteksinya membangkitkan ulang template dari `PoinKonteks` yang sama lalu membandingkan teksnya
    — BUKAN mencocokkan string yang disalin ke sini. Kalau templatenya diubah, deteksi ini ikut
    berubah sendiri, ketimbang diam-diam mulai menganggap teks template sebagai keluaran LLM.
    """
    narasi = _normalisasi(keluaran.reasoning_panjang)
    if not narasi:
        return "llm"
    for nama, bangun in (("template_low_confidence", template_low_confidence),
                         ("template_aman", template_aman)):
        try:
            if narasi == _normalisasi(bangun(konteks).reasoning_panjang):
                return nama
        except Exception:
            continue
    return "llm"


@dataclass
class Kasus:
    """Satu poin dari satu permohonan: konteks deterministik (hasil adapter atas `request` yang
    tersimpan) disandingkan dengan luaran LLM yang tercatat."""

    permohonan: str
    timestamp: str
    poin_id: str
    konteks: PoinKonteks
    keluaran: PoinOutput
    zona: str | None
    zona_subzone: str | None
    jalur: str = "llm"

    @property
    def teks_narasi(self) -> str:
        """Teks yang discan — persis ruang lingkup guardrail Cek #6 (`_cek_konsistensi_numerik`):
        kedua reasoning + saran, TANPA `sitasi[].kutipan` (kutipan pasal sah memuat nomor)."""
        return (f"{self.keluaran.reasoning_pendek} {self.keluaran.reasoning_panjang} "
                f"{self.keluaran.rekomendasi.saran}")


def muat_kasus(path: Path, sejak: str | None = None,
               semua_jalan: bool = False) -> tuple[list[Kasus], dict]:
    """Baca log, bangun ulang PoinKonteks dari `request`, dan laporkan apa saja yang dibuang.

    Membangun ulang konteks lewat `adaptasi()` (bukan menyimpan fakta hasil adapter di log) berarti
    metrik ini selalu diukur terhadap fakta yang DITURUNKAN KODE HARI INI dari permohonan asli —
    kalau adapter berubah, angkanya ikut, ketimbang membeku pada snapshot lama.

    SATU NARASI PER (PERMOHONAN, POIN) — yang terbaru. Log bukan trafik unik: 318 narasi di log
    hanya berasal dari 20 permohonan, satu di antaranya dijalankan 30 kali selama pengembangan.
    Merata-ratakan seluruh baris berarti membobot permohonan menurut berapa kali ia kebetulan
    di-replay, dan permohonan yang paling sering diulang justru yang paling sering bermasalah —
    jadi biasnya bukan acak. `semua_jalan=True` mematikan dedup ini (dipakai untuk memeriksa
    sebaran antar-jalan, bukan untuk mengutip angka).
    """
    dok_mock = _dokumen_mock()
    keluar: list[Kasus] = []
    lewat: Counter = Counter()

    for baris in path.read_text(encoding="utf-8").splitlines():
        if not baris.strip():
            continue
        try:
            d = json.loads(baris)
        except Exception:
            lewat["baris tak terbaca"] += 1
            continue
        if sejak and str(d.get("timestamp") or "")[:10] < sejak:
            lewat[f"sebelum {sejak}"] += 1
            continue

        resp = d.get("response") or {}
        poin = [p for p in (resp.get("poin") or []) if isinstance(p, dict)]
        if not any(p.get("poin_id") in _POIN_SEKARANG for p in poin):
            lewat["era 8-indikator lama"] += 1
            continue
        if any("stub" in (p.get("reasoning_pendek") or "").lower() for p in poin):
            lewat["narasi stub (pytest/fixture)"] += 1
            continue
        sitasi = [s for p in poin for s in (p.get("sitasi") or []) if isinstance(s, dict)]
        if any((s.get("dokumen") or "") in dok_mock for s in sitasi):
            lewat["MockRetriever (replay/test lokal)"] += 1
            continue

        req = d.get("request") or {}
        try:
            hasil_adapter = adaptasi(L2Assessment.model_validate(req))
        except Exception as exc:
            lewat[f"request tak bisa divalidasi ulang: {type(exc).__name__}"] += 1
            continue
        konteks = {p.poin_id: p for p in hasil_adapter.poin}

        lok = req.get("lokasi") or {}
        pid_permohonan = str(req.get("application_number") or req.get("application_id") or "?")
        tstamp = str(d.get("timestamp") or "")[:19]
        for p in poin:
            pid = p.get("poin_id")
            if pid not in _POIN_SEKARANG or pid not in konteks:
                continue
            if not (p.get("reasoning_panjang") or "").strip():
                lewat[f"{pid}: narasi kosong"] += 1
                continue
            try:
                keluaran = PoinOutput.model_validate(p)
            except Exception:
                lewat[f"{pid}: luaran tak bisa divalidasi"] += 1
                continue
            keluar.append(Kasus(
                permohonan=pid_permohonan, timestamp=tstamp, poin_id=pid,
                konteks=konteks[pid], keluaran=keluaran,
                zona=lok.get("rdtr_zone"), zona_subzone=lok.get("rdtr_subzone"),
                jalur=_jalur_narasi(konteks[pid], keluaran),
            ))

    if not dok_mock:
        lewat["PERINGATAN: penanda dokumen mock tak terbaca"] += 1

    if not semua_jalan:
        terbaru: dict[tuple[str, str], Kasus] = {}
        for k in keluar:
            kunci = (k.permohonan, k.poin_id)
            if kunci not in terbaru or k.timestamp > terbaru[kunci].timestamp:
                terbaru[kunci] = k
        lewat["jalan ulang permohonan yg sama (dedup)"] = len(keluar) - len(terbaru)
        keluar = sorted(terbaru.values(), key=lambda k: (k.permohonan, k.poin_id))

    return keluar, dict(lewat)


def _muat_chunks_disitasi(ids: set[str]) -> dict[str, Chunk]:
    """Chunk yang id-nya benar-benar disitasi, dari DB.

    Tanpa DB, `faithfulness_numerik` dilaporkan sebagai TAK TERUKUR, bukan dihitung tanpa teks
    chunk — menjalankan cek itu tanpa `sumber` dari chunk akan menolak angka yang sah dan
    menghasilkan laju lolos yang terlalu rendah, yaitu angka yang salah dan bukan angka yang hilang.
    """
    if not ids:
        return {}
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[generasi] DATABASE_URL tak ada; faithfulness numerik dilewati")
        return {}
    try:
        import psycopg

        with psycopg.connect(dsn) as conn:
            baris = conn.execute(
                "SELECT id, level, teks, dokumen, pasal, ayat, zona FROM chunks WHERE id = ANY(%s)",
                (sorted(ids),),
            ).fetchall()
        return {
            r[0]: Chunk(id=r[0], level=r[1], teks=r[2], dokumen=r[3] or "",
                        pasal=r[4], ayat=r[5], zona=r[6])
            for r in baris
        }
    except Exception as exc:
        print(f"[generasi] DB tak terjangkau ({exc}); faithfulness numerik dilewati")
        return {}


# ---------------------------------------------------------------------------
# 1. Faithfulness numerik
# ---------------------------------------------------------------------------


def faithfulness_numerik(kasus: list[Kasus], chunks: dict[str, Chunk]) -> dict:
    """Laju lolos cek provenance angka produksi, dilaporkan sebagai angka.

    Dihitung dua tingkat: per-ANGKA (berapa banyak angka yang terlacak) dan per-NARASI (berapa
    banyak narasi yang seluruh angkanya terlacak). Yang kedua itulah yang setara dengan keputusan
    guardrail: satu angka tak terlacak sudah cukup memicu regenerasi.
    """
    if not chunks:
        return {"terukur": False, "alasan": "teks chunk yang disitasi tak tersedia (DB mati / tanpa DSN)"}

    n_angka = kena_angka = 0
    n_narasi = kena_narasi = 0
    per_poin: dict[str, Counter] = {p: Counter() for p in _POIN_SEKARANG}
    contoh_gagal: list[dict] = []

    for k in kasus:
        disitasi = [chunks[s.citation_id] for s in k.keluaran.sitasi
                    if s.citation_id in chunks]
        gagal_di_narasi: list[str] = []
        for m in _RE_ANGKA_MENCURIGAKAN.finditer(k.teks_narasi):
            angka = m.group(0)
            n_angka += 1
            per_poin[k.poin_id]["angka"] += 1
            if _angka_terlacak_ke_sumber(angka, k.konteks, k.keluaran, disitasi):
                kena_angka += 1
                per_poin[k.poin_id]["angka_kena"] += 1
            else:
                gagal_di_narasi.append(angka)
        n_narasi += 1
        per_poin[k.poin_id]["narasi"] += 1
        if not gagal_di_narasi:
            kena_narasi += 1
            per_poin[k.poin_id]["narasi_kena"] += 1
        elif len(contoh_gagal) < 15:
            contoh_gagal.append({
                "permohonan": k.permohonan, "timestamp": k.timestamp, "poin": k.poin_id,
                "angka_tak_terlacak": sorted(set(gagal_di_narasi))[:6],
                "kutipan_narasi": k.keluaran.reasoning_panjang[:200],
            })

    return {
        "terukur": True,
        "per_angka": {"n": n_angka, "kena": kena_angka, "nilai": _proporsi(kena_angka, n_angka)},
        "per_narasi": {"n": n_narasi, "kena": kena_narasi, "nilai": _proporsi(kena_narasi, n_narasi)},
        "per_poin": {
            p: {
                "angka": {"n": c["angka"], "kena": c["angka_kena"],
                          "nilai": _proporsi(c["angka_kena"], c["angka"])},
                "narasi": {"n": c["narasi"], "kena": c["narasi_kena"],
                           "nilai": _proporsi(c["narasi_kena"], c["narasi"])},
            }
            for p, c in per_poin.items() if c["narasi"]
        },
        "contoh_gagal": contoh_gagal,
    }


# ---------------------------------------------------------------------------
# 2. Kekhususan (answer relevance, diadaptasi)
# ---------------------------------------------------------------------------


def _jangkar_zona(kasus: Kasus) -> list[str]:
    """Penanda zona pemohon yang sah muncul di narasi: kode sub-zona kalau ada, dan nama zona induk
    tanpa kata "zona" (mis. "Zona Perumahan" -> "perumahan")."""
    jangkar = []
    if kasus.zona_subzone:
        jangkar.append(_normalisasi(kasus.zona_subzone))
    if kasus.zona:
        nama = _normalisasi(kasus.zona)
        nama = re.sub(r"^zona\s+", "", nama)
        if nama:
            jangkar.append(nama)
    return [j for j in jangkar if j]


def _jangkar_poin(kasus: Kasus) -> list[tuple[str, list[str], bool]]:
    """(nama_jangkar, alternatif_yang_diterima, wajib) spesifik per poin, dari fakta adapter.

    Tiap jangkar adalah fakta permohonan ini yang nilai kebenarannya tak bisa diperdebatkan — bukan
    penilaian gaya bahasa. Narasi yang tak menyebut satu pun di antaranya cocok untuk permohonan
    mana saja, dan itulah definisi tidak-relevan di sistem yang query-nya template tetap.
    """
    f = kasus.konteks.fakta
    out: list[tuple[str, list[str], bool]] = []

    if kasus.poin_id == "itbx":
        kegiatan = f.get("kegiatan_diusulkan")
        if kegiatan:
            out.append(("kegiatan_diusulkan", [_normalisasi(str(kegiatan))], True))
        # Kode KBLI hanya INFORMASIONAL: ia mengidentifikasi fakta yang SAMA dengan nama kegiatan
        # di atas, jadi mewajibkan keduanya berarti menghitung satu sinyal relevansi dua kali dan
        # memperketat agregat tanpa menambah apa pun yang diukur.
        kbli = f.get("kbli_diusulkan")
        if kbli:
            out.append(("kbli_diusulkan", [_normalisasi(str(kbli))], False))

    elif kasus.poin_id == "intensitas":
        param = f.get("parameter") or {}
        gagal = [p for p, v in param.items() if isinstance(v, dict) and v.get("memenuhi") is False]
        # Kalau ada parameter yang GAGAL, menyebut parameter itu wajib — itu inti verdict-nya.
        # Kalau semua lolos, cukup salah satu parameter disebut: narasi yang tak menyebut kdb/klb/kdh
        # sama sekali tidak menjelaskan intensitas apa pun.
        for p in (gagal or list(param.keys())):
            alias = _ALIAS_PARAMETER.get(p, (p,))
            label = f"parameter_{p}_gagal" if p in gagal else f"parameter_{p}_lolos"
            out.append((label, [_normalisasi(a) for a in alias], True))

    elif kasus.poin_id == "dampak":
        if kasus.konteks.status and kasus.konteks.status != "Tidak Dinilai":
            out.append(("kategori_dampak", [_normalisasi(kasus.konteks.status)], True))

    return out


def kekhususan(kasus: list[Kasus]) -> dict:
    """Berapa bagian narasi yang benar-benar menyebut partikular permohonannya."""
    per_jangkar: dict[str, Counter] = {}
    per_poin: dict[str, Counter] = {p: Counter() for p in _POIN_SEKARANG}
    contoh_generik: list[dict] = []

    for k in kasus:
        teks = _normalisasi(k.teks_narasi)
        # (nama, alternatif, wajib). Zona pemohon WAJIB disebut untuk itbx & intensitas — ambangnya
        # memang berasal dari tabel per-zona (Lampiran V.B / VI), jadi narasi yang tak menyebut
        # zonanya tidak menjelaskan dari mana ambangnya datang. Untuk `dampak` ia hanya
        # INFORMASIONAL: ketentuannya pasal prosa lintas-zona (mis. Pasal 53 kawasan resapan air),
        # jadi menuntutnya di sana akan menandai jawaban yang benar sebagai salah — alasan yang sama
        # dengan `metrik_atribusi._POIN_BERZONA`.
        jangkar: list[tuple[str, list[str], bool]] = _jangkar_poin(k)
        zona = _jangkar_zona(k)
        if zona:
            jangkar.append(("zona_pemohon", zona, k.poin_id != "dampak"))
        if not jangkar:
            per_poin[k.poin_id]["tak_ada_jangkar"] += 1
            continue

        hilang: list[str] = []
        for nama, alternatif, wajib in jangkar:
            label = f"{k.poin_id}/{nama}" + ("" if wajib else " (informasional)")
            c = per_jangkar.setdefault(label, Counter())
            c["n"] += 1
            if any(a and a in teks for a in alternatif):
                c["kena"] += 1
            elif wajib:
                hilang.append(nama)
        per_poin[k.poin_id]["n"] += 1
        if not hilang:
            per_poin[k.poin_id]["kena"] += 1
        elif len(contoh_generik) < 15:
            contoh_generik.append({
                "permohonan": k.permohonan, "timestamp": k.timestamp, "poin": k.poin_id,
                "jangkar_hilang": hilang,
                "kutipan_narasi": k.keluaran.reasoning_panjang[:200],
            })

    total_n = sum(c["n"] for c in per_poin.values())
    total_kena = sum(c["kena"] for c in per_poin.values())
    return {
        "semua_jangkar": {"n": total_n, "kena": total_kena, "nilai": _proporsi(total_kena, total_n)},
        "per_poin": {p: {"n": c["n"], "kena": c["kena"], "nilai": _proporsi(c["kena"], c["n"]),
                         "tak_ada_jangkar": c["tak_ada_jangkar"]}
                     for p, c in per_poin.items() if c["n"] or c["tak_ada_jangkar"]},
        "per_jangkar": {nama: {"n": c["n"], "kena": c["kena"], "nilai": _proporsi(c["kena"], c["n"])}
                        for nama, c in sorted(per_jangkar.items())},
        "contoh_generik": contoh_generik,
    }


# ---------------------------------------------------------------------------
# 3. Ketepatan arah
# ---------------------------------------------------------------------------


def _posisi_parameter(teks_norm: str, nama_param: list[str]) -> list[tuple[int, int, str]]:
    """Seluruh penyebutan parameter di teks, terurut posisi: [(awal, akhir, nama)]."""
    pos: list[tuple[int, int, str]] = []
    for nama in nama_param:
        for a in _ALIAS_PARAMETER.get(nama, (nama,)):
            if not a:
                continue
            for m in re.finditer(rf"\b{re.escape(a.lower())}\b", teks_norm):
                pos.append((m.start(), m.end(), nama))
    return sorted(pos)


def _jendela_parameter(teks_norm: str, pos: list[tuple[int, int, str]], i: int) -> str:
    """Potongan teks yang klaim arahnya boleh diatribusikan ke penyebutan ke-i.

    Ke DEPAN: sampai `_JENDELA_PARAMETER` karakter, atau sampai penyebutan parameter LAIN, mana
    yang lebih dulu. Tanpa pemotongan ini, kalimat daftar seperti "KDB melebihi 10%, KLB melebihi
    1.0, serta KDH kurang dari 88%" membuat jendela KDH ikut memuat kata kerja milik KLB — dan
    itulah yang membuat jalan pertama modul ini melaporkan 55% "arah salah" yang seluruhnya palsu.

    Ke BELAKANG: hanya sampai batas klausa (atau penyebutan parameter lain). Margin karakter tetap
    dipakai mula-mula dan terbukti salah: ia memotong di tengah frasa dan MENINGGALKAN negatornya
    di luar jendela, sehingga "KDH tidak memenuhi ambang" terbaca sebagai klaim patuh dan
    parameternya jadi ambigu. Batas klausa adalah pembatas yang benar di kedua arah.

    Penyebutan parameter yang SAMA tidak memotong: narasi sah menyebut satu parameter beberapa
    kali dalam satu klausa.
    """
    awal_i, akhir_i, nama = pos[i]
    batas_kiri = 0
    for m in _RE_BATAS_KLAUSA.finditer(teks_norm[:awal_i]):
        batas_kiri = m.end()
    for j in range(i - 1, -1, -1):
        if pos[j][2] != nama:
            batas_kiri = max(batas_kiri, pos[j][1])
            break
    batas_kanan = min(len(teks_norm), akhir_i + _JENDELA_PARAMETER)
    for j in range(i + 1, len(pos)):
        if pos[j][2] != nama:
            batas_kanan = min(batas_kanan, pos[j][0])
            break
    return teks_norm[batas_kiri:batas_kanan]


def _arah_intensitas(k: Kasus) -> tuple[int, int, int, list[str]]:
    """(dinilai, sesuai, ambigu, catatan) per-parameter untuk satu narasi intensitas."""
    teks = _teks_klausa(k.teks_narasi)
    param = {n: v for n, v in (k.konteks.fakta.get("parameter") or {}).items()
             if isinstance(v, dict) and v.get("memenuhi") is not None}
    if not param:
        return 0, 0, 0, []

    pos = _posisi_parameter(teks, list(param))
    klaim_per_param: dict[str, set[str]] = {}
    for i in range(len(pos)):
        nama = pos[i][2]
        klaim_per_param.setdefault(nama, set()).update(
            _klaim_arah(_jendela_parameter(teks, pos, i)))

    dinilai = sesuai = ambigu = 0
    catatan: list[str] = []
    for nama, v in param.items():
        klaim = klaim_per_param.get(nama) or set()
        if not klaim:
            continue  # tak disebut, atau disebut tanpa klaim arah -> tak ada yang bisa dinilai
        if len(klaim) > 1:
            # Kedua arah terbaca di jendela yang sama (mis. narasi membandingkan usulan dengan
            # ambang dua kali). TAK BISA DINILAI -> tidak masuk penyebut. Memasukkannya akan
            # menghitung "tak terbaca" sebagai "salah": jalan sebelumnya melaporkan 76.2% padahal
            # seluruh 5 selisihnya ambigu, bukan menyimpang.
            ambigu += 1
            continue
        dinilai += 1
        benar = "patuh" if v["memenuhi"] else "langgar"
        if next(iter(klaim)) == benar:
            sesuai += 1
        else:
            catatan.append(f"{nama}: fakta={benar}, narasi={next(iter(klaim))}")
    return dinilai, sesuai, ambigu, catatan


def _arah_mitigasi_dampak(k: Kasus) -> tuple[bool, bool, str]:
    """(bisa_dinilai, sesuai, catatan) untuk klaim kebutuhan mitigasi pada poin dampak."""
    mit = (k.konteks.fakta.get("mitigasi") or {})
    if "perlu_mitigasi" not in mit:
        return False, True, ""
    perlu = bool(mit["perlu_mitigasi"])
    teks = _teks_klausa(k.teks_narasi)
    klaim_perlu = klaim_tak_perlu = False
    for frasa in _FRASA_PERLU_MITIGASI:
        for m in re.finditer(rf"\b{re.escape(frasa.lower())}\b", teks):
            if _ternegasi(teks, m.start()):
                klaim_tak_perlu = True
            else:
                klaim_perlu = True
    if klaim_perlu == klaim_tak_perlu:
        # tak ada klaim, atau keduanya muncul (narasi membahas syarat & pengecualian) -> lewati
        return False, True, ""
    narasi = "perlu" if klaim_perlu else "tak perlu"
    fakta = "perlu" if perlu else "tak perlu"
    return True, narasi == fakta, "" if narasi == fakta else f"fakta={fakta}, narasi={narasi}"


def _arah_kategori_itbx(k: Kasus) -> tuple[bool, bool, str]:
    """(bisa_dinilai, sesuai, catatan) untuk klaim self-classification huruf ITBX.

    Dipakai teks huruf-kecil TANPA normalisasi tanda baca: `_RE_KLAIM_ITBX` bersandar pada titik
    sebagai batas klausa (`[^.]{0,80}`) dan pada tanda kurung bentuk "ITBX I (Diizinkan)". Kalau
    teksnya dinormalisasi lebih dulu, kedua penanda itu hilang dan polanya akan melompati batas
    kalimat — menandai penyebutan kategori lain sebagai konteks ("kegiatan Terbatas di zona ini
    meliputi ...") sebagai klaim, yaitu temuan palsu.
    """
    status = (k.konteks.status or "").strip().upper()
    if status not in ("I", "T", "B", "X"):
        return False, True, ""
    teks = k.teks_narasi.lower()
    huruf = {(m.group(1) or m.group(2)).upper() for m in _RE_KLAIM_ITBX.finditer(teks)}
    if not huruf:
        return False, True, ""
    if huruf == {status}:
        return True, True, ""
    return True, False, f"fakta={status}, diklaim={sorted(huruf)}"


def ketepatan_arah(kasus: list[Kasus]) -> dict:
    p_n = p_ok = p_ambigu = 0
    m_n = m_ok = 0
    i_n = i_ok = 0
    contoh: list[dict] = []

    for k in kasus:
        if k.poin_id == "intensitas":
            n, ok, amb, cat = _arah_intensitas(k)
            p_n += n
            p_ok += ok
            p_ambigu += amb
            if cat and len(contoh) < 20:
                contoh.append({"permohonan": k.permohonan, "timestamp": k.timestamp,
                               "poin": "intensitas", "sub": "parameter", "temuan": cat,
                               "kutipan_narasi": k.keluaran.reasoning_panjang[:220]})
        elif k.poin_id == "dampak":
            bisa, ok, cat = _arah_mitigasi_dampak(k)
            if bisa:
                m_n += 1
                m_ok += int(ok)
                if not ok and len(contoh) < 20:
                    contoh.append({"permohonan": k.permohonan, "timestamp": k.timestamp,
                                   "poin": "dampak", "sub": "mitigasi", "temuan": [cat],
                                   "kutipan_narasi": k.keluaran.reasoning_panjang[:220]})
        elif k.poin_id == "itbx":
            bisa, ok, cat = _arah_kategori_itbx(k)
            if bisa:
                i_n += 1
                i_ok += int(ok)
                if not ok and len(contoh) < 20:
                    contoh.append({"permohonan": k.permohonan, "timestamp": k.timestamp,
                                   "poin": "itbx", "sub": "kategori", "temuan": [cat],
                                   "kutipan_narasi": k.keluaran.reasoning_panjang[:220]})

    n_total, ok_total = p_n + m_n + i_n, p_ok + m_ok + i_ok
    return {
        "gabungan": {"n": n_total, "kena": ok_total, "nilai": _proporsi(ok_total, n_total)},
        "parameter_intensitas": {"n": p_n, "kena": p_ok, "nilai": _proporsi(p_ok, p_n),
                                 "ambigu_dilewati": p_ambigu},
        "mitigasi_dampak": {"n": m_n, "kena": m_ok, "nilai": _proporsi(m_ok, m_n)},
        "kategori_itbx": {"n": i_n, "kena": i_ok, "nilai": _proporsi(i_ok, i_n)},
        "contoh_menyimpang": contoh,
    }


# ---------------------------------------------------------------------------
# 4. Boilerplate
# ---------------------------------------------------------------------------


def boilerplate(kasus: list[Kasus]) -> dict:
    """Kemiripan 5-gram ke tetangga terdekat, dipilah status sama vs berbeda.

    Dibandingkan HANYA antar permohonan berbeda: dua poin dari permohonan yang sama tak pernah jadi
    pasangan, karena kemiripan di sana tidak mengatakan apa pun soal kekhususan.
    """
    hasil_per_poin: dict[str, dict] = {}
    contoh: list[dict] = []

    for pid in _POIN_SEKARANG:
        anggota = [k for k in kasus if k.poin_id == pid]
        gram = [(k, _n_gram(_normalisasi(k.keluaran.reasoning_panjang))) for k in anggota]
        gram = [(k, g) for k, g in gram if g]
        if len(gram) < 2:
            hasil_per_poin[pid] = {"n": len(gram), "catatan": "terlalu sedikit utk dibandingkan"}
            continue

        nn_beda: list[float] = []
        nn_sama: list[float] = []
        puncak: tuple[float, Kasus, Kasus] | None = None
        for i, (ka, ga) in enumerate(gram):
            best_beda = best_sama = 0.0
            for j, (kb, gb) in enumerate(gram):
                if i == j or ka.permohonan == kb.permohonan:
                    continue
                s = _jaccard(ga, gb)
                if ka.konteks.status == kb.konteks.status:
                    best_sama = max(best_sama, s)
                else:
                    best_beda = max(best_beda, s)
                    if puncak is None or s > puncak[0]:
                        puncak = (s, ka, kb)
            nn_beda.append(best_beda)
            nn_sama.append(best_sama)

        atas_ambang = sum(1 for s in nn_beda if s >= _AMBANG_MIRIP)
        hasil_per_poin[pid] = {
            "n": len(gram),
            "status_berbeda": {
                "median": statistics.median(nn_beda),
                "p90": statistics.quantiles(nn_beda, n=10)[-1] if len(nn_beda) >= 10 else None,
                "maks": max(nn_beda),
                f"n_di_atas_{_AMBANG_MIRIP}": atas_ambang,
                "proporsi_di_atas_ambang": _proporsi(atas_ambang, len(nn_beda)),
            },
            "status_sama": {
                "median": statistics.median(nn_sama),
                "maks": max(nn_sama),
            },
        }
        if puncak and puncak[0] >= _AMBANG_MIRIP:
            s, ka, kb = puncak
            contoh.append({
                "poin": pid, "kemiripan": round(s, 3),
                "a": {"permohonan": ka.permohonan, "status": ka.konteks.status,
                      "kutipan": ka.keluaran.reasoning_panjang[:180]},
                "b": {"permohonan": kb.permohonan, "status": kb.konteks.status,
                      "kutipan": kb.keluaran.reasoning_panjang[:180]},
            })

    return {"ambang": _AMBANG_MIRIP, "n_gram": _N_GRAM,
            "per_poin": hasil_per_poin, "contoh_paling_mirip": contoh}


# ---------------------------------------------------------------------------
# Perakitan & pelaporan
# ---------------------------------------------------------------------------


def sebaran_jalur(kasus: list[Kasus]) -> dict:
    """Berapa bagian narasi yang benar-benar dihasilkan LLM, dan berapa yang jatuh ke template.

    Ini bukan metrik mutu LLM — ini metrik OPERASIONAL, dan penyebut bagi semua metrik lain di
    modul ini. Narasi template tak pernah menyebut partikular apa pun dan tak pernah membawa
    sitasi, jadi ia tak bisa dinilai mutu; yang bermakna adalah seberapa sering ia terpakai.
    """
    per_poin: dict[str, Counter] = {}
    for k in kasus:
        per_poin.setdefault(k.poin_id, Counter())[k.jalur] += 1
    total = Counter(k.jalur for k in kasus)
    n = len(kasus)
    return {
        "n": n,
        "total": dict(total),
        "proporsi_llm": _proporsi(total.get("llm", 0), n),
        "per_poin": {
            p: {"n": sum(c.values()), **{j: c.get(j, 0) for j in
                                         ("llm", "template_aman", "template_low_confidence")},
                "proporsi_llm": _proporsi(c.get("llm", 0), sum(c.values()))}
            for p, c in sorted(per_poin.items())
        },
    }


def hitung(kasus: list[Kasus], chunks: dict[str, Chunk]) -> dict:
    tstamp = sorted(k.timestamp for k in kasus if k.timestamp)
    # Keempat metrik mutu dihitung HANYA pada narasi jalur LLM — lihat `_jalur_narasi`.
    llm = [k for k in kasus if k.jalur == "llm"]
    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cakupan": {
            "rentang_tanggal_dinilai": [tstamp[0], tstamp[-1]] if tstamp else None,
            "narasi_dimuat": len(kasus),
            "narasi_jalur_llm": len(llm),
            "permohonan_dinilai": len({k.permohonan for k in kasus}),
            "narasi_per_poin": dict(Counter(k.poin_id for k in llm)),
            "status_per_poin": {
                p: dict(Counter(k.konteks.status for k in llm if k.poin_id == p))
                for p in _POIN_SEKARANG
            },
            "chunk_disitasi_tersedia": len(chunks),
        },
        "sebaran_jalur": sebaran_jalur(kasus),
        "faithfulness_numerik": faithfulness_numerik(llm, chunks),
        "kekhususan": kekhususan(llm),
        "ketepatan_arah": ketepatan_arah(llm),
        "boilerplate": boilerplate(llm),
    }


def _pct(e: dict | None) -> str:
    if not e:
        return "tak terukur"
    n = e.get("nilai")
    return "tak terukur" if n is None else f"{n:.1%}"


def cetak(h: dict, lewat: dict) -> None:
    c = h["cakupan"]
    print("\n=== CAKUPAN (baca ini sebelum angka mana pun) ===")
    rt = c.get("rentang_tanggal_dinilai")
    print(f"  rentang tanggal terhitung : {rt[0]} .. {rt[1]}" if rt
          else "  rentang tanggal           : (tak ada narasi terpakai)")
    print(f"  permohonan dinilai        : {c['permohonan_dinilai']}")
    print(f"  narasi dimuat             : {c['narasi_dimuat']}")
    print(f"  narasi jalur LLM (dinilai): {c['narasi_jalur_llm']}  {c['narasi_per_poin']}")
    print(f"  chunk disitasi dari DB    : {c['chunk_disitasi_tersedia']}")
    for k, v in sorted(lewat.items(), key=lambda t: -t[1]):
        print(f"  dibuang: {k:52s} {v}")

    j = h["sebaran_jalur"]
    print("\n=== 0. SEBARAN JALUR NARASI (penyebut bagi metrik di bawahnya) ===")
    print(f"  narasi dari LLM           {_pct({'nilai': j['proporsi_llm']}):>10s}   "
          f"(n={j['n']})  {j['total']}")
    for p, e in j["per_poin"].items():
        print(f"    {p:11s} {_pct({'nilai': e['proporsi_llm']}):>10s}  (llm={e['llm']}, "
              f"aman={e['template_aman']}, low_conf={e['template_low_confidence']})")

    f = h["faithfulness_numerik"]
    print("\n=== 1. FAITHFULNESS NUMERIK (replay cek provenance produksi) ===")
    if not f.get("terukur"):
        print(f"  TAK TERUKUR — {f.get('alasan')}")
    else:
        print(f"  per angka   {_pct(f['per_angka']):>10s}   (n={f['per_angka']['n']}, "
              f"kena={f['per_angka']['kena']})")
        print(f"  per narasi  {_pct(f['per_narasi']):>10s}   (n={f['per_narasi']['n']}, "
              f"kena={f['per_narasi']['kena']})  <- setara keputusan guardrail")
        for p, e in f["per_poin"].items():
            print(f"    {p:11s} angka {_pct(e['angka']):>8s} (n={e['angka']['n']:4d})   "
                  f"narasi {_pct(e['narasi']):>8s} (n={e['narasi']['n']:3d})")

    ks = h["kekhususan"]
    print("\n=== 2. KEKHUSUSAN (answer relevance: narasi membahas permohonan INI?) ===")
    print(f"  seluruh jangkar tersebut  {_pct(ks['semua_jangkar']):>10s}   "
          f"(n={ks['semua_jangkar']['n']}, kena={ks['semua_jangkar']['kena']})")
    for p, e in ks["per_poin"].items():
        extra = f"  [{e['tak_ada_jangkar']} tanpa jangkar]" if e["tak_ada_jangkar"] else ""
        print(f"    {p:11s} {_pct(e):>10s}  (n={e['n']}, kena={e['kena']}){extra}")
    print("  rincian per jangkar:")
    for nama, e in ks["per_jangkar"].items():
        print(f"    {nama:34s} {_pct(e):>10s}  (n={e['n']}, kena={e['kena']})")

    a = h["ketepatan_arah"]
    print("\n=== 3. KETEPATAN ARAH (narasi vs fakta deterministik back-end) ===")
    print(f"  gabungan                  {_pct(a['gabungan']):>10s}   "
          f"(n={a['gabungan']['n']}, kena={a['gabungan']['kena']})")
    pi = a["parameter_intensitas"]
    print(f"    parameter intensitas    {_pct(pi):>10s}  (n={pi['n']}, kena={pi['kena']}, "
          f"{pi['ambigu_dilewati']} ambigu dilewati)")
    print(f"    mitigasi dampak         {_pct(a['mitigasi_dampak']):>10s}  "
          f"(n={a['mitigasi_dampak']['n']}, kena={a['mitigasi_dampak']['kena']})")
    print(f"    kategori itbx           {_pct(a['kategori_itbx']):>10s}  "
          f"(n={a['kategori_itbx']['n']}, kena={a['kategori_itbx']['kena']})")
    if a["contoh_menyimpang"]:
        print(f"  !! {len(a['contoh_menyimpang'])} penyimpangan arah tercatat (lihat JSON keluaran); "
              "tiga teratas:")
        for e in a["contoh_menyimpang"][:3]:
            print(f"     {e['permohonan']} {e['poin']}/{e['sub']}: {e['temuan']}")

    b = h["boilerplate"]
    kunci_ambang = f"n_di_atas_{b['ambang']}"
    print(f"\n=== 4. BOILERPLATE (kemiripan {b['n_gram']}-gram ke tetangga terdekat) ===")
    for p, e in b["per_poin"].items():
        if "catatan" in e:
            print(f"  {p:11s} {e['catatan']} (n={e['n']})")
            continue
        sb, ss = e["status_berbeda"], e["status_sama"]
        p90 = f"{sb['p90']:.3f}" if sb["p90"] is not None else "n/a"
        print(f"  {p:11s} n={e['n']:3d}  status BEDA: median {sb['median']:.3f} "
              f"p90 {p90:>5s} maks {sb['maks']:.3f}  |  >={b['ambang']}: "
              f"{sb[kunci_ambang]}/{e['n']}")
        print(f"              status SAMA: median {ss['median']:.3f} maks {ss['maks']:.3f} "
              "(kemiripan di sini wajar)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Metrik sisi generasi (faithfulness numerik, kekhususan, ketepatan arah, "
                    "boilerplate) — offline dari log precheck + korpus, tanpa panggilan API.")
    ap.add_argument("--log", type=Path, default=_LOG)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--sejak", default=None,
                    help="hanya permohonan sejak tanggal ini (ISO, mis. 2026-09-07) — pakai ini "
                         "saat angkanya dikutip, supaya tak mencampur versi kode")
    ap.add_argument("--semua-jalan", action="store_true",
                    help="jangan dedup ke satu narasi terbaru per (permohonan, poin) — untuk "
                         "memeriksa sebaran antar-jalan, BUKAN untuk mengutip angka")
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"log tidak ada: {args.log}")

    kasus, lewat = muat_kasus(args.log, sejak=args.sejak, semua_jalan=args.semua_jalan)
    if not kasus:
        print("tak ada narasi yang bisa dinilai setelah penyaringan:")
        for k, v in sorted(lewat.items(), key=lambda t: -t[1]):
            print(f"  {k}: {v}")
        raise SystemExit(1)

    id_disitasi = {s.citation_id for k in kasus for s in k.keluaran.sitasi
                   if s.citation_id and not s.citation_id.startswith("anchor-")}
    chunks = _muat_chunks_disitasi(id_disitasi)

    hasil = hitung(kasus, chunks)
    hasil["berkas_log"] = args.log.name
    hasil["sejak"] = args.sejak
    hasil["dilewati"] = lewat

    cetak(hasil, lewat)
    args.out.write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[generasi] -> {args.out}")


if __name__ == "__main__":
    main()
