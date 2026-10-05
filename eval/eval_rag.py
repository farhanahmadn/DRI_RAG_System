"""eval/eval_rag.py — evaluasi kinerja RETRIEVAL RAG + laporan HTML mandiri (satu berkas).

Beda dari `eval/run_eval.py` (yang menilai sisi REASONING/LLM atas gold set dan masih terikat
MockRetriever): modul ini menilai sisi **retrieval** terhadap korpus Postgres SUNGGUHAN, memakai
`tests/eval_set.jsonl` (query -> chunk yang relevan) sebagai ground truth.

Yang dihitung:

1. **Metrik retrieval standar** — Hit-Rate@k, MRR, Recall@k, Precision@k, nDCG@k, MAP.
   Relevansi biner (sebuah chunk relevan atau tidak), sesuai bentuk ground truth yang tersedia.

2. **Ablasi per komponen** — pipeline produksi adalah dense + lexical -> RRF -> rerank. Tiap
   lapis dinilai sendiri supaya kontribusinya terlihat, bukan diasumsikan:
     - `dense`        : hanya pencarian vektor
     - `lexical`      : hanya Postgres FTS
     - `rrf`          : fusi keduanya, TANPA rerank
     - `dense+rerank` : dense lalu rerank — TANPA sisi lexical sama sekali
     - `rrf+rerank`   : pipeline produksi penuh
   `dense+rerank` ada khusus untuk menjawab satu pertanyaan yang tak bisa dijawab empat lainnya:
   apakah sisi lexical benar-benar membayar tempatnya, atau justru mengencerkan peringkat dense
   lewat RRF yang tak berbobot.
   Tanpa ablasi, mustahil tahu apakah reranker (yang berbayar per panggilan) benar-benar membayar
   dirinya sendiri, atau apakah sisi lexical benar-benar menyumbang.

3. **Latensi per konfigurasi** — biaya waktu tiap lapis.

4. **Sisi generasi (historis)** — sebaran `sebab` dari kunci `diagnostik` di `logs/precheck.jsonl`
   kalau ada. Ini data yang SUDAH terkumpul, bukan dihitung ulang di sini.

Keluaran: satu berkas HTML mandiri (grafik SVG inline, tanpa dependensi eksternal) + JSON mentah
di sebelahnya untuk reproduksi.

CLI:
  python -m eval.eval_rag                      # tulis eval/laporan_rag.html
  python -m eval.eval_rag --out /tmp/lap.html  # lokasi lain
  python -m eval.eval_rag --tanpa-rerank       # lewati konfigurasi ber-rerank (hemat kuota API)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.reasoning.generator import (  # noqa: E402
    _QUERY_FALLBACK_PER_POIN,
    _QUERY_INTENSITAS_TAJAM,
    _TANPA_LEXICAL_PER_POIN,
    _pilih_chunks_referensi,
)
from app.retrieval import db, fusion, rerank  # noqa: E402
from app.schemas import PoinKonteks  # noqa: E402
from eval.statistik import bootstrap_ci, wilcoxon_berpasangan  # noqa: E402
from app.retrieval.base import RetrievalFilters  # noqa: E402
from app.retrieval.embeddings import encode_dense_one  # noqa: E402
from app.retrieval.retriever import _expand  # noqa: E402

_EVAL_SET = Path(__file__).parent.parent / "tests" / "eval_set.jsonl"
_LOG_PRECHECK = Path(__file__).parent.parent / "logs" / "precheck.jsonl"
_K_LIST = (1, 3, 5, 10)
_TOP_N = 10          # panjang daftar hasil yang dinilai
_KANDIDAT = 30       # sejajar dgn candidate_k/rerank_pool retriever produksi

# TITIK OPERASI: app/reasoning/generator.py mengirim `top_k_dukungan=3` chunk ke LLM. Metrik pada
# kedalaman LAIN berguna untuk memahami perilaku, tapi hanya k=3 yang mewakili apa yang benar-benar
# diterima sistem — laporan menyorotnya secara terpisah supaya tak terbaca dari kedalaman yang salah.
_K_OPERASI = 3

# Cabang yang BENAR-BENAR dijalankan produksi, menurut awalan id topik (konvensi penamaan
# eval/bangun_eval_set_operasi.py). Ditaruh di sini, bukan di modul laporan, supaya hanya ada
# SATU daftar: eval/ringkasan_produksi.py mengimpornya, jadi tabel per-poin dan baris ringkas
# di tabel titik operasi mustahil menyimpang satu sama lain.
#
# Yang SENGAJA di luar daftar, beserta alasannya:
#   intensitas-kdb-*   — query lama; lengan "sebelum" dalam ablasi string query
#   itbx-kegiatan-*    — jalur search utk itbx; cadangan yang hampir tak pernah menyala,
#                        karena setiap payload back-end membawa dasar_hukum
#   dampak-query-produksi — kontrol tanpa filter zona; produksi selalu memakai filter
_CABANG_PRODUKSI = (
    ("intensitas-ambang-", "intensitas", "sub-zona presisi diketahui", "ada_subzona"),
    ("intensitas-tajam-keluarga-", "intensitas", "tanpa sub-zona (filter keluarga)",
     "tanpa_subzona"),
    ("dampak-zona-", "dampak", "sub-zona presisi diketahui", "ada_subzona"),
    ("dampak-keluarga-", "dampak", "tanpa sub-zona (filter keluarga)", "tanpa_subzona"),
)


def indeks_jalur_produksi(detail: list[dict]) -> list[int]:
    """Indeks topik yang mewakili jalur produksi (jalur `search` saja)."""
    awalan = tuple(a for a, _p, _c, _b in _CABANG_PRODUKSI)
    return [i for i, d in enumerate(detail) if str(d.get("query", "")).startswith(awalan)]
# Sepanjang daftar yang DINILAI, bukan sekadar sepanjang titik operasi. Dengan begitu pelabelan
# ulang (mis. saat kriteria relevansi berubah) bisa dihitung ulang dari bukti ini tanpa memanggil
# API retrieval lagi — menyimpan hanya 5 akan membuat metrik pada kedalaman >5 mustahil dihitung
# ulang, dan itu persis yang memaksa satu run penuh terbuang saat kriteria dampak direvisi.
_BUKTI_TERATAS = 10
# Metrik yang dipakai untuk klaim & uji signifikansi. Sengaja dibatasi: menguji SEMUA metrik x SEMUA
# pasangan konfigurasi menaikkan peluang temuan palsu tanpa menambah informasi.
_METRIK_KLAIM = (f"ndcg@{_K_OPERASI}", f"recall@{_K_OPERASI}", f"hit@{_K_OPERASI}", "mrr", "map")
# MRR dan MAP adalah nama AGREGAT: per-query yang ada hanyalah reciprocal rank (`rr`) dan average
# precision (`ap`) — rata-ratanyalah yang disebut MRR/MAP. Statistik bekerja atas skor per-topik,
# jadi namanya harus diterjemahkan dulu; menyamakannya begitu saja menghasilkan KeyError.
_KUNCI_PER_QUERY = {"mrr": "rr", "map": "ap"}


def _nilai_klaim(per_query_satu: dict, metrik: str) -> float:
    """Ambil skor satu topik untuk metrik klaim, menerjemahkan nama agregat ke kunci per-query."""
    return per_query_satu[_KUNCI_PER_QUERY.get(metrik, metrik)]


# ---------------------------------------------------------------------------
# Metrik — relevansi biner, sesuai bentuk ground truth (`relevan`: daftar id chunk)
# ---------------------------------------------------------------------------
# Backoff antar-percobaan SATU topik. Retry di dalam _provider_http hanya menutup gangguan
# beberapa detik; yang membunuh run adalah gangguan jaringan semenit-dua.
_JEDA_PULIH_S = (30.0, 60.0, 120.0)


def _coba_ulang(fn, nama: str):
    """Jalankan `fn`, ulangi dgn backoff bila provider/jaringan gagal.

    Alasannya empiris, bukan defensif-berjaga-jaga: run 140 topik mati di topik ke-80 karena satu
    kegagalan DNS sesaat (`getaddrinfo failed`) — ~25 menit panggilan API berbayar hangus karena
    gangguan beberapa detik.
    """
    for percobaan, jeda in enumerate((*_JEDA_PULIH_S, None), 1):
        try:
            return fn()
        except Exception as exc:
            if jeda is None:
                raise
            print(f"    ! {nama}: percobaan {percobaan} gagal ({type(exc).__name__}: {exc}); "
                  f"ulangi dalam {jeda:.0f}s", flush=True)
            time.sleep(jeda)


def _muat_checkpoint(nama: str, konfigurasi: list[str], id_topik: list[str]) -> dict | None:
    """Muat checkpoint hanya bila ia benar-benar milik run yang sama.

    Syaratnya ketat dengan sengaja: konfigurasi identik dan id topik yang sudah selesai harus
    sama persis dengan awalan eval set sekarang. Checkpoint dari eval set lain yang dipaksa
    masuk akan menghasilkan tabel yang campur aduk tanpa jejak apa pun — lebih baik menolak dan
    mulai dari awal daripada melaporkan angka yang tidak bisa ditelusuri.
    """
    berkas = Path(__file__).parent / nama
    if not berkas.exists():
        print(f"[eval] --lanjutkan diminta tapi {nama} tidak ada; mulai dari awal")
        return None
    try:
        cp = json.loads(berkas.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[eval] checkpoint tak terbaca ({exc}); mulai dari awal")
        return None

    selesai = [d["query"] for d in cp.get("detail_per_query", [])]
    if cp.get("konfigurasi") != konfigurasi:
        print("[eval] konfigurasi checkpoint beda; mulai dari awal")
        return None
    if id_topik[:len(selesai)] != selesai:
        print("[eval] urutan topik checkpoint tak cocok dgn eval set; mulai dari awal")
        return None
    if not all(len(cp.get("latensi", {}).get(k, [])) == len(selesai) for k in konfigurasi):
        print("[eval] latensi checkpoint tak sejajar dgn skor; mulai dari awal")
        return None
    print(f"[eval] lanjut dari checkpoint: {len(selesai)} topik sudah dinilai, dilewati")
    return cp


def _simpan_checkpoint(nama: str, muatan: dict) -> None:
    """Tulis skor mentah per-topik SEBELUM tahap statistik/HTML.

    Tahap retrieval membakar kuota API nyata — ~200 topik x 5 konfigurasi berjalan puluhan menit.
    Pernah terjadi: tahap statistik gagal setelah seluruh pemanggilan API selesai, dan semua skor
    hangus. Checkpoint ini memuat cukup data untuk menghitung ulang seluruh statistik tanpa
    menyentuh DB/API lagi. Kegagalan menulisnya sendiri tidak boleh menggagalkan evaluasi.
    """
    try:
        berkas = Path(__file__).parent / nama
        berkas.write_text(json.dumps(muatan, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] checkpoint skor mentah -> {berkas.name}")
    except Exception as exc:
        print(f"[eval] checkpoint gagal (dilanjutkan): {exc}")


def _metrik_satu_query(urutan: list[str], relevan: set[str]) -> dict:
    """Hitung seluruh metrik untuk SATU query dari daftar id terurut (peringkat 1 = indeks 0)."""
    hasil: dict = {}
    n_relevan = len(relevan)

    peringkat_pertama = next((i + 1 for i, cid in enumerate(urutan) if cid in relevan), None)
    hasil["peringkat_pertama"] = peringkat_pertama
    hasil["rr"] = 1.0 / peringkat_pertama if peringkat_pertama else 0.0

    for k in _K_LIST:
        atas = urutan[:k]
        kena = sum(1 for cid in atas if cid in relevan)
        hasil[f"hit@{k}"] = 1.0 if kena else 0.0
        hasil[f"recall@{k}"] = kena / n_relevan if n_relevan else 0.0
        hasil[f"precision@{k}"] = kena / k
        # nDCG biner: gain 1 utk chunk relevan, diskon logaritmik menurut posisi.
        dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(atas) if cid in relevan)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevan, k)))
        hasil[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    # Average Precision: rata-rata precision pada tiap posisi yang relevan, dibagi jumlah relevan
    # (dibatasi panjang daftar yang dinilai — AP jadi konservatif, tidak menghukum ganda chunk
    # relevan yang memang berada di luar top-N).
    kena = 0
    jumlah_p = 0.0
    for i, cid in enumerate(urutan):
        if cid in relevan:
            kena += 1
            jumlah_p += kena / (i + 1)
    hasil["ap"] = jumlah_p / min(n_relevan, len(urutan)) if n_relevan and urutan else 0.0
    return hasil


def _agregat(per_query: list[dict]) -> dict:
    """Rata-rata makro seluruh metrik (tiap query berbobot sama)."""
    if not per_query:
        return {}
    kunci = [k for k in per_query[0] if k != "peringkat_pertama"]
    agg = {k: statistics.fmean(q[k] for q in per_query) for k in kunci}
    agg["mrr"] = agg.pop("rr")
    agg["map"] = agg.pop("ap")
    ketemu = [q["peringkat_pertama"] for q in per_query if q["peringkat_pertama"]]
    agg["median_peringkat_pertama"] = statistics.median(ketemu) if ketemu else None
    agg["query_tanpa_hasil_relevan"] = sum(1 for q in per_query if q["peringkat_pertama"] is None)
    return agg


# ---------------------------------------------------------------------------
# Konfigurasi retrieval yang diuji (ablasi lapis demi lapis)
# ---------------------------------------------------------------------------
# Memo embedding per teks query. Eval set 161 topik hanya memakai 9 string query unik — yang
# bervariasi di produksi adalah FILTER zona, bukan teks querinya. Tanpa memo, harness membayar
# 161 panggilan embedding untuk 9 teks: 94% terbuang, dan itulah yang menghabiskan saldo provider
# di tengah run sebelumnya. Aman karena embedding adalah fungsi dari teksnya saja; filter bekerja
# di sisi SQL, sesudah vektornya ada.
_memo_embedding: dict[str, list[float]] = {}


def _embedding_query(teks: str) -> list[float]:
    if teks not in _memo_embedding:
        _memo_embedding[teks] = encode_dense_one(teks)
    return _memo_embedding[teks]


def _urutan_dense(conn, query: str, filters: RetrievalFilters) -> list[str]:
    """Hanya pencarian vektor. Query DIPERLUAS — persis seperti jalur produksi."""
    qvec = _embedding_query(_expand(query))
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider == "local":
        hasil = db.dense_search(conn, qvec, filters, _KANDIDAT)
    else:
        hasil = db.dense_search_ab(conn, qvec, provider, filters, _KANDIDAT)
    return [cid for cid, _ in hasil]


def _urutan_lexical(conn, query: str, filters: RetrievalFilters) -> list[str]:
    """Hanya Postgres FTS. Query ASLI (tanpa expansion) — persis seperti jalur produksi."""
    return [cid for cid, _ in db.fts_search(conn, query, filters, _KANDIDAT)]


def _urutan_rrf(dense: list[str], lexical: list[str]) -> list[str]:
    return [cid for cid, _ in fusion.reciprocal_rank_fusion([dense, lexical])]


def _urutan_rerank(conn, query: str, fused: list[str]) -> list[str]:
    """Rerank cross-encoder atas kandidat hasil fusi — lapis terakhir pipeline produksi."""
    if not fused:
        return []
    kandidat = db.hydrate(conn, fused[:_KANDIDAT])
    skor = rerank.rerank(_expand(query), [c.teks for c in kandidat], top_k=_TOP_N)
    return [kandidat[i].id for i, _ in skor]


def _muat_topik(path: Path) -> list[dict]:
    """Terima DUA skema: eval set lama (query/relevan) dan eval set titik operasi (ber-`jalur`,
    `filter`, `sumber_label`). Yang lama dinormalkan ke bentuk baru supaya sisa kode satu jalur."""
    topik = []
    for garis in path.read_text(encoding="utf-8").splitlines():
        if not garis.strip():
            continue
        row = json.loads(garis)
        row.setdefault("id", row["query"])
        row.setdefault("jalur", "search")
        row.setdefault("filter", {})
        row.setdefault("sumber_label", "seeded")
        row.setdefault("bobot_traffic", 0)
        topik.append(row)
    return topik


def jalankan_evaluasi(pakai_rerank: bool = True, jeda_s: float = 0.0,
                      eval_set: Path | None = None, lanjutkan: bool = False) -> dict:
    import psycopg

    berkas_eval = eval_set or _EVAL_SET
    semua_topik = _muat_topik(berkas_eval)
    baris = [x for x in semua_topik if x["jalur"] == "search"]
    topik_ref = [x for x in semua_topik if x["jalur"] == "reference"]
    wilayah = os.getenv("RETRIEVER_WILAYAH", "Sleman Tengah").strip('"')
    conn = psycopg.connect(os.getenv("DATABASE_URL"))

    konfigurasi = ["dense", "lexical", "rrf"] + (["dense+rerank", "rrf+rerank"] if pakai_rerank else [])
    per_query: dict[str, list[dict]] = {k: [] for k in konfigurasi}
    latensi: dict[str, list[float]] = {k: [] for k in konfigurasi}
    detail: list[dict] = []

    dilewati = 0
    if lanjutkan:
        cp = _muat_checkpoint("_checkpoint_search.json", konfigurasi, [x["id"] for x in baris])
        if cp:
            per_query = {k: list(cp["per_query"][k]) for k in konfigurasi}
            latensi = {k: list(cp["latensi"][k]) for k in konfigurasi}
            detail = list(cp["detail_per_query"])
            dilewati = len(detail)

    for i, row in enumerate(baris, 1):
        if i <= dilewati:
            continue
        query, relevan = row["query"], set(row["relevan"])
        # Filter per-topik: inilah sumbu yang benar-benar bervariasi di produksi (zona/sub-zona),
        # bukan teks querinya. Wilayah selalu disematkan, persis seperti _apply_default_wilayah.
        filters = RetrievalFilters(dokumen=wilayah, **(row.get("filter") or {}))

        def _seluruh_jalur(query=query, filters=filters):
            """Semua jalur utk SATU topik, sebagai satuan yang bisa diulang utuh.

            Latensi dikumpulkan lokal dan baru disalin setelah seluruh jalur berhasil: kalau topik
            diulang, waktu percobaan yang gagal tidak boleh masuk rata-rata, dan panjang daftar
            latensi harus tetap sejajar dgn daftar skor.
            """
            u: dict[str, list[str]] = {}
            lat: dict[str, float] = {}

            t0 = time.perf_counter()
            u["dense"] = _urutan_dense(conn, query, filters)
            lat["dense"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            u["lexical"] = _urutan_lexical(conn, query, filters)
            lat["lexical"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            u["rrf"] = _urutan_rrf(u["dense"], u["lexical"])
            lat["rrf"] = time.perf_counter() - t0

            if pakai_rerank:
                t0 = time.perf_counter()
                u["dense+rerank"] = _urutan_rerank(conn, query, u["dense"])
                lat["dense+rerank"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                u["rrf+rerank"] = _urutan_rerank(conn, query, u["rrf"])
                lat["rrf+rerank"] = time.perf_counter() - t0
            return u, lat

        urutan, lat_topik = _coba_ulang(_seluruh_jalur, row["id"])
        for kfg, nilai in lat_topik.items():
            latensi[kfg].append(nilai)

        baris_detail = {"query": row["id"], "n_relevan": len(relevan),
                        "sumber_label": row["sumber_label"],
                        # Bukti mentah: apa yang BENAR-BENAR terambil, bukan hanya skornya. Tanpa
                        # ini laporan hanya bisa mengklaim angka, dan pembacanya tak punya cara
                        # memeriksa klaim itu selain menjalankan ulang seluruh evaluasi.
                        "teks_query": row["query"], "filter": row.get("filter") or {},
                        "relevan": sorted(relevan),
                        "terambil": {k: urutan[k][:_BUKTI_TERATAS] for k in konfigurasi}}
        for kfg in konfigurasi:
            m = _metrik_satu_query(urutan[kfg][:_TOP_N], relevan)
            per_query[kfg].append(m)
            baris_detail[kfg] = m["peringkat_pertama"]
        detail.append(baris_detail)
        print(f"  [{i}/{len(baris)}] {row['id'][:46]:48s} " +
              "  ".join(f"{k}={baris_detail[k] or '-'}" for k in konfigurasi), flush=True)
        # Berkala, bukan hanya di akhir: gangguan jaringan yang lebih panjang daripada backoff
        # masih bisa membunuh run, dan skor yang sudah dibayar tidak boleh ikut mati.
        if i % 20 == 0:
            _simpan_checkpoint("_checkpoint_search.json",
                               {"tahap": "search", "wilayah": wilayah, "konfigurasi": konfigurasi,
                                "per_query": per_query, "latensi": latensi,
                                "detail_per_query": detail})
        if jeda_s and i < len(baris):
            time.sleep(jeda_s)

    _simpan_checkpoint("_checkpoint_search.json",
                       {"tahap": "search", "wilayah": wilayah, "konfigurasi": konfigurasi,
                        "per_query": per_query, "latensi": latensi,
                        "detail_per_query": detail})
    print(f"[eval] embedding: {len(_memo_embedding)} teks query unik utk {len(baris)} topik")

    produksi = "rrf+rerank" if "rrf+rerank" in konfigurasi else konfigurasi[-1]

    # Jalur rujukan dievaluasi terpisah: konfigurasinya beda (bukan dense/lexical/rrf) dan
    # inilah satu-satunya jalur yang dipakai poin itbx di produksi.
    hasil_anchor = None
    if topik_ref:
        from app.retrieval.retriever import RetrieverAsli

        print(f"\n[eval] jalur rujukan (anchor): {len(topik_ref)} topik")
        hasil_anchor = evaluasi_anchor(RetrieverAsli(default_wilayah=wilayah), topik_ref)
        for k in hasil_anchor["konfigurasi"]:
            a = hasil_anchor["agregat"][k]
            print(f"  {k:20s} nDCG@{_K_OPERASI}={a[f'ndcg@{_K_OPERASI}']:.3f}  "
                  f"Recall@{_K_OPERASI}={a[f'recall@{_K_OPERASI}']:.0%}")

    # --- konteks korpus ---
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE dokumen ILIKE %s", [f"%{wilayah}%"])
        n_chunk = cur.fetchone()[0]
        cur.execute("""SELECT level, count(*) FROM chunks WHERE dokumen ILIKE %s
                       GROUP BY 1 ORDER BY 1""", [f"%{wilayah}%"])
        per_level = dict(cur.fetchall())
        cur.execute("""SELECT count(*) FROM chunk_embeddings_ab a JOIN chunks c ON c.id = a.chunk_id
                       WHERE c.dokumen ILIKE %s""", [f"%{wilayah}%"])
        n_vektor = cur.fetchone()[0]
    conn.close()

    return {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wilayah": wilayah,
        "provider_embedding": os.getenv("EMBEDDING_PROVIDER", "local"),
        "provider_rerank": os.getenv("RERANK_PROVIDER", "local"),
        "n_query": len(baris),
        "top_n_dinilai": _TOP_N,
        "korpus": {"chunk": n_chunk, "per_level": per_level, "vektor": n_vektor},
        "konfigurasi": konfigurasi,
        "produksi": produksi,
        "agregat": {k: _agregat(per_query[k]) for k in konfigurasi},
        "statistik": _statistik(per_query, konfigurasi, produksi),
        "latensi_ms": {k: statistics.fmean(latensi[k]) * 1000 for k in konfigurasi},
        "detail_per_query": detail,
        # Skor mentah per-topik utk metrik klaim — supaya pihak lain bisa menghitung ulang CI/uji
        # sendiri tanpa menjalankan ulang retrieval (dan memverifikasi statistik kita).
        "skor_per_query": {k: [{m: _nilai_klaim(q, m) for m in _METRIK_KLAIM} for q in per_query[k]]
                           for k in konfigurasi},
        "metadata": _metadata_reproduksi(),
        "berkas_eval": berkas_eval.name,
        "sidik_eval_set": _sidik_eval_set(berkas_eval),
        "sumber_label": dict(Counter(x["sumber_label"] for x in semua_topik)),
        "anchor": hasil_anchor,
        "ablasi_query": ablasi_string_query(detail, per_query, konfigurasi),
        "generasi": _ringkas_diagnostik_log(),
    }


# --- Jalur rujukan (anchor) — menangani 100% poin itbx di produksi ---------------------------
_KFG_ANCHOR = ("anchor-urutan-db", "anchor-sadar-zona")
_LABEL_ANCHOR = {
    "anchor-urutan-db": "Anchor menurut urutan DB (sebelum perbaikan)",
    "anchor-sadar-zona": "Anchor sadar-zona (produksi)",
}


def _poin_tiruan(zona: str | None, subzona: str | None) -> PoinKonteks:
    """PoinKonteks minimal — hanya zona yang dibaca `_pilih_chunks_referensi`."""
    return PoinKonteks(
        poin_id="itbx", kategori="Klasifikasi Kegiatan (ITBX)", tipe_rekomendasi="kategorikal",
        status="I", fakta={"lolos": True, "reason": "x"}, zona=zona, zona_subzone=subzona,
    )


def evaluasi_anchor(rt, topik: list[dict]) -> dict:
    """Ablasi pemilihan sitasi sadar-zona pada jalur `get_by_reference`.

    Jalur ini menangani SELURUH poin itbx di produksi (349/349 payload BE membawa `dasar_hukum`),
    tapi tak pernah dievaluasi sebelumnya. Dua konfigurasi dibandingkan atas topik yang sama:
    urutan DB apa adanya (perilaku sebelum commit `d1a194c`) vs pemilihan sadar-zona (produksi).
    Bedanya bukan akademis — di payload nyata APP-2026-2428, urutan DB menyodorkan Lampiran V.B
    Cagar Alam untuk pemohon Zona Pertanian.
    """
    per_query: dict[str, list[dict]] = {k: [] for k in _KFG_ANCHOR}
    latensi: dict[str, list[float]] = {k: [] for k in _KFG_ANCHOR}
    detail: list[dict] = []

    for row in topik:
        relevan = set(row["relevan"])
        filt = row.get("filter") or {}
        poin = _poin_tiruan(
            zona=filt.get("zona_induk") or _zona_induk_dari_prefix(filt.get("zona_prefix")),
            subzona=filt.get("zona"),
        )
        t0 = time.perf_counter()
        kandidat = rt.get_by_reference([row["query"]])
        biaya_ambil = time.perf_counter() - t0

        urutan = {
            # Perilaku lama: potong apa adanya menurut urutan yang dikembalikan DB.
            "anchor-urutan-db": [c.id for c in kandidat[:_TOP_N]],
            "anchor-sadar-zona": [c.id for c in _pilih_chunks_referensi(kandidat, poin, _TOP_N)],
        }
        baris = {"query": row.get("id", row["query"]), "n_relevan": len(relevan),
                 "teks_query": row["query"], "filter": row.get("filter") or {},
                 "relevan": sorted(relevan),
                 "terambil": {k: v[:_BUKTI_TERATAS] for k, v in urutan.items()}}
        for kfg in _KFG_ANCHOR:
            m = _metrik_satu_query(urutan[kfg], relevan)
            per_query[kfg].append(m)
            latensi[kfg].append(biaya_ambil)
            baris[kfg] = m["peringkat_pertama"]
        detail.append(baris)

    _simpan_checkpoint("_checkpoint_anchor.json",
                       {"tahap": "anchor", "konfigurasi": list(_KFG_ANCHOR),
                        "per_query": per_query, "detail_per_query": detail})

    return {
        "konfigurasi": list(_KFG_ANCHOR),
        "produksi": "anchor-sadar-zona",
        "n_query": len(topik),
        "agregat": {k: _agregat(per_query[k]) for k in _KFG_ANCHOR},
        "statistik": _statistik(per_query, list(_KFG_ANCHOR), "anchor-sadar-zona"),
        "latensi_ms": {k: statistics.fmean(latensi[k]) * 1000 for k in _KFG_ANCHOR},
        "detail_per_query": detail,
        "skor_per_query": {k: [{m: _nilai_klaim(q, m) for m in _METRIK_KLAIM} for q in per_query[k]]
                           for k in _KFG_ANCHOR},
    }


def _zona_induk_dari_prefix(prefix: str | None) -> str | None:
    """Balik dari kode keluarga ('R') ke nama zona induk ('Zona Perumahan') — `_pilih_chunks_referensi`
    menerima NAMA zona induk, bukan kodenya."""
    if not prefix:
        return None
    from app.reasoning.generator import _ZONA_KODE_PREFIX

    return next((nama.title() for nama, kode in _ZONA_KODE_PREFIX.items() if kode == prefix), None)


def _statistik(per_query: dict[str, list[dict]], konfigurasi: list[str], produksi: str) -> dict:
    """Selang kepercayaan tiap konfigurasi + uji berpasangan TERHADAP konfigurasi produksi.

    Pembanding sengaja satu (produksi), bukan semua-lawan-semua: pertanyaan yang relevan adalah
    "apakah ada yang berbeda nyata dari yang kita jalankan sekarang", dan membatasi jumlah uji
    menekan peluang temuan palsu.
    """
    ci: dict[str, dict[str, dict]] = {}
    for kfg in konfigurasi:
        ci[kfg] = {}
        for metrik in _METRIK_KLAIM:
            s = bootstrap_ci([_nilai_klaim(q, metrik) for q in per_query[kfg]])
            ci[kfg][metrik] = {"rata": s.rata, "bawah": s.bawah, "atas": s.atas, "n": s.n}

    uji: dict[str, dict[str, dict]] = {}
    for kfg in konfigurasi:
        if kfg == produksi:
            continue
        uji[kfg] = {}
        for metrik in _METRIK_KLAIM:
            u = wilcoxon_berpasangan([_nilai_klaim(q, metrik) for q in per_query[kfg]],
                                     [_nilai_klaim(q, metrik) for q in per_query[produksi]])
            uji[kfg][metrik] = {"p": u.p, "n_beda": u.n_beda, "efek": u.efek,
                                "selisih_rata": u.selisih_rata, "signifikan": u.signifikan}
    return {"ci": ci, "uji_vs_produksi": uji, "pembanding": produksi}


_PRA_TAJAM = "intensitas-tajam-keluarga-"
_PRA_KDB = "intensitas-kdb-keluarga-"


def ablasi_string_query(detail: list[dict], per_query: dict[str, list[dict]],
                        konfigurasi: list[str]) -> dict | None:
    """Ablasi STRING QUERY pada cabang tanpa sub-zona presisi (mayoritas traffic).

    Dihitung sebagai TURUNAN dari skor per-topik yang sudah dinilai di run yang sama — bukan
    retrieval ulang — sehingga tidak ada panggilan API tambahan dan tidak ada celah di mana kedua
    lengan bisa dinilai dengan indeks atau versi kode yang berbeda.

    Pasangannya sempurna: `intensitas-tajam-keluarga-<K>` dan `intensitas-kdb-keluarga-<K>`
    memakai filter yang IDENTIK dan label yang IDENTIK; satu-satunya yang berbeda adalah teks
    query. Karena itu selisihnya tidak bisa dijelaskan oleh apa pun selain string query.
    """
    tajam = {d["query"][len(_PRA_TAJAM):]: i for i, d in enumerate(detail)
             if d["query"].startswith(_PRA_TAJAM)}
    kdb = {d["query"][len(_PRA_KDB):]: i for i, d in enumerate(detail)
           if d["query"].startswith(_PRA_KDB)}
    keluarga = sorted(set(tajam) & set(kdb))
    if not keluarga:
        return None       # eval set tanpa topik kontrafaktual — jangan karang bagiannya

    hasil: dict[str, dict] = {}
    for kfg in konfigurasi:
        lengan: dict[str, dict] = {}
        for nama, peta in (("tajam", tajam), ("kdb", kdb)):
            lengan[nama] = {}
            for metrik in _METRIK_KLAIM:
                c = bootstrap_ci([_nilai_klaim(per_query[kfg][peta[k]], metrik)
                                  for k in keluarga])
                lengan[nama][metrik] = {"rata": c.rata, "bawah": c.bawah, "atas": c.atas,
                                        "n": c.n}
            lengan[nama]["peringkat1"] = sum(
                1 for k in keluarga if detail[peta[k]].get(kfg) == 1)
        uji: dict[str, dict] = {}
        for metrik in _METRIK_KLAIM:
            u = wilcoxon_berpasangan(
                [_nilai_klaim(per_query[kfg][tajam[k]], metrik) for k in keluarga],
                [_nilai_klaim(per_query[kfg][kdb[k]], metrik) for k in keluarga])
            uji[metrik] = {"p": u.p, "n_beda": u.n_beda, "efek": u.efek,
                           "selisih_rata": u.selisih_rata, "signifikan": u.signifikan}
        hasil[kfg] = {"lengan": lengan, "uji": uji}
    return {"n_keluarga": len(keluarga), "keluarga": keluarga, "per_konfigurasi": hasil,
            "query_tajam": _QUERY_INTENSITAS_TAJAM,
            "query_kdb": _QUERY_FALLBACK_PER_POIN["intensitas"]}


def _sidik_eval_set(berkas: Path) -> str | None:
    """Sidik jari isi eval set, direkam bersama hasilnya.

    Label bisa berubah tanpa mengubah nama berkas — dan saat itu terjadi, berkas hasil lama
    tetap terlihat sah padahal skornya dinilai atas label yang sudah tidak berlaku. Pernah
    terjadi: kriteria relevansi `dampak` direvisi, eval set diperbarui, tapi hasil lama masih
    di tempatnya. Sidik jari ini membuat ketidakcocokan itu terdeteksi, bukan terlewat.
    """
    try:
        import hashlib

        return hashlib.sha256(berkas.read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def _metadata_reproduksi() -> dict:
    """Segala yang dibutuhkan pihak lain untuk memproduksi ulang angka ini."""
    import subprocess

    def _git(*arg: str) -> str | None:
        try:
            return subprocess.run(["git", *arg], capture_output=True, text=True,
                                  timeout=10, cwd=Path(__file__).parent.parent).stdout.strip() or None
        except Exception:
            return None

    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "commit_kotor": bool(_git("status", "--porcelain")),
        "model_embedding": os.getenv("JINA_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL"),
        "model_rerank": os.getenv("JINA_RERANK_MODEL") or os.getenv("RERANKER_MODEL"),
        "model_llm": os.getenv("LLM_MODEL"),
        "kandidat_k": _KANDIDAT,
        "top_n_dinilai": _TOP_N,
        "k_operasi": _K_OPERASI,
    }


def _ringkas_diagnostik_log() -> dict:
    """Sebaran `sebab` dari log operasional — data historis yang SUDAH terkumpul, bukan uji baru."""
    if not _LOG_PRECHECK.exists():
        return {}
    sebab: dict[str, int] = {}
    n_rec = n_poin = sitasi = sitasi_ver = 0
    for l in _LOG_PRECHECK.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            rec = json.loads(l)
        except Exception:
            continue
        if "diagnostik" not in rec:
            continue
        n_rec += 1
        for d in rec["diagnostik"]:
            n_poin += 1
            sebab[d.get("sebab", "?")] = sebab.get(d.get("sebab", "?"), 0) + 1
        for p in rec.get("response", {}).get("poin", []):
            for s in p.get("sitasi", []) or []:
                sitasi += 1
                sitasi_ver += int(bool(s.get("terverifikasi")))
    return {"n_permohonan": n_rec, "n_poin": n_poin, "sebab": sebab,
            "sitasi": sitasi, "sitasi_terverifikasi": sitasi_ver}


# ---------------------------------------------------------------------------
# Laporan HTML mandiri — SVG inline, tanpa dependensi eksternal
# ---------------------------------------------------------------------------
_WARNA = {"dense": "#2563eb", "lexical": "#d97706", "rrf": "#7c3aed",
          "dense+rerank": "#0891b2", "rrf+rerank": "#059669"}
_LABEL = {"dense": "Dense saja", "lexical": "Lexical saja", "rrf": "RRF (tanpa rerank)",
          "dense+rerank": "Dense + rerank (tanpa lexical)", "rrf+rerank": "RRF + rerank (produksi)"}


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _bar_berkelompok(judul: str, kategori: list[str], seri: dict[str, list[float]],
                     maks: float = 1.0, fmt: str = "{:.0%}") -> str:
    """Bar chart berkelompok, SVG murni."""
    W, H = 760, 300
    kiri, bawah, atas = 56, 54, 34
    lebar_grup = (W - kiri - 16) / max(len(kategori), 1)
    lebar_bar = lebar_grup / (len(seri) + 0.6)
    tinggi = H - atas - bawah
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="{kiri}" y="20" class="jdl">{_esc(judul)}</text>']
    for g in range(5):
        y = atas + tinggi * g / 4
        p.append(f'<line x1="{kiri}" y1="{y:.1f}" x2="{W-16}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{kiri-8}" y="{y+4:.1f}" class="ax" text-anchor="end">'
                 f'{fmt.format(maks*(1-g/4))}</text>')
    for gi, kat in enumerate(kategori):
        x0 = kiri + gi * lebar_grup
        for si, (nama, nilai) in enumerate(seri.items()):
            v = max(0.0, min(nilai[gi] / maks if maks else 0, 1.0))
            h = tinggi * v
            x = x0 + 8 + si * lebar_bar
            y = atas + tinggi - h
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{lebar_bar-3:.1f}" height="{h:.1f}" '
                     f'rx="2" fill="{_WARNA.get(nama, "#64748b")}"><title>{_esc(nama)} — '
                     f'{_esc(kat)}: {fmt.format(nilai[gi])}</title></rect>')
            if h > 22:
                p.append(f'<text x="{x+(lebar_bar-3)/2:.1f}" y="{y-4:.1f}" class="val" '
                         f'text-anchor="middle">{fmt.format(nilai[gi])}</text>')
        p.append(f'<text x="{x0+lebar_grup/2:.1f}" y="{atas+tinggi+18}" class="ax" '
                 f'text-anchor="middle">{_esc(kat)}</text>')
    lx = kiri
    for nama in seri:
        p.append(f'<rect x="{lx}" y="{H-22}" width="10" height="10" rx="2" fill="{_WARNA.get(nama,"#64748b")}"/>')
        p.append(f'<text x="{lx+15}" y="{H-13}" class="lg">{_esc(_LABEL.get(nama, nama))}</text>')
        lx += 20 + 7.2 * len(_LABEL.get(nama, nama))
    p.append("</svg>")
    return "".join(p)


def _bar_horizontal(judul: str, label: list[str], nilai: list[float], warna: list[str],
                    satuan: str = "") -> str:
    W = 760
    tinggi_baris = 30
    H = 40 + tinggi_baris * len(label) + 10
    kiri = 190
    maks = max(nilai) if nilai and max(nilai) > 0 else 1.0
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="16" y="20" class="jdl">{_esc(judul)}</text>']
    for i, (lb, v) in enumerate(zip(label, nilai)):
        y = 38 + i * tinggi_baris
        w = (W - kiri - 90) * (v / maks)
        p.append(f'<text x="{kiri-10}" y="{y+13}" class="ax" text-anchor="end">{_esc(lb)}</text>')
        p.append(f'<rect x="{kiri}" y="{y}" width="{w:.1f}" height="18" rx="3" fill="{warna[i]}"/>')
        p.append(f'<text x="{kiri+w+8:.1f}" y="{y+13}" class="val">{v:,.0f}{_esc(satuan)}</text>')
    p.append("</svg>")
    return "".join(p)


def _strip_peringkat(detail: list[dict], konfigurasi: list[str]) -> str:
    """Peringkat chunk relevan pertama per query — memperlihatkan query mana yang sulit."""
    W = 760
    tinggi_baris = 22
    kiri = 250
    H = 46 + tinggi_baris * len(detail) + 16
    kotak = 26
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peringkat per query">',
         '<text x="16" y="20" class="jdl">Peringkat chunk relevan pertama, per query '
         '(makin kecil makin baik)</text>']
    for ci, kfg in enumerate(konfigurasi):
        x = kiri + ci * (kotak + 30)
        singkat = {"dense+rerank": "d+rr", "rrf+rerank": "rrf+rr"}.get(kfg, kfg)
        p.append(f'<text x="{x+kotak/2}" y="38" class="ax" text-anchor="middle">'
                 f'{_esc(singkat)}</text>')
    for i, d in enumerate(detail):
        y = 46 + i * tinggi_baris
        p.append(f'<text x="{kiri-12}" y="{y+14}" class="ax" text-anchor="end">'
                 f'{_esc(d["query"][:34])}</text>')
        for ci, kfg in enumerate(konfigurasi):
            r = d.get(kfg)
            x = kiri + ci * (kotak + 30)
            if r is None:
                isi, teks, warna_t = "#fee2e2", "—", "#b91c1c"
            else:
                tingkat = min(max((r - 1) / 9.0, 0.0), 1.0)
                isi = f"hsl(152 55% {int(42 + tingkat * 44)}%)"
                teks = str(r)
                warna_t = "#052e1a" if tingkat < 0.5 else "#334155"
            p.append(f'<rect x="{x}" y="{y}" width="{kotak}" height="18" rx="3" fill="{isi}"/>')
            p.append(f'<text x="{x+kotak/2}" y="{y+13}" class="cel" text-anchor="middle" '
                     f'fill="{warna_t}">{teks}</text>')
    p.append("</svg>")
    return "".join(p)


def _donat(judul: str, data: dict[str, int], palet: dict[str, str]) -> str:
    total = sum(data.values()) or 1
    W, H, cx, cy, r, tebal = 760, 220, 120, 118, 74, 30
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(judul)}">',
         f'<text x="16" y="20" class="jdl">{_esc(judul)}</text>']
    sudut = -math.pi / 2
    for nama, n in sorted(data.items(), key=lambda kv: -kv[1]):
        porsi = n / total
        akhir = sudut + porsi * 2 * math.pi
        x1, y1 = cx + r * math.cos(sudut), cy + r * math.sin(sudut)
        x2, y2 = cx + r * math.cos(akhir), cy + r * math.sin(akhir)
        besar = 1 if porsi > 0.5 else 0
        p.append(f'<path d="M {x1:.2f} {y1:.2f} A {r} {r} 0 {besar} 1 {x2:.2f} {y2:.2f}" '
                 f'fill="none" stroke="{palet.get(nama, "#94a3b8")}" stroke-width="{tebal}">'
                 f'<title>{_esc(nama)}: {n} ({porsi:.0%})</title></path>')
        sudut = akhir
    p.append(f'<text x="{cx}" y="{cy+2}" class="ctr" text-anchor="middle">{total}</text>')
    p.append(f'<text x="{cx}" y="{cy+20}" class="ax" text-anchor="middle">poin</text>')
    ly = 54
    for nama, n in sorted(data.items(), key=lambda kv: -kv[1]):
        p.append(f'<rect x="250" y="{ly}" width="11" height="11" rx="2" fill="{palet.get(nama,"#94a3b8")}"/>')
        p.append(f'<text x="268" y="{ly+10}" class="lg">{_esc(nama)} — {n} ({n/total:.0%})</text>')
        ly += 22
    p.append("</svg>")
    return "".join(p)


def _fmt_ci(e: dict, persen: bool = False) -> str:
    """'0.821 [0.74-0.89]' — angka telanjang tanpa selang tidak boleh muncul di laporan ini."""
    if persen:
        return (f'{e["rata"]:.0%} <span class="ci">[{e["bawah"]:.0%}–{e["atas"]:.0%}]</span>')
    return (f'{e["rata"]:.3f} <span class="ci">[{e["bawah"]:.3f}–{e["atas"]:.3f}]</span>')


def _fmt_uji(e: dict | None) -> str:
    """Verdict perbandingan terhadap konfigurasi produksi, bukan sekadar selisih angka."""
    if e is None:
        return '<span class="net">— (pembanding)</span>'
    if e["n_beda"] == 0:
        return '<span class="net">identik</span>'
    if not e["signifikan"]:
        return f'<span class="net">setara</span> <span class="ci">(p={e["p"]:.2f})</span>'
    arah = "lebih baik" if e["selisih_rata"] > 0 else "lebih buruk"
    warna = "baik" if e["selisih_rata"] > 0 else "buruk"
    return (f'<span class="{warna}">{arah} {e["selisih_rata"]:+.3f}</span> '
            f'<span class="ci">(p={e["p"]:.3f})</span>')


def muat_ringkasan_produksi(berkas: Path | None = None) -> dict | None:
    """Baca hasil eval/ringkasan_produksi.py kalau sudah dijalankan."""
    berkas = berkas or (Path(__file__).parent / "ringkasan_produksi.json")
    if not berkas.exists():
        return None
    try:
        return json.loads(berkas.read_text(encoding="utf-8"))
    except Exception:
        return None


def muat_baseline(berkas: Path | None = None) -> dict | None:
    """Baca lantai acak & langit-langit dari eval/baseline_acak.py kalau sudah dijalankan."""
    berkas = berkas or (Path(__file__).parent / "baseline.json")
    if not berkas.exists():
        return None
    try:
        return json.loads(berkas.read_text(encoding="utf-8"))
    except Exception:
        return None


def muat_atribusi(berkas: Path | None = None) -> dict | None:
    """Baca hasil eval/metrik_atribusi.py kalau sudah pernah dijalankan.

    Terpisah dari evaluasi retrieval dengan sengaja: metrik atribusi dihitung dari log keluaran
    nyata dan tidak memanggil DB/API retrieval, jadi keduanya punya siklus hidup berbeda dan
    tidak boleh saling memaksa dijalankan ulang. Kalau belum ada, bagiannya TIDAK dicetak —
    laporan tanpa bagian itu lebih jujur daripada bagian berisi angka basi.
    """
    berkas = berkas or (Path(__file__).parent / "atribusi.json")
    if not berkas.exists():
        return None
    try:
        return json.loads(berkas.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_atribusi(e: dict) -> str:
    """Penyebut nol -> "tak terukur", bukan 0%. 0% terbaca sebagai gagal total."""
    nilai = e.get("nilai")
    if nilai is None:
        return '<span class="net">tak terukur</span>'
    inti = e.get("kena", e.get("sepakat"))
    return f'<b>{nilai:.1%}</b> <span class="net">(n={e["n"]}, {inti})</span>'


def _konfigurasi_produksi_per_poin() -> dict[str, str]:
    """Poin -> baris tabel mana yang mewakili jalur produksinya.

    Diturunkan dari `generator._TANPA_LEXICAL_PER_POIN`, bukan ditulis ulang di sini: sejak fusi
    jadi per-poin, "produksi" BUKAN satu baris tabel lagi, dan laporan yang menyorot satu baris
    sebagai produksi akan menyesatkan pembacanya. Kalau konfigurasi di generator berubah, tabel
    ini ikut berubah dengan sendirinya.
    """
    peta = {}
    for poin in ("itbx", "intensitas", "dampak"):
        if poin == "itbx":
            # Produksi selalu lewat get_by_reference (anchor dari back-end); jalur search hanya
            # cadangan kalau BE tak mengirim dasar_hukum. Lihat bagian ablasi jalur rujukan.
            peta[poin] = "get_by_reference (lihat bagian ablasi jalur rujukan)"
        elif poin in _TANPA_LEXICAL_PER_POIN:
            peta[poin] = "dense+rerank"
        else:
            peta[poin] = "rrf+rerank"
    return peta


def bangun_html(r: dict) -> str:
    kfg = r["konfigurasi"]
    agg = r["agregat"]
    # Pembanding statistik = konfigurasi yang dipakai saat angka ini DIUKUR (tersimpan di hasil),
    # bukan konfigurasi produksi hari ini. Keduanya bisa beda setelah perubahan, dan menamainya
    # "produksi" tanpa syarat adalah cara tercepat membuat tabel ini berbohong.
    produksi = r.get("produksi") or ("rrf+rerank" if "rrf+rerank" in kfg else "rrf")
    peta_produksi = _konfigurasi_produksi_per_poin()
    baris_peta = "".join(
        f"<tr><td><b>{_esc(poin)}</b></td><td>{_esc(cfg)}</td></tr>"
        for poin, cfg in peta_produksi.items())

    hit = {k: [agg[k][f"hit@{n}"] for n in _K_LIST] for k in kfg}
    mutu_kat = ["MRR", "MAP", "nDCG@5", "Recall@5"]
    mutu = {k: [agg[k]["mrr"], agg[k]["map"], agg[k]["ndcg@5"], agg[k]["recall@5"]] for k in kfg}

    baris_tabel = ""
    for k in kfg:
        a = agg[k]
        sorot = ' class="sorot"' if k == produksi else ""
        baris_tabel += (
            f"<tr{sorot}><td><b>{_esc(_LABEL.get(k,k))}</b></td>"
            + "".join(f"<td>{a[f'hit@{n}']:.0%}</td>" for n in _K_LIST)
            + f"<td>{a['mrr']:.3f}</td><td>{a['map']:.3f}</td>"
            f"<td>{a['ndcg@5']:.3f}</td><td>{a['recall@5']:.0%}</td>"
            f"<td>{a['precision@5']:.0%}</td>"
            f"<td>{a['query_tanpa_hasil_relevan']}</td>"
            f"<td>{r['latensi_ms'][k]:,.0f} ms</td></tr>")

    stat = r.get("statistik") or {}
    ci_all, uji_all = stat.get("ci", {}), stat.get("uji_vs_produksi", {})

    # Baris ringkas jalur produksi. Ditaruh DI DALAM tabel yang sama, bukan menunggu pembaca
    # menggulir ke bagian 2: keterangan di bawah tabel sudah menyatakan "bukan angka produksi",
    # dan ternyata itu tidak cukup — pembaca tetap membaca rata-rata 210 topik sebagai kinerja
    # sistem. Kalau keterangan saja tak cukup, yang kurang desain tabelnya.
    baris_produksi_ringkas = ""
    try:
        from app.reasoning.generator import _TANPA_LEXICAL_PER_POIN

        idx_prod = indeks_jalur_produksi(r["detail_per_query"])
        skor = r.get("skor_per_query") or {}
        if idx_prod and skor:
            nilai: dict[str, list[float]] = {}
            for i in idx_prod:
                poin = str(r["detail_per_query"][i]["query"]).split("-")[0]
                # Konfigurasi yang dipakai poin ini di produksi, diturunkan dari kode.
                k = "dense+rerank" if poin in _TANPA_LEXICAL_PER_POIN else "rrf+rerank"
                if k not in skor:
                    continue
                for m in _METRIK_KLAIM:
                    nilai.setdefault(m, []).append(skor[k][i][m])
            if nilai:
                sel = ""
                for m in (f"ndcg@{_K_OPERASI}", f"recall@{_K_OPERASI}",
                          f"hit@{_K_OPERASI}", "mrr"):
                    c = bootstrap_ci(nilai[m])
                    persen = m.startswith(("recall", "hit"))
                    sel += ("<td>" + _fmt_ci({"rata": c.rata, "bawah": c.bawah,
                                              "atas": c.atas, "n": c.n}, persen=persen)
                            + "</td>")
                baris_produksi_ringkas = (
                    f'<tr class="sorot"><td><b>Jalur produksi saja</b>'
                    f'<span class="ci">{len(idx_prod)} dari {r["n_query"]} topik, '
                    f'konfigurasi per-poin</span></td>{sel}'
                    f'<td class="net">inilah yang dijalankan sistem</td></tr>')
    except Exception:
        baris_produksi_ringkas = ""

    baris_operasi = ""
    for k in kfg:
        c = ci_all.get(k, {})
        sorot = ' class="sorot"' if k == produksi else ""
        if not c:      # hasil lama tanpa statistik — jangan mengarang selang
            a = agg[k]
            baris_operasi += (
                f'<tr{sorot}><td><b>{_esc(_LABEL.get(k,k))}</b></td>'
                f'<td>{a[f"ndcg@{_K_OPERASI}"]:.3f}</td><td>{a[f"recall@{_K_OPERASI}"]:.0%}</td>'
                f'<td>{a[f"hit@{_K_OPERASI}"]:.0%}</td><td>{a["mrr"]:.3f}</td>'
                f'<td class="net">(tanpa statistik)</td></tr>')
            continue
        baris_operasi += (
            f'<tr{sorot}><td><b>{_esc(_LABEL.get(k,k))}</b></td>'
            f'<td>{_fmt_ci(c[f"ndcg@{_K_OPERASI}"])}</td>'
            f'<td>{_fmt_ci(c[f"recall@{_K_OPERASI}"], persen=True)}</td>'
            f'<td>{_fmt_ci(c[f"hit@{_K_OPERASI}"], persen=True)}</td>'
            f'<td>{_fmt_ci(c["mrr"])}</td>'
            f'<td>{_fmt_uji(uji_all.get(k, {}).get(f"ndcg@{_K_OPERASI}"))}</td></tr>')

    # Bagian 1 & 4-6 selalu ada; bagian 2 bergantung ringkasan_produksi.json dan bagian 3
    # bergantung baseline.json. Sisanya bergantung isi
    # hasil. Nomornya diturunkan dari data,
    # bukan ditanam, supaya laporan dari eval set tanpa jalur rujukan tidak melompati nomor.
    _ada_anchor = bool((r.get("anchor") or {}).get("n_query"))
    _ada_abq = bool((r.get("ablasi_query") or {}).get("n_keluarga"))
    _ada_gen = bool((r.get("generasi") or {}).get("n_poin"))
    _ada_atr = bool(r.get("atribusi") or muat_atribusi())
    _nomor_anchor = 7
    _nomor_abq = _nomor_anchor + (1 if _ada_anchor else 0)
    _nomor_atr = _nomor_abq + (1 if _ada_abq else 0)
    _nomor_gen = _nomor_atr + (1 if _ada_atr else 0)
    _nomor_batas = _nomor_gen + (1 if _ada_gen else 0)

    # --- Ringkasan kinerja per-poin pada jalur produksi -----------------------------------
    rp = r.get("ringkasan_produksi") or muat_ringkasan_produksi()
    blok_produksi = ""
    if rp and rp.get("baris"):
        bt = rp["bobot_trafik"]
        baris_rp = ""
        poin_terakhir = None
        for row in rp["baris"]:
            m = row["metrik"]

            def sel(kunci, e=m):
                d = e.get(kunci) or {}
                u = d.get("ukur")
                if u is None:
                    return "<td>&mdash;</td>"
                dl = d.get("dari_langit")
                tail = f' <span class="net">{dl:.0%}</span>' if dl is not None else ""
                return f"<td><b>{u:.3f}</b>{tail}</td>"

            bobot = row.get("bobot_nilai")
            bobot_txt = f"{bobot:.0%}" if bobot is not None else "&mdash;"
            # Nama poin dicetak sekali per kelompok supaya mata membaca tiga poin, bukan lima baris.
            nama_poin = f"<b>{_esc(row['poin'])}</b>" if row["poin"] != poin_terakhir else ""
            poin_terakhir = row["poin"]
            baris_rp += (
                f"<tr><td>{nama_poin}</td><td>{_esc(row['cabang'])}</td>"
                f"<td>{bobot_txt}</td><td>{row['n']}</td>"
                # Nama jalur bisa datang dari dua kamus: konfigurasi search (_LABEL) atau
                # konfigurasi jalur rujukan (_LABEL_ANCHOR). Jangan biarkan id mentah tampil.
                f"<td class='net'>{_esc(_LABEL.get(row['konfigurasi']) or _LABEL_ANCHOR.get(row['konfigurasi'], row['konfigurasi']))}</td>"
                + sel(f"ndcg@{_K_OPERASI}") + sel(f"recall@{_K_OPERASI}")
                + sel(f"hit@{_K_OPERASI}") + sel("mrr") + "</tr>")
        blok_produksi = f"""
  <h2>1. Kinerja sistem &mdash; jalur produksi per poin</h2>
  <div class="peringatan" style="margin-bottom:14px">
  <b>Inilah satu-satunya tabel di laporan ini yang boleh dibaca sebagai kinerja produksi.</b>
  Tabel bagian 2 merata-ratakan seluruh topik dengan bobot sama, dan itu sengaja memuat jalur
  yang produksi TIDAK pakai: 52 topik menguji <code>itbx</code> lewat <code>search</code> yang
  hampir tak pernah menyala, dan 52 lagi memakai query lama <code>kdb</code> sebagai lengan
  &ldquo;sebelum&rdquo; dalam ablasi &mdash; keduanya ada untuk membuktikan sebab-akibat, bukan
  untuk mewakili perilaku nyata. Tabel ini hanya memuat cabang yang benar-benar dijalankan.
  </div>
  <p class="cat">Angka tebal = skor terukur. Angka kecil abu-abu di sebelahnya =
  <b>persen dari langit-langit</b>, yaitu nilai tertinggi yang MUNGKIN dicapai mengingat
  struktur label subset itu &mdash; bukan dari 1,0. Inilah angka yang layak dibaca sebagai
  &ldquo;seberapa dekat ke sempurna&rdquo;; bagian 3 menjelaskan alasannya. Lantai acak dan
  langit-langit dihitung <b>per subset</b>, bukan dipinjam dari angka global.</p>
  <div class="kartu"><table>
    <thead><tr><th>Poin</th><th>Cabang</th><th>Bobot trafik</th><th>n topik</th>
      <th>Jalur</th><th>nDCG@{_K_OPERASI}</th><th>Recall@{_K_OPERASI}</th>
      <th>Hit@{_K_OPERASI}</th><th>MRR</th></tr></thead>
    <tbody>{baris_rp}</tbody>
  </table></div>
  <p class="cat">Bobot trafik dihitung dari <code>{_esc(bt["sumber"])}</code>
  (n={bt["n"]} permohonan): <b>{bt["tanpa_subzona"]:.1%}</b> permohonan tiba TANPA sub-zona
  presisi, sehingga cabang berfilter keluarga yang dominan &mdash; bukan cabang exact.
  Poin <code>itbx</code> tidak punya pembagian cabang karena seluruh payload back-end membawa
  <code>dasar_hukum</code>, jadi ia selalu lewat <code>get_by_reference</code>.</p>
  <div class="peringatan">
  <b>Bacaannya.</b> <code>intensitas</code> dan <code>itbx</code> berada di
  <b>85&ndash;100%</b> dari langit-langitnya; <code>dampak</code> di kisaran <b>63&ndash;65%</b>.
  <br><br>
  Angka <code>dampak</code> sempat terbaca jauh lebih rendah (sekitar 33%), dan itu <b>mengukur
  label, bukan sistem</b>. Label lama memasukkan seluruh ketentuan kebencanaan Pasal 50; buktinya
  kategoris — ayat kawasan resapan air terambil pada <b>39 dari 39</b> kesempatan, sedangkan ayat
  gempa bumi dan banjir lahar pada <b>0 dari 95</b>. Query produksi
  <code>dampak tata guna lahan</code> diperluas retriever menjadi istilah hidrologi saja
  (limpasan, runoff, sumur resapan, zero delta Q, drainase), sehingga ketentuan kebencanaan memang
  tak terjangkau olehnya.
  <br><br>
  Kriteria label kini <b>kausal</b>, atas keputusan pemilik domain: yang sah disitasi hanya
  ketentuan atas dampak yang <b>ditimbulkan</b> pembangunan, bukan bahaya alam yang mengancam
  lokasi. Diterapkan sebagai aturan atas teks ayatnya (kewajiban pengendalian limpasan), bukan
  daftar nomor pasal. Sisa jaraknya ke 100% adalah kekurangan yang nyata dan belum
  ditindaklanjuti.
  </div>"""

    # --- Cara membaca skor: lantai, langit-langit, rujukan terbitan ----------------------
    bl = r.get("baseline") or muat_baseline()
    blok_tafsir = ""
    if bl:
        la = bl["lantai_acak"]["metrik"]
        bk = bl["lantai_acak"]["besar_kandidat"]
        lg = bl["langit_langit"]
        lgm = lg["maks"]
        # Nama agregat MRR/MAP -> kunci per-topik rr/ap, sama seperti _KUNCI_PER_QUERY.
        _BASE = {"mrr": "rr", "map": "ap"}
        pembanding_a, pembanding_b = produksi, (
            "dense+rerank" if produksi != "dense+rerank" and "dense+rerank" in kfg else None)
        baris_metrik = ""
        for metrik, catatan_metrik in (
            ("hit@1", "apakah chunk benar langsung di peringkat 1"),
            (f"hit@{_K_OPERASI}", "apakah retrieval GAGAL TOTAL pada kedalaman operasi"),
            ("hit@5", "idem, kedalaman lain (bukan titik operasi)"),
            ("hit@10", "idem, sejauh daftar dinilai"),
            (f"recall@{_K_OPERASI}", "berapa bagian chunk otoritatif yang sampai ke LLM"),
            ("recall@5", "idem, kedalaman lain"),
            (f"precision@{_K_OPERASI}", "LANGIT-LANGITNYA RENDAH — lihat catatan di bawah"),
            ("precision@5", "idem, langit-langitnya lebih rendah lagi"),
            (f"ndcg@{_K_OPERASI}", "metrik primer: jumlah DAN posisi, pada titik operasi"),
            ("ndcg@5", "idem, kedalaman lain"),
            ("mrr", "posisi chunk relevan PERTAMA"),
            ("map", "presisi rata-rata di tiap posisi relevan"),
        ):
            kunci_base = _BASE.get(metrik, metrik)
            lantai_v = la.get(kunci_base)
            langit_v = lgm.get(kunci_base)
            ukur_a = agg.get(produksi, {}).get(metrik)
            ukur_b = agg.get(pembanding_b, {}).get(metrik) if pembanding_b else None
            if lantai_v is None or ukur_a is None:
                continue
            kali = f"{ukur_a / lantai_v:,.0f}&times;" if lantai_v > 0 else "&mdash;"
            dari_langit = (f"{ukur_a / langit_v:.0%}" if langit_v else "&mdash;")
            sorot = ' class="sorot"' if metrik == f"ndcg@{_K_OPERASI}" else ""
            baris_metrik += (
                f"<tr{sorot}><td><b>{_esc(metrik)}</b></td>"
                f"<td>{lantai_v:.4f}</td>"
                f"<td><b>{ukur_a:.3f}</b></td>"
                + (f"<td>{ukur_b:.3f}</td>" if ukur_b is not None else "<td>&mdash;</td>")
                + f"<td>{langit_v:.3f}</td><td>{kali}</td><td><b>{dari_langit}</b></td>"
                f"<td class='net'>{catatan_metrik}</td></tr>")
        blok_tafsir = f"""
  <h2>3. Cara membaca skor — dan mengapa tidak ada ambang &ldquo;bagus/buruk&rdquo; yang baku</h2>
  <div class="peringatan" style="margin-bottom:14px">
  <b>Tidak ada ambang absolut untuk nDCG di literatur yang ditelaah sejawat.</b> Yang beredar
  (&ldquo;&gt;0,9 sangat baik, 0,7&ndash;0,9 baik, &lt;0,5 buruk&rdquo;) berasal dari dokumentasi
  vendor dan blog, bukan dari makalah. Ambang semacam itu <b>tidak bisa</b> ada, karena nDCG
  bergantung pada koleksinya: berapa kandidat yang lolos filter, berapa chunk yang dilabeli
  relevan, dan serapat apa pesaingnya. Skor yang sama bisa berarti nyaris sempurna di satu
  koleksi dan nyaris acak di koleksi lain. Mencantumkan ambang pinjaman di laporan ini justru
  akan <b>melemahkan</b> keabsahannya di mata pemeriksa yang paham IR.
  </div>
  <p class="cat">Yang sah adalah tiga pembanding berikut: <b>lantai</b> yang dihitung dari
  koleksi ini sendiri, <b>langit-langit</b> yang ditentukan struktur label, dan <b>rujukan
  terbitan</b> sebagai konteks besaran.</p>

  <h3>a. Lantai, langit-langit, dan posisi sistem &mdash; tiap metrik</h3>
  <p class="cat"><b>Lantai</b> = harapan metrik bila sistem mengambil {_TOP_N} chunk ACAK dari
  kandidat yang lolos filter topik itu; filternya sama, yang hilang hanya peringkatnya. Besar
  kandidat per topik: min {bk['min']}, median {bk['median']:.0f}, maks {bk['maks']} chunk, jadi
  menebak bukan hal yang mudah. <b>Langit-langit</b> = nilai tertinggi yang MUNGKIN mengingat
  struktur label (rata-rata {lg['n_relevan_rata']:.2f} chunk relevan per topik). Kolom terakhir
  sebelum catatan &mdash; <b>% dari langit-langit</b> &mdash; adalah angka yang paling layak
  dibaca sebagai &ldquo;seberapa dekat ke sempurna&rdquo;.</p>
  <div class="kartu"><table>
    <thead><tr><th>Metrik</th><th>Lantai acak</th>
      <th>{_esc(_LABEL.get(produksi, produksi))}</th>
      <th>{_esc(_LABEL.get(pembanding_b, pembanding_b or '&mdash;'))}</th>
      <th>Langit-langit</th><th>&divide; lantai</th><th>% dari langit</th>
      <th>Yang diukurnya</th></tr></thead>
    <tbody>{baris_metrik}</tbody>
  </table></div>

  <h3>b. Catatan per metrik &mdash; apa yang boleh dan tidak boleh disimpulkan</h3>
  <div class="peringatan">
  <ul>
    <li><b>Precision@k tidak boleh dibandingkan dengan 1,0 di eval set ini.</b>
        {lg['topik_relevan_kurang_dari_k']} dari {r['n_query']} topik punya <i>kurang</i> dari
        {_K_OPERASI} chunk relevan, sehingga sebagian slot PASTI terisi chunk tak relevan walau
        sistem sempurna. Langit-langitnya Precision@{_K_OPERASI} =
        <b>{lgm[f'precision@{_K_OPERASI}']:.3f}</b> dan Precision@5 =
        <b>{lgm['precision@5']:.3f}</b>. Jadi membaca P@5 sekitar 0,2 sebagai &ldquo;hanya 20%,
        buruk&rdquo; adalah salah baca: yang benar adalah membandingkannya dengan 0,311, dan
        kolom &ldquo;% dari langit&rdquo; di tabel atas sudah melakukannya.</li>
    <li><b>Hit@k adalah lantai, bukan mutu.</b> Ia hanya menjawab &ldquo;apakah retrieval gagal
        total&rdquo;. Nilainya naik dengan sendirinya bila k diperbesar &mdash; lantai acaknya
        pun naik dari {la['hit@1']:.4f} di k=1 menjadi {la['hit@10']:.4f} di k=10. Hit@k tinggi
        pada k besar <b>bukan</b> bukti sistem bagus; yang menentukan adalah k titik operasi.</li>
    <li><b>Recall@k dibatasi jumlah label.</b> Langit-langit Recall@{_K_OPERASI} =
        {lgm[f'recall@{_K_OPERASI}']:.3f} ({lg['topik_relevan_lebih_dari_k']} topik punya lebih
        dari {_K_OPERASI} chunk relevan), jadi di sini perbandingan dengan 1,0 masih nyaris adil.
        Tapi pemeriksaan ini wajib diulang setiap eval set berubah: begitu satu topik punya 6
        chunk relevan sementara sistem mengirim {_K_OPERASI}, langit-langitnya 50%.</li>
    <li><b>nDCG@k, MRR, dan MAP tidak terkena batas itu</b> &mdash; idealnya ikut dibatasi
        min(jumlah relevan, k), sehingga 1,0 tetap bisa dicapai. Karena itu nDCG@{_K_OPERASI}
        dipakai sebagai metrik primer di laporan ini, dan Precision dilaporkan hanya sebagai
        pelengkap.</li>
    <li><b>MRR hanya melihat chunk relevan PERTAMA.</b> Ia buta terhadap sisanya, jadi MRR
        tinggi dengan Recall rendah berarti &ldquo;satu jawaban benar di atas, sisanya tak
        terambil&rdquo; &mdash; itu justru pola yang terlihat pada poin dampak.</li>
    <li><b>Seluruh angka di tabel ini tak tertimbang per topik</b>, jadi bukan angka produksi.
        Jalur produksi berbeda per poin &mdash; lihat bagian 1.</li>
  </ul>
  </div>

  <h3>c. Rujukan terbitan: berapa skor yang wajar di koleksi nyata</h3>
  <p class="cat">Pada <b>BEIR</b> &mdash; tolok ukur retrieval zero-shot yang paling banyak
  dipakai, 18 koleksi lintas domain &mdash; <b>BM25</b>, baseline leksikal standar yang kuat,
  mencatat rata-rata <b>nDCG@10 &asymp; 0,43</b>. Artinya kisaran 0,3&ndash;0,5 adalah wilayah
  kerja baseline yang terhormat di koleksi heterogen nyata, bukan tanda kegagalan. Angka itu
  <b>bukan ambang</b> dan tidak sebanding langsung dengan angka kita (beda koleksi, beda k, beda
  bentuk relevansi) &mdash; ia hanya memberi rasa besaran.</p>

  <h3>d. Untuk klaim perbandingan: signifikansi dan ukuran efek</h3>
  <p class="cat">Karena ambang absolut tak tersedia, klaim yang dapat dipertahankan adalah klaim
  <b>relatif</b>: konfigurasi A versus B atas topik dan label yang IDENTIK. Untuk itu laporan ini
  memakai uji berpasangan <b>Wilcoxon signed-rank</b> dan melaporkan <b>ukuran efek</b>
  (rank-biserial) berdampingan dengan p-value, supaya &ldquo;signifikan&rdquo; tidak tertukar
  dengan &ldquo;besar&rdquo;. Konvensi besaran yang dipakai sebagai rujukan kasar adalah
  Cohen: 0,10 kecil &middot; 0,30 sedang &middot; 0,50 besar &mdash; dengan catatan Cohen sendiri
  kemudian memperingatkan agar konvensi itu tidak dipakai lepas dari konteks bidangnya.
  Selang kepercayaan dihitung lewat <b>bootstrap</b> atas topik.</p>

  <div class="peringatan">
  <b>Rujukan.</b> Metodologi di laporan ini bersandar pada:
  <ul>
    <li>Järvelin, K. &amp; Kekäläinen, J. (2002). <i>Cumulated Gain-Based Evaluation of IR
        Techniques.</i> ACM TOIS 20(4):422&ndash;446. doi:10.1145/582415.582418 &mdash; asal
        metrik nDCG. <b>Catatan penting:</b> nDCG dirancang untuk relevansi BERGRADASI,
        sedangkan label kita biner, sehingga nDCG di sini lebih kasar daripada maksud aslinya.</li>
    <li>Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. &amp; Gurevych, I. (2021).
        <i>BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
        Models.</i> NeurIPS 2021 Datasets &amp; Benchmarks Track &mdash; alasan nDCG@k dipakai
        sebagai metrik primer.</li>
    <li>Kamalloo, E., Thakur, N., Lassance, C., Ma, X., Yang, J. &amp; Lin, J. (2024).
        <i>Resources for Brewing BEIR: Reproducible Reference Models and Statistical Analyses.</i>
        SIGIR 2024, 1431&ndash;1440 &mdash; sumber angka baseline BM25 di butir (c).</li>
    <li>Sakai, T. (2018). <i>Laboratory Experiments in Information Retrieval: Sample Sizes,
        Effect Sizes, and Statistical Power.</i> Springer &mdash; dasar pelaporan ukuran efek,
        daya statistik, dan penentuan jumlah topik.</li>
    <li>Wilcoxon, F. (1945). <i>Individual Comparisons by Ranking Methods.</i> Biometrics
        Bulletin 1(6):80&ndash;83 &mdash; uji berpasangan yang dipakai.</li>
    <li>Efron, B. (1979). <i>Bootstrap Methods: Another Look at the Jackknife.</i> Annals of
        Statistics 7(1):1&ndash;26 &mdash; dasar selang kepercayaan bootstrap.</li>
    <li>Cohen, J. (1988). <i>Statistical Power Analysis for the Behavioral Sciences</i> (ed. ke-2).
        Lawrence Erlbaum &mdash; konvensi besaran ukuran efek.</li>
  </ul>
  Rincian bibliografis di atas dirangkum dari pencarian sumber sekunder. <b>Sebelum dikutip di
  dokumen resmi (mis. berkas pengajuan paten), tiap entri &mdash; terutama angka baseline BM25
  dan nomor halaman &mdash; wajib diperiksa ulang ke makalah aslinya.</b>
  </div>"""

    # --- Ablasi jalur rujukan (anchor) ---------------------------------------------------
    anc = r.get("anchor") or {}
    blok_anchor = ""
    if anc.get("n_query"):
        baris_anc = ""
        for k in anc["konfigurasi"]:
            a, c = anc["agregat"][k], anc["statistik"]["ci"][k]
            sorot = ' class="sorot"' if k == anc["produksi"] else ""
            uji = anc["statistik"]["uji_vs_produksi"].get(k, {}).get(f"ndcg@{_K_OPERASI}")
            baris_anc += (
                f'<tr{sorot}><td><b>{_esc(_LABEL_ANCHOR.get(k, k))}</b></td>'
                f'<td>{_fmt_ci(c[f"ndcg@{_K_OPERASI}"])}</td>'
                f'<td>{_fmt_ci(c[f"recall@{_K_OPERASI}"], persen=True)}</td>'
                f'<td>{a[f"hit@{_K_OPERASI}"]:.0%}</td><td>{a["mrr"]:.3f}</td>'
                f'<td>{a["query_tanpa_hasil_relevan"]}/{anc["n_query"]}</td>'
                f'<td>{_fmt_uji(uji)}</td></tr>')
        blok_anchor = f"""
  <h2>{_nomor_anchor}. Ablasi jalur rujukan — pemilihan sitasi sadar-zona</h2>
  <p class="cat">Jalur <code>get_by_reference</code> menangani <b>seluruh poin ITBX di
  produksi</b> (setiap payload back-end membawa <code>dasar_hukum</code>), jadi inilah jalur yang
  paling sering dipakai sistem — dan jalur yang sebelumnya tidak pernah dievaluasi. Kedua
  konfigurasi dinilai atas <b>{anc["n_query"]} topik yang sama</b>: memotong daftar kandidat
  menurut urutan yang dikembalikan DB, versus memilih menurut kecocokan zona pemohon.</p>
  <table>
    <thead><tr><th>Konfigurasi</th><th>nDCG@{_K_OPERASI}</th><th>Recall@{_K_OPERASI}</th>
      <th>Hit@{_K_OPERASI}</th><th>MRR</th><th>Topik tanpa hasil relevan</th>
      <th>vs produksi (nDCG@{_K_OPERASI})</th></tr></thead>
    <tbody>{baris_anc}</tbody>
  </table>
  <p class="cat">Latensi jalur ini <b>{anc["latensi_ms"][anc["produksi"]]:,.0f} ms</b> — murni SQL,
  tanpa panggilan embedding maupun rerank.</p>"""

    # --- Ablasi string query -------------------------------------------------------------
    abq = r.get("ablasi_query") or {}
    blok_abq = ""
    if abq.get("n_keluarga"):
        nk = abq["n_keluarga"]
        baris_abq = ""
        for k in kfg:
            d = abq["per_konfigurasi"].get(k)
            if not d:
                continue
            sorot = ' class="sorot"' if k == produksi else ""
            t, b = d["lengan"]["tajam"], d["lengan"]["kdb"]
            u = d["uji"][f"ndcg@{_K_OPERASI}"]
            baris_abq += (
                f'<tr{sorot}><td><b>{_esc(_LABEL.get(k, k))}</b></td>'
                f'<td>{_fmt_ci(t[f"ndcg@{_K_OPERASI}"])}</td>'
                f'<td>{_fmt_ci(b[f"ndcg@{_K_OPERASI}"])}</td>'
                f'<td>{t[f"recall@{_K_OPERASI}"]["rata"]:.0%} &rarr; '
                f'{b[f"recall@{_K_OPERASI}"]["rata"]:.0%}</td>'
                f'<td>{t["peringkat1"]}/{nk} vs {b["peringkat1"]}/{nk}</td>'
                f'<td>{_fmt_uji(u)}</td></tr>')
        blok_abq = f"""
  <h2>{_nomor_abq}. Ablasi string query — cabang tanpa sub-zona presisi</h2>
  <p class="cat">Saat back-end tidak mengirim sub-zona presisi, sistem hanya bisa menyaring per
  <b>keluarga</b> zona. Dulu pada cabang itu ia menerbitkan query pendek
  <code>{_esc(abq["query_kdb"])}</code>, dengan alasan bahwa query tajam tanpa filter exact
  berisiko mengutip tabel sub-zona yang salah. <b>Ablasi inilah yang membuka gating itu:</b>
  produksi kini menerbitkan <code>{_esc(abq["query_tajam"])}</code> pada kedua cabang,
  berpasangan dengan caveat sub-zona tak terkonfirmasi di narasi dan catatan global.
  <b>Filter dan label kedua lengan identik</b> — satu-satunya yang berbeda adalah teks query,
  atas {nk} keluarga zona yang sama. Keduanya dinilai di run yang sama, jadi selisihnya tidak
  bisa dijelaskan oleh perbedaan indeks atau versi kode.</p>
  <table>
    <thead><tr><th>Konfigurasi</th><th>nDCG@{_K_OPERASI} (tajam)</th>
      <th>nDCG@{_K_OPERASI} (&ldquo;{_esc(abq["query_kdb"])}&rdquo;)</th>
      <th>Recall@{_K_OPERASI} tajam &rarr; pendek</th>
      <th>Peringkat 1 (tajam vs pendek)</th>
      <th>Uji berpasangan</th></tr></thead>
    <tbody>{baris_abq}</tbody>
  </table>
  <div class="peringatan">
  <b>Batas pembacaan yang tidak boleh dilewat.</b> Dengan filter keluarga, tiga chunk teratas
  bisa berisi tabel <b>sub-zona berbeda dengan ambang KDB yang berbeda</b>. Label di sini
  menganggap seluruh tabel satu keluarga sah — karena itu memang batas informasi yang tersedia
  sistem — sehingga ablasi ini <b>tidak bisa memutuskan</b> apakah narasi lalu menyajikan angka
  sub-zona yang salah sebagai milik pemohon. Risiko itu ada di lapis generasi, bukan retrieval,
  dan harus ditangani di sana (mis. narasi menyatakan sub-zona belum terkonfirmasi).
  </div>"""

    # --- Atribusi sitasi (dari log keluaran nyata) --------------------------------------
    atr = r.get("atribusi") or muat_atribusi()
    blok_atr = ""
    if atr:
        ca = atr["cakupan"]
        rt = ca.get("rentang_tanggal_dinilai")
        rz = atr["recall_zona"]
        baris_atr = "".join(
            f"<tr><td>{_esc(label)}</td><td>{_fmt_atribusi(atr[kunci])}</td>"
            f"<td class='net'>{_esc(ket)}</td></tr>"
            for kunci, label, ket in (
                ("presisi_korpus", "Presisi sitasi — id chunk ada di korpus",
                 "menangkap halusinasi rujukan"),
                ("presisi_anchor", "Presisi sitasi anchor — indeks dasar_hukum sah",
                 "anchor-N harus menunjuk item yang benar-benar dikirim back-end"),
                ("groundedness_kutipan", "Groundedness kutipan — teks ada di chunk disitasi",
                 "menangkap kutipan karangan; normalisasi spasi &amp; tanda baca"),
                ("recall_zona", "Recall sitasi terhadap keluarga zona pemohon",
                 "menangkap kelas bug APP-2026-6191/2428"),
                ("kesepakatan_flag_terverifikasi", "Kesepakatan flag <code>terverifikasi</code>",
                 "flag sistem hanya memeriksa keberadaan id, bukan isi kutipan"),
            ))
        baris_rz = "".join(
            f"<tr><td><code>{_esc(pid)}</code></td><td>{_fmt_atribusi(e)}</td></tr>"
            for pid, e in rz.get("per_poin", {}).items())
        g = atr["groundedness_kutipan"]
        catatan_judul = (
            f"<br><br><b>Mode gagal yang tertangkap:</b> {g['kutipan_judul_dokumen']} kutipan "
            "berisi <b>judul dokumen</b>, bukan isi pasal — formalnya bersitasi dan "
            "<code>terverifikasi</code> tetap true, substansinya tidak menjelaskan apa pun."
        ) if g.get("kutipan_judul_dokumen") else ""
        blok_atr = f"""
  <h2>{_nomor_atr}. Atribusi sitasi — dihitung dari keluaran nyata</h2>
  <p class="cat">Bagian-bagian di atas mengukur leg <b>retrieval</b>: apakah chunk yang benar
  masuk top-k. Bagian ini mengukur hal lain, dan tidak butuh anotator: apakah sitasi yang
  <b>akhirnya muncul di luaran</b> menunjuk chunk yang nyata, dan apakah kutipannya benar ada di
  chunk itu. Dihitung dari <code>logs/precheck.jsonl</code> + korpus oleh
  <code>eval/metrik_atribusi.py</code>.</p>
  <div class="peringatan">
  <b>Baca cakupannya dulu.</b> Log bukan trafik produksi yang bersih: dari
  <b>{ca['permohonan_era_sekarang']}</b> permohonan era 3-poin, hanya
  <b>{ca['permohonan_retriever_nyata']}</b> yang sitasinya berasal dari retriever nyata —
  <b>{ca['permohonan_dari_mock']}</b> berasal dari <code>MockRetriever</code> (replay/test
  lokal) dan dibuang. Kalau tidak dipilah, yang terukur adalah mock, bukan sistem.
  {f"Rentang tanggal yang terhitung: <b>{rt[0][:10]} .. {rt[1][:10]}</b>." if rt else ""}
  {f"Disaring sejak <b>{_esc(atr['sejak'])}</b>." if atr.get("sejak") else
   "<b>TIDAK disaring per tanggal</b>, jadi angkanya mencampur beberapa versi kode — "
   "pakai <code>--sejak</code> sebelum mengutipnya."}
  Sitasi yang dinilai: <b>{ca['sitasi_dinilai']}</b> {_esc(str(ca['sitasi_per_jenis_id']))}.
  </div>
  <div class="kartu"><table>
    <thead><tr><th>Metrik</th><th>Nilai</th><th>Apa yang ditangkap</th></tr></thead>
    <tbody>{baris_atr}</tbody>
  </table></div>
  <p class="cat">Recall zona dihitung <b>per poin</b>, bukan per permohonan: ketentuan
  <code>dampak</code> tersaji sebagai pasal prosa lintas-zona, jadi menuntut sitasi ber-zona di
  sana akan menandai jawaban yang benar sebagai salah — karena itu ia tidak muncul di tabel
  berikut.</p>
  <div class="kartu"><table>
    <thead><tr><th>Poin</th><th>Recall sitasi thd zona pemohon</th></tr></thead>
    <tbody>{baris_rz}</tbody>
  </table></div>
  <div class="peringatan">{catatan_judul}</div>"""

    gen = r.get("generasi") or {}
    blok_gen = ""
    if gen.get("n_poin"):
        palet = {"berhasil": "#059669", "guardrail_menolak": "#d97706",
                 "panggilan_llm_gagal": "#dc2626", "retrieval_kosong": "#7c3aed"}
        sitasi_txt = (f"{gen['sitasi_terverifikasi']}/{gen['sitasi']} sitasi terverifikasi"
                      if gen.get("sitasi") else "tidak ada sitasi tercatat")
        n_gagal = gen["sebab"].get("panggilan_llm_gagal", 0)
        blok_gen = f"""
  <h2>{_nomor_gen}. Sisi generasi — data operasional historis</h2>
  <div class="peringatan" style="margin-bottom:14px">
  <b>Baca dengan hati-hati — ini BUKAN tingkat kegagalan produksi.</b> Rekap ini diambil dari kunci
  <code>diagnostik</code> yang sudah terkumpul di <code>logs/precheck.jsonl</code>
  ({gen['n_permohonan']} permohonan, {gen['n_poin']} poin), dan isinya <b>didominasi replay
  pengujian</b>, bukan lalu lintas pemohon. Porsi <code>panggilan_llm_gagal</code>
  ({n_gagal} poin) sebagian besar berasal dari replay batch yang sengaja dijalankan beruntun
  sampai memicu rate limit provider, plus satu uji yang memang dirancang selalu gagal. Angka
  produksi yang sesungguhnya baru bisa dihitung setelah lalu lintas organik terkumpul.
  Yang tetap bermakna di sini: <b>tidak ada satu pun poin dengan sebab
  <code>guardrail_menolak</code></b> — konsisten dengan hasil verifikasi setelah perbaikan.
  </div>
  <div class="kartu">{_donat("Sebab akhir tiap poin", gen["sebab"], palet)}</div>
  <p class="cat">Integritas sitasi pada periode yang sama: <b>{sitasi_txt}</b>. Perlu diingat,
  "terverifikasi" berarti <i>citation_id</i> memang ada di daftar kandidat yang disodorkan
  (anti-halusinasi) — bukan jaminan kandidatnya relevan secara hukum.</p>"""

    korpus = r["korpus"]
    lvl = ", ".join(f"{v} {k}" for k, v in sorted(korpus["per_level"].items()))

    md = r.get("metadata") or {}
    # Eval set bisa berubah setelah hasil ini diukur; kalau ya, seluruh angka di bawah dinilai
    # atas label yang sudah tidak berlaku dan laporan HARUS mengatakannya.
    _sidik_kini = _sidik_eval_set(_EVAL_SET.parent / (r.get("berkas_eval") or ""))
    _sidik_ukur = r.get("sidik_eval_set")
    blok_basi = ""
    if _sidik_kini and not _sidik_ukur:
        # Hasil dari versi harness sebelum sidik jari direkam. Tak bisa dipastikan masih sepadan
        # dgn eval set sekarang — dan diam soal itu persis yang membuat hasil basi terlihat sah.
        blok_basi = ('<div class="peringatan"><b>Kesesuaian hasil ini dgn eval set sekarang tidak '
                     'dapat dipastikan.</b> Ia diukur oleh versi harness sebelum sidik jari eval '
                     'set direkam, sehingga perubahan label setelahnya tidak akan terdeteksi. '
                     'Jalankan ulang <code>python -m eval.eval_rag</code> untuk memastikan.</div>')
    if _sidik_ukur and _sidik_kini and _sidik_ukur != _sidik_kini:
        blok_basi = (
            '<div class="peringatan"><b>PERINGATAN: hasil ini BASI.</b> Eval set '
            f'<code>{_esc(r.get("berkas_eval"))}</code> sudah berubah sejak pengukuran ini '
            f'(sidik saat diukur <code>{_esc(_sidik_ukur)}</code>, sekarang '
            f'<code>{_esc(_sidik_kini)}</code>). Seluruh angka di bawah dinilai atas label '
            'yang sudah tidak berlaku. Jalankan ulang <code>python -m eval.eval_rag</code> '
            'sebelum angka mana pun dikutip.</div>')
    if md:
        kotor = ' <b>(working tree kotor — ada perubahan belum ter-commit)</b>' if md.get("commit_kotor") else ""
        meta_txt = _esc(
            f'commit {md.get("commit") or "?"}{"" if not kotor else ""} · '
            f'embedding {md.get("model_embedding")} · rerank {md.get("model_rerank")} · '
            f'LLM {md.get("model_llm")} · kandidat={md.get("kandidat_k")} · '
            f'dinilai sampai peringkat {md.get("top_n_dinilai")}') + kotor
    else:
        meta_txt = "(metadata tidak tercatat — hasil dari versi harness lama)"

    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluasi Retrieval RAG — RDTR {_esc(r['wilayah'])}</title>
<style>
  :root {{ color-scheme: light; --bg:#f8fafc; --kartu:#fff; --grs:#e2e8f0; --tks:#0f172a; --redup:#64748b; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:28px 18px 60px; background:var(--bg); color:var(--tks);
         font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size:25px; margin:0 0 4px; letter-spacing:-.02em; }}
  h2 {{ font-size:18px; margin:34px 0 10px; letter-spacing:-.01em; }}
  h3 {{ font-size:15px; margin:22px 0 6px; color:var(--redup); font-weight:600; }}
  .sub {{ color:var(--redup); margin:0 0 6px; }}
  .kartu {{ background:var(--kartu); border:1px solid var(--grs); border-radius:10px;
            padding:14px; margin:14px 0; overflow-x:auto; }}
  svg {{ display:block; width:100%; height:auto; min-width:640px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; min-width:640px; }}
  th,td {{ padding:7px 9px; border-bottom:1px solid var(--grs); text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ background:#f1f5f9; font-weight:600; }}
  tr.sorot td {{ background:#ecfdf5; }}
  .cat {{ color:var(--redup); font-size:14px; }}
  .pil {{ display:inline-block; background:#eef2ff; color:#3730a3; border-radius:999px;
          padding:2px 10px; font-size:12.5px; margin:0 6px 6px 0; }}
  .peringatan {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:12px 14px; }}
  .peringatan h2 {{ margin-top:0; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:13px; }}
  .jdl {{ font:600 13.5px system-ui; fill:#0f172a; }}
  .ax  {{ font:11.5px system-ui; fill:#64748b; }}
  .val {{ font:600 11px system-ui; fill:#0f172a; }}
  .cel {{ font:600 11px system-ui; }}
  .lg  {{ font:12px system-ui; fill:#334155; }}
  .ctr {{ font:600 22px system-ui; fill:#0f172a; }}
  .grid {{ stroke:#e2e8f0; stroke-width:1; }}
  .ci   {{ color:#64748b; font-size:11.5px; white-space:nowrap; }}
  .net  {{ color:#64748b; }}
  .baik {{ color:#047857; font-weight:600; }}
  .buruk{{ color:#b91c1c; font-weight:600; }}
  .meta {{ color:#64748b; font-size:12.5px; border-top:1px solid var(--grs); padding-top:12px; margin-top:26px; }}
</style></head><body><main>

  <h1>Evaluasi Retrieval RAG</h1>
  <p class="sub">RDTR {_esc(r['wilayah'])} · dibuat {_esc(r['dibuat'])} ·
     {r['n_query']} query · dinilai sampai peringkat {r['top_n_dinilai']}</p>
  <p>
    <span class="pil">embedding: {_esc(r['provider_embedding'])}</span>
    <span class="pil">rerank: {_esc(r['provider_rerank'])}</span>
    <span class="pil">korpus: {korpus['chunk']} chunk ({_esc(lvl)})</span>
    <span class="pil">vektor: {korpus['vektor']}</span>
  </p>

{blok_basi}
{blok_produksi}

  <h2>2. Titik operasi &amp; ablasi konfigurasi (k={_K_OPERASI})</h2>
  <div class="peringatan" style="margin-bottom:14px">
  <b>Tabel ini membandingkan KONFIGURASI, bukan melaporkan kinerja.</b> Angka kinerja ada di
  bagian 1. Di sini tiap lapis retrieval dinilai atas seluruh topik uji &mdash; termasuk lengan
  kontrafaktual yang sengaja dibuat berskor rendah &mdash; supaya sumbangan tiap komponen
  terlihat, bukan diasumsikan. Sistem mengirim <b>{_K_OPERASI} chunk</b> ke LLM
  (<code>generator.top_k_dukungan</code>), jadi hanya kedalaman ini yang mewakili apa yang
  benar-benar diterima sistem. Metrik pada kedalaman lain (bagian 4 &amp; 5) berguna untuk memahami
  perilaku, tapi <b>tidak boleh dipakai sebagai klaim kinerja</b>.
  Tiap angka disertai selang kepercayaan 95% (bootstrap atas topik, n={r['n_query']}); kolom
  terakhir menguji apakah selisihnya nyata (Wilcoxon signed-rank berpasangan).
  <b>Selisih yang dinyatakan "setara" tidak boleh diklaim sebagai keunggulan.</b>

  <div class="peringatan" style="margin-top:14px">
  <b>&ldquo;Hit@5 dan Recall@5 lebih tinggi &mdash; kenapa produksi tidak memakai k=5 saja?&rdquo;</b>
  Pertanyaan yang wajar, dan jawabannya <b>kapasitas, bukan mutu</b>.
  <br><br>
  Pada metrik, k=5 memang lebih baik: Recall naik dari <b>63%</b> ke <b>94%</b> dari
  langit-langitnya. Tapi biayanya diukur, bukan ditaksir &mdash; dari pesan batas provider yang
  tercatat saat replay dijalankan pada konfigurasi produksi (k={_K_OPERASI}):
  <code>Limit 8000, Used 1800, Requested 6707</code>. Satu panggilan poin <b>sudah meminta
  4.661&ndash;6.707 token</b> (rata-rata 5.804) dari batas 8.000 token/menit.
  <br><br>
  Sebabnya bukan ukuran prompt semata: provider menghitung <b>anggaran keluaran yang dicadangkan</b>
  (<code>max_tokens=2048</code>) ke dalam kuota yang sama. Jadi biaya nyata = prompt + cadangan
  keluaran. Menaikkan ke k=5 menambah 485&ndash;1.042 token per poin, sehingga satu panggilan
  menyentuh <b>~7.700 dari 8.000</b> &mdash; sementara <b>tiga poin berjalan paralel</b>. Batas
  harian ikut mengikat: <code>tokens per day: Limit 200000</code> per kunci.
  <br><br>
  Jadi k={_K_OPERASI} adalah batas yang dipaksakan kapasitas, dan angka di tabel ini harus dibaca
  dengan itu. <b>Sisi mutunya belum terjawab:</b> apakah LLM memakai 5 chunk lebih baik daripada 3
  &mdash; atau justru tersesat, seperti pernah terjadi saat prompt membanjirinya &mdash; hanya bisa
  diuji dengan menjalankan generasi sungguhan (<code>eval/replay_k.py</code>). Percobaan pertama
  gagal karena kuota harian habis di tengah jalan, dan hasilnya <b>tidak dipakai</b>.
  </div>
  </div>
  <div class="kartu"><table>
    <thead><tr><th>Konfigurasi</th><th>nDCG@{_K_OPERASI}</th><th>Recall@{_K_OPERASI}</th>
      <th>Hit@{_K_OPERASI}</th><th>MRR</th>
      <th>vs {_esc(_LABEL.get(produksi, produksi))} (nDCG@{_K_OPERASI})</th></tr></thead>
    <tbody>{baris_operasi}{baris_produksi_ringkas}</tbody>
  </table></div>
  <div class="peringatan">
  <b>Tiap poin memakai jalur yang berbeda.</b> Sejak fusi retrieval dibuat per-poin, tak ada
  satu baris konfigurasi pun yang mewakili seluruh sistem — baris "Jalur produksi saja" di
  tabel atas sudah menggabungkannya menurut poin. Rinciannya:
  <table style="margin:10px 0">
    <thead><tr><th>Poin</th><th>Jalur produksi</th></tr></thead>
    <tbody>{baris_peta}</tbody>
  </table>
  Tabel ini menyatakan <code>dense+rerank</code> dan <code>rrf+rerank</code> <b>setara</b>
  (selisih +0,006, p=0,49) — dan itu justru contoh terbaik mengapa rata-rata tak tertimbang
  menyesatkan. Per poin, keduanya berbeda secara signifikan ke arah yang <b>berlawanan</b>, lalu
  saling meniadakan di rata-rata:
  <ul>
    <li><code>intensitas</code>: dense+rerank <b>unggul</b> +0,077 (p=0,001, efek +1,00)</li>
    <li><code>dampak</code>: dense+rerank <b>kalah</b> −0,289 (p&lt;0,001, efek −0,86)</li>
  </ul>
  Itulah sebabnya fusi dibuat per-poin: menyimpulkan &ldquo;keduanya setara, pilih mana saja&rdquo;
  dari tabel ini akan salah untuk kedua poin sekaligus. Bagi <code>itbx</code> pilihan fusi tak
  relevan — di produksi ia tak menyentuh fusi sama sekali.
  <br><br>Kolom pembanding memakai <b>{_esc(_LABEL.get(produksi, produksi))}</b> — konfigurasi
  yang berlaku saat angka ini diukur. Empat baris pertama adalah rata-rata tak tertimbang atas
  SELURUH {r['n_query']} topik, dan itu <b>bukan angka produksi</b>: separuhnya menguji jalur
  yang produksi tidak pakai — 52 topik memakai query lama sebagai lengan &ldquo;sebelum&rdquo;
  dalam ablasi (karena itu skornya memang rendah) dan 52 lagi menguji jalur <code>itbx</code>
  lewat <code>search</code> yang hampir tak pernah menyala. <b>Baris terakhir yang disorot</b>
  hanya menghitung cabang yang benar-benar dijalankan, dengan konfigurasi per-poin seperti di
  produksi. Poin <code>itbx</code> tidak termasuk di sana karena ia tak lewat
  <code>search</code> sama sekali — angkanya ada di bagian ablasi jalur rujukan.
  </div>
{blok_tafsir}

  <h2>4. Ringkasan metrik &amp; ablasi per lapis</h2>
  <p class="cat">Tiap lapis dinilai sendiri supaya kontribusinya terlihat, bukan diasumsikan.
  Baris hijau = konfigurasi pembanding ({_esc(_LABEL.get(produksi, produksi))}), yaitu yang
  berlaku saat angka ini diukur — bukan jalur produksi tiap poin hari ini (lihat bagian 1).</p>
  <div class="kartu"><table>
    <thead><tr><th>Konfigurasi</th>{''.join(f'<th>Hit@{n}</th>' for n in _K_LIST)}
      <th>MRR</th><th>MAP</th><th>nDCG@5</th><th>Recall@5</th><th>P@5</th>
      <th>Gagal</th><th>Latensi</th></tr></thead>
    <tbody>{baris_tabel}</tbody>
  </table></div>

  <h2>5. Hit-Rate menurut kedalaman</h2>
  <p class="cat">Seberapa sering minimal satu chunk relevan masuk peringkat-k teratas.</p>
  <div class="kartu">{_bar_berkelompok("Hit-Rate@k", [f"@{n}" for n in _K_LIST], hit)}</div>

  <h2>6. Mutu peringkat</h2>
  <p class="cat">Hit-Rate hanya menanyakan "ketemu atau tidak". Metrik di bawah menanyakan
  "ketemu di posisi berapa" dan "berapa banyak yang ketemu" — di sinilah reranker seharusnya
  membayar dirinya sendiri.</p>
  <div class="kartu">{_bar_berkelompok("MRR · MAP · nDCG@5 · Recall@5", mutu_kat, mutu, 1.0, "{:.2f}")}</div>
  <div class="kartu">{_strip_peringkat(r["detail_per_query"], kfg)}</div>
  <div class="kartu">{_bar_horizontal("Latensi rata-rata per query",
      [_LABEL.get(k, k) for k in kfg], [r["latensi_ms"][k] for k in kfg],
      [_WARNA.get(k, "#64748b") for k in kfg], " ms")}</div>
{blok_anchor}
{blok_abq}
{blok_atr}
{blok_gen}

  <h2>{_nomor_batas}. Batas pembacaan</h2>
  <div class="peringatan">
  <ul>
    <li><b>Ground truth diseed developer, belum divalidasi ahli tata ruang.</b> Angka di sini
        mengukur konsistensi sistem terhadap label kami sendiri — bukan kebenaran hukum.</li>
    <li><b>{r['n_query']} query itu sampel kecil.</b> Satu query berpindah peringkat menggeser
        rata-rata secara kasat mata. Perlakukan selisih kecil antar-konfigurasi sebagai setara.</li>
    <li><b>Satu wilayah saja</b> ({_esc(r['wilayah'])}). Wilayah lain belum punya eval set, dan
        cakupan vektornya berbeda.</li>
    <li><b>Relevansi biner.</b> Ground truth tidak membedakan "sangat relevan" dan "agak relevan",
        jadi nDCG di sini lebih kasar daripada nDCG bergradasi.</li>
    <li><b>Latensi diukur di mesin dev</b> terhadap Postgres lokal dan API eksternal — bukan angka
        produksi, hanya untuk membandingkan biaya antar-lapis.</li>
  </ul>
  </div>

  <div class="meta">
    <b>Reproduksibilitas.</b> {meta_txt}<br>
    Dihasilkan oleh <code>python -m eval.eval_rag</code>; data mentah berdampingan sebagai
    <code>.json</code> (termasuk skor per-topik, supaya selang &amp; uji di atas bisa dihitung ulang
    secara independen). Nilai konfigurasi di <code>.env.example</code>; arsitektur di
    <code>docs/ARSITEKTUR_RAG.md</code>.
  </div>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluasi retrieval RAG + laporan HTML mandiri.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "laporan_rag.html")
    ap.add_argument("--tanpa-rerank", action="store_true",
                    help="lewati konfigurasi ber-rerank (hemat kuota API provider)")
    ap.add_argument("--jeda", type=float, default=0.0,
                    help="jeda detik antar-query, utk menghindari rate limit provider")
    ap.add_argument("--eval-set", type=Path, default=None,
                    help="eval set yang dipakai (default tests/eval_set.jsonl; "
                         "pakai tests/eval_set_operasi.jsonl utk titik operasi produksi)")
    ap.add_argument("--lanjutkan", action="store_true",
                    help="lanjutkan run yang tumbang dari eval/_checkpoint_search.json, "
                         "tanpa membayar ulang panggilan API topik yang sudah dinilai")
    ap.add_argument("--dari-json", type=Path, default=None,
                    help="bangun ulang HTML dari hasil JSON yang sudah ada, TANPA memanggil "
                         "DB/API lagi — untuk mengubah tampilan laporan tanpa membakar kuota")
    args = ap.parse_args()

    if args.dari_json:
        print(f"[eval] bangun ulang dari {args.dari_json.name} (tanpa panggilan DB/API)")
        hasil = json.loads(args.dari_json.read_text(encoding="utf-8"))
    else:
        print(f"[eval] {(args.eval_set or _EVAL_SET).name} -> {args.out.name}")
        hasil = jalankan_evaluasi(pakai_rerank=not args.tanpa_rerank, jeda_s=args.jeda,
                                  eval_set=args.eval_set, lanjutkan=args.lanjutkan)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bangun_html(hasil), encoding="utf-8")
    if not args.dari_json:
        args.out.with_suffix(".json").write_text(
            json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[eval] ringkasan:")
    for k in hasil["konfigurasi"]:
        a = hasil["agregat"][k]
        print(f"  {k:14s} Hit@5={a['hit@5']:.0%}  MRR={a['mrr']:.3f}  "
              f"nDCG@5={a['ndcg@5']:.3f}  MAP={a['map']:.3f}  {hasil['latensi_ms'][k]:,.0f}ms")
    print(f"\n[eval] laporan -> {args.out}")


if __name__ == "__main__":
    main()
