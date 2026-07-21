# Integrasi Back-End (B2G) — Kontrak `/reasoning`

Dokumen ini untuk tim back-end (rule engine) yang akan MENGIRIM jejak aturan ke komponen AI
Reasoning ini dan MENERIMA kembali JSON siap-render untuk web. Semua field di bawah diambil
langsung dari `app/schemas.py` — tidak ada field yang dikarang.

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/reasoning` | Kirim jejak aturan, terima `OutputPreCheck` |
| `GET` | `/health` | Liveness check — `{"status": "ok"}`, tidak dibatasi rate limit |

**Rate limit**: `/reasoning` dibatasi 10 permintaan/menit per-IP (default, lihat
`RATE_LIMIT_PER_MENIT` di `.env`). Lebihi batas → HTTP `429` dengan body
`{"detail": "Terlalu banyak permintaan. Maksimum 10 per 60 detik."}`.

**Error tak terduga** (bug internal, dsb) → HTTP `500` dengan body
`{"error": "internal_error", "message": "..."}` — TIDAK PERNAH membocorkan stack trace.

**Input tidak valid** (field wajib hilang/salah tipe) → HTTP `422` (validasi Pydantic bawaan
FastAPI, format standar `{"detail": [...]}`).

## Skema Request — `JejakAturanRequest`

```
JejakAturanRequest
├── skor_total: float                    (WAJIB)
├── level: str | None                    (opsional — lihat "Item Kontrak #2")
├── zona: str | None                     (opsional, informasi umum lokasi)
└── indikator: list[IndikatorJejak]      (WAJIB, minimal 1)

IndikatorJejak (per poin/indikator)
├── poin_id: str                         (WAJIB — jadi PoinOutput.poin_id di respons)
├── kategori: str                        (WAJIB — nama indikator, mis. "KDB", "Lokasional LP2B")
├── bobot: float                         (WAJIB)
├── skor: float                          (WAJIB — skor==0 berarti "Aman", tidak melalui LLM sama sekali)
├── kontribusi: float                    (WAJIB — dipakai apa adanya di PoinOutput.kontribusi)
├── nilai_input: float | str             (WAJIB — nilai aktual yang dibandingkan)
├── ambang: float | str                  (WAJIB — nilai batas/threshold)
├── operator: str                        (WAJIB — "<=", ">=", "==", dst.)
├── formula: str                         (WAJIB, boleh string kosong "")
├── zona: str | None                     (opsional)
├── luas_lahan: float | None             (opsional — dipakai kalkulator utk indikator numerik KDB/KLB/KDH)
├── referensi_hukum: list[str]           (default [] — lihat "Item Kontrak #3")
├── fakta_spasial: FaktaSpasial | None   (opsional — lihat "Item Kontrak #1")
└── target_rekomendasi: dict | None      (opsional — kalau back-end SUDAH hitung target numerik, dipakai apa adanya, TIDAK dihitung ulang)

FaktaSpasial (semua field opsional, isi HANYA yang relevan)
├── in_lp2b: bool | None
├── banjir: bool | None
├── tingkat_banjir: "Tinggi" | "Sedang" | "Rendah" | None
├── resapan: bool | None
├── in_sempadan: bool | None
├── jarak_sungai_m: float | None
├── nama_sungai: str | None
└── arah: str | None
```

## Contoh Request LENGKAP (bukan karangan — dari `eval/gold_set.jsonl`, kasus `lp2b_berisiko`)

```json
{
  "skor_total": 20.0,
  "level": "Tinggi",
  "indikator": [
    {
      "poin_id": "LP2B-01",
      "kategori": "Lokasional LP2B",
      "bobot": 20.0,
      "skor": 100.0,
      "kontribusi": 20.0,
      "nilai_input": "dalam_lp2b",
      "ambang": "tidak_dalam_lp2b",
      "operator": "==",
      "formula": "in_lp2b == True",
      "zona": "LP2B",
      "referensi_hukum": ["UU No. 41 Tahun 2009 Pasal 44"],
      "fakta_spasial": {"in_lp2b": true, "banjir": false, "resapan": false}
    }
  ]
}
```

## Contoh Response NYATA (hasil panggilan live sesungguhnya ke request di atas, bukan mock)

```json
{
  "ringkasan": {
    "skor_total": 20.0,
    "level": "Tinggi",
    "kalimat": "Ditemukan 1 dari 1 indikator berisiko: Lokasional LP2B. Skor risiko total: 20.0."
  },
  "poin": [
    {
      "poin_id": "LP2B-01",
      "kategori": "Lokasional LP2B",
      "status": "Tidak Aman",
      "kontribusi": 20.0,
      "reasoning_pendek": "Lokasi berada dalam zona LP2B sehingga perlu dilindungi dan dilarang dialihfungsikan tanpa izin. Hal ini berdasarkan peraturan yang melindungi lahan pertanian pangan berkelanjutan.",
      "reasoning_panjang": "Lokasi yang berada dalam zona LP2B memiliki skor risiko yang tinggi karena peraturan yang melindungi lahan pertanian pangan berkelanjutan. Menurut peraturan, lahan pertanian pangan berkelanjutan yang sudah ditetapkan dilarang dialihfungsikan kecuali untuk kepentingan umum dengan syarat tertentu. Oleh karena itu, perlu dilakukan kajian kelayakan strategis dan penyediaan lahan pengganti jika ingin melakukan alih fungsi lahan.",
      "sitasi": [
        {
          "citation_id": "uu41-2009-p44",
          "dokumen": "UU No. 41 Tahun 2009 tentang Perlindungan Lahan Pertanian Pangan Berkelanjutan",
          "pasal": "44",
          "halaman": 21,
          "kutipan": "Pasal 44: Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilindungi dan dilarang dialihfungsikan, kecuali untuk kepentingan umum dengan syarat tertentu",
          "terverifikasi": true
        }
      ],
      "rekomendasi": {
        "tipe": "lokasional",
        "target": null,
        "saran": "Perlu dilakukan kajian kelayakan strategis dan penyediaan lahan pengganti jika ingin melakukan alih fungsi lahan di zona LP2B. Selain itu, perlu mempertimbangkan kepentingan umum dan mematuhi peraturan yang berlaku.",
        "disclaimer": null
      },
      "low_confidence": false
    }
  ],
  "kesimpulan": {
    "langkah_berdampak": [
      "Lokasional LP2B: Perlu dilakukan kajian kelayakan strategis dan penyediaan lahan pengganti jika ingin melakukan alih fungsi lahan di zona LP2B. Selain itu, perlu mempertimbangkan kepentingan umum dan mematuhi peraturan yang berlaku."
    ],
    "catatan_lokasi": "Lokasi berkaitan dengan faktor risiko lokasional: Lokasional LP2B."
  }
}
```
(Contoh di atas dipangkas 2 sitasi tambahan — `uu41-2009-p44-a1`, `uu41-2009-p44-a2` — supaya
ringkas; bentuk field sama persis.)

Field `rekomendasi.tipe` selalu salah satu dari `"numerik"` (KDB/KLB/KDH — `target` diisi angka),
`"kegiatan"` (klasifikasi I/T/B/X), atau `"lokasional"` (LP2B/Banjir/Resapan/Sempadan —
`target` SELALU `null`, karena faktor lokasional tidak bisa "ditawar" jadi angka).

## 3 Item Kontrak — WAJIB Disediakan Back-End

### 1. `fakta_spasial` (fakta GIS)
Reasoning **HANYA** boleh menyebut fakta spasial yang benar-benar ada di objek ini (aturan
sistem-prompt eksplisit: "JANGAN mengarang detail geometri, koordinat, atau kondisi lokasi lain
yang tidak ada di data"). Kalau back-end tidak kirim `fakta_spasial` sama sekali (atau field-nya
kosong), reasoning tetap jalan tapi TANPA konteks spasial tambahan — hanya berdasarkan
`nilai_input`/`ambang`/`kategori`. Isi HANYA field yang relevan untuk indikator itu (jangan kirim
semua field sekaligus kalau tidak relevan).

### 2. `skor_total` + `level`
Ini **komposit dari rule engine kalian**, BUKAN dihitung ulang oleh AI reasoning (Prinsip 3
arsitektur: "semua ANGKA berasal dari kode/back-end, BUKAN dari LLM"). `skor_total` WAJIB diisi.
**`level` sangat disarankan diisi** — kalau kosong, `ringkasan.level` di respons akan berisi
sentinel eksplisit `"BELUM_DITENTUKAN (menunggu level dari back-end)"` (bukan angka karangan AI),
supaya jelas terlihat data belum lengkap, bukan disamarkan.

### 3. `referensi_hukum` (per indikator)
List string rujukan (mis. `"UU No. 41 Tahun 2009 Pasal 44"`, `"RDTR Lampiran VI"`) yang dipakai
untuk mencari pasal/tabel yang relevan via `retriever.get_by_reference(...)`. Kalau kosong ATAU
tidak ada yang cocok di korpus RAG, `sitasi` di respons akan **kosong** (`[]`) — sistem TIDAK
PERNAH mengarang nomor pasal untuk mengisi kekosongan ini (prinsip Grounded). Semakin akurat
`referensi_hukum` (nama dokumen + nomor pasal), semakin besar kemungkinan sitasi terisi.

## Pakai OpenAPI (`/docs`, Swagger, Postman)

1. Jalankan server: `uvicorn app.api.main:app --reload` (lihat `README.md` root).
2. **Cara tercepat**: buka `http://127.0.0.1:8000/docs` di browser — Swagger UI bawaan FastAPI,
   bisa langsung coba `/reasoning` dari situ (klik "Try it out").
3. **editor.swagger.io**: ambil skema mentah dari `http://127.0.0.1:8000/openapi.json`
   (JSON), copy-paste isinya ke https://editor.swagger.io/ (menu File → Paste JSON/Import) untuk
   lihat/generate client dari skema tanpa perlu server jalan terus-menerus.
4. **Postman**: import `http://127.0.0.1:8000/openapi.json` langsung via
   "Import → Link" di Postman — semua endpoint & skema request/response otomatis jadi collection.

## Catatan

- Retriever yang dipakai saat ini masih `MockRetriever` (data dummy, lihat
  `docs/INTEGRASI_RETRIEVER.md`) — sitasi yang keluar HANYA akan mengutip dari korpus dummy itu
  sampai RAG asli (punya tim lain) terpasang. Kontrak `/reasoning` sendiri TIDAK berubah saat
  RAG asli dipasang.
- Data yang dikirim ke endpoint ini saat dev **HARUS data uji/dummy**, bukan PII warga asli —
  Groq (LLM provider saat ini) server-nya di luar negeri (lihat CLAUDE.md § Residensi data).
