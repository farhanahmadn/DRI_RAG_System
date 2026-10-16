# Permintaan Penambahan Field Output L2 untuk AI Reasoning (L3)

Tanggal: 2026-10-15
Dari: tim AI Reasoning (L3)
Untuk: tim Back-End (L2 Spatial Risk Assessment)

## Latar belakang

Dari hasil uji L3 dengan payload L2 nyata dan payload uji, ada beberapa jawaban yang tidak bisa kami buat benar
karena informasinya tidak ada di payload. Dua contoh paling jelas:

1. **Status kawasan resapan air.** RDTR Sleman Tengah (Perbup 80/2023) Pasal 53 menetapkan ketentuan khusus
   kawasan resapan air **per blok**, bukan per zona. Payload tidak membawa blok, jadi L3 hanya bisa menebak dari
   kode zona. Akibatnya lokasi non-resapan ikut dijawab dengan pasal kawasan resapan.
2. **Zona pemilik ambang intensitas.** Pada lahan yang menyilang beberapa zona, `lokasi.rdtr_zone` (mis. Zona
   Pertanian) bukan zona yang ambangnya dipakai di `tahapan.intensitas` (mis. Zona Perumahan). L3 sekarang
   menebaknya dengan mencocokkan `intensitas_zones`.

Semua field di bawah **opsional**. Bila tidak dikirim, L3 berperilaku seperti sebelumnya.

## Prioritas 1 (paling berdampak)

### 1. `lokasi.blok` dan `lokasi.swp`

Kode blok RDTR tempat lokasi berada, sesuai penamaan RDTR (contoh `"I.B.2"` untuk SWP B).

```json
"lokasi": { "swp": "B", "blok": "I.B.2", "blok_list": ["I.B.2"] }
```

- `blok`: blok dominan (luas irisan terbesar). `blok_list`: semua blok yang beririsan, bila poligon menyilang
  lebih dari satu blok.
- Kegunaan: RDTR menyebut blok secara eksplisit di banyak ketentuan khusus. Pasal 53 ayat (1) mendaftar blok
  kawasan resapan air (SWP B Blok I.B.1 sampai I.B.6, SWP C Blok I.C.1, 2, 3, 5, 6, 7, dan SWP F Blok I.F.3).
  Pasal 50 ayat (2), (5), (7), dan (10) mendaftar blok kawasan rawan bencana (gempa bumi, gunung api, banjir
  lahar, kekeringan). Dengan ID blok, L3 dapat menentukan sendiri ketentuan mana yang berlaku, langsung dari
  teks peraturan.

### 2. `lokasi.ketentuan_khusus` (hasil overlay peta Lampiran VII)

Status setiap ketentuan khusus pada Pasal 47 RDTR: nilai `true`, `false`, atau `null` bila tidak diketahui.

```json
"lokasi": {
  "ketentuan_khusus": {
    "kawasan_resapan_air": false,
    "rawan_kekeringan": false,
    "rawan_gempa_bumi": true,
    "rawan_gunung_api": false,
    "rawan_banjir_lahar": false,
    "cagar_budaya": false,
    "sempadan": false,
    "kkop": false,
    "hankam": false,
    "tempat_evakuasi": false
  }
}
```

- **Minimal yang kami butuhkan: `kawasan_resapan_air`.** Untuk sementara L3 sudah menerima boolean tunggal
  `lokasi.kawasan_resapan` (nama usulan sementara, sudah aktif di sisi kami). Mohon konfirmasi apakah BE lebih
  suka nama itu atau struktur `ketentuan_khusus` di atas. Kami bisa mengikuti salah satunya.
- Sumber yang kami harapkan: peta Ketentuan Khusus Kawasan Resapan Air (Lampiran VII.F) atau hasil overlay BE
  yang setara.
- Catatan: `tahapan.lp2b` dan `tahapan.sempadan` sudah ada di payload dan sejalan dengan kategori ini.

### 3. Dasar pemilihan baris ambang pada `tahapan.intensitas`

Ambang KDB, KLB, dan KDH di Lampiran VI dipilih berdasarkan zona, status resapan, dan hierarki jalan (mis. R-3:
resapan 50/3,0/30 versus non-resapan 60/4,8/25). Mohon sertakan dasar yang BE pakai:

```json
"intensitas": {
  "nama_zona": "Zona Perumahan",
  "kode_zona": "R-3",
  "hierarki_jalan": "lokal",
  "kawasan_resapan": false
}
```

- Kegunaan: L3 dapat menyebut dasar ambang secara tepat (zona mana, jalan apa, resapan atau tidak) dan tidak lagi
  menebak zona pemilik ambang pada lahan multi-zona.

## Prioritas 2

### 4. `lokasi.rdtr_subzone` lebih konsisten

Pada log kami, sebagian besar payload (kurang lebih 96%) tidak mengirim `rdtr_subzone`. Tanpa sub-zona, L3 jatuh
ke tabel tingkat keluarga zona dan harus menambahkan peringatan bahwa ambang perlu diverifikasi. Mohon diisi
sedapat mungkin, dan tetap `null` bila BE tidak yakin.

### 5. Konsistensi `rekomendasi_mitigasi` dengan ketentuan intensitas

Contoh payload uji: syarat penurunan dampak (`rekomendasi_penyesuaian_lahan`) menyarankan luas bangunan maksimal
1.077 m² (KDB 75%), sedangkan ketentuan intensitas membatasi 861,6 m² (KDB 60%). Keduanya benar menurut
aturannya masing-masing, tetapi pemohon akan bingung. Saat ini L3 menambahkan catatan "ikuti yang lebih ketat".
Pertanyaan untuk BE:

- Apakah angka mitigasi sebaiknya sudah dibatasi oleh ambang intensitas, atau cukup ditambah penanda
  (mis. `melampaui_batas_intensitas: true`)?
- Mohon jelaskan makna `dimensi_minimum` (rumus dengan faktor 0,013659). Apakah itu kapasitas sumur resapan, dan
  apakah berlaku juga untuk lokasi di luar kawasan resapan? Dokumen RDTR mendorong sumur resapan di area
  terbangun, tetapi tidak menetapkan rumus dimensi.

## Prioritas 3 (kontrak operasional, untuk didiskusikan)

### 6. Penanganan error dan waktu tunggu

- Saat LLM atau basis pengetahuan L3 bermasalah, L3 saat ini tetap membalas HTTP 200 dengan poin berstatus
  `low_confidence` dan teks "peninjauan manual". Kami mempertimbangkan membalas **HTTP 503** untuk kondisi
  infrastruktur seperti itu. Mohon konfirmasi apakah BE dapat menangani 503 dengan mengulang (retry dengan
  backoff) dan tidak menyimpannya sebagai hasil akhir.
- Satu panggilan L3 memanggil LLM beberapa kali. Karena Cloudflare memutus permintaan yang tidak dibalas
  dalam 100 detik, L3 kini menjamin membalas dalam sekitar 85 detik (bisa diatur lewat `REASONING_DEADLINE_S`).
  Bila waktu habis, poin yang belum selesai dikembalikan dengan `low_confidence: true` dan `catatan_global`
  memuat kalimat batas waktu. Mohon timeout klien dari BE ke L3 diatur sekitar 100 detik (jangan lebih
  pendek dari 90 detik), dan perlakukan hasil dengan `low_confidence_keseluruhan: true` sebagai hasil
  parsial yang boleh diulang.

## Ringkasan

| # | Field | Prioritas | Wajib? |
|---|---|---|---|
| 1 | `lokasi.swp`, `lokasi.blok`, `lokasi.blok_list` | 1 | Opsional |
| 2 | `lokasi.ketentuan_khusus` (minimal `kawasan_resapan_air`) atau `lokasi.kawasan_resapan` | 1 | Opsional |
| 3 | `tahapan.intensitas.{nama_zona, kode_zona, hierarki_jalan, kawasan_resapan}` | 1 | Opsional |
| 4 | `lokasi.rdtr_subzone` lebih sering terisi | 2 | Sudah ada |
| 5 | Konsistensi mitigasi dengan intensitas, dan makna `dimensi_minimum` | 2 | Diskusi |
| 6 | Penanganan 503 dan timeout | 3 | Diskusi |
