# Kontrak L3 untuk lahan multi-zona

Berlaku sejak 2026-10-20, atas permintaan BE: bila permohonan menyilang **2 zona RDTR atau lebih**, ITBX dan
intensitas dinilai dan dijelaskan **per zona**, sedangkan hidrologi tetap **satu** (digabung).

## Kapan jalur ini aktif

Hanya bila `gate_hukum.itbx_zones` berisi **minimal 2 entri**. Selain itu (lahan satu zona, atau BE tidak
mengirim `itbx_zones`) keluaran **tidak berubah sama sekali**.

## Masukan yang dipakai (L2)

| Field L2 | Dipakai untuk |
|---|---|
| `gate_hukum.itbx_zones[]` | satu poin ITBX per zona (status, kegiatan, `keterangan_ketentuan`, `dasar_hukum`, `reason`) |
| `gate_hukum.intensitas_zones[]` | satu poin intensitas per zona (`parameter`, ambang, `hierarki_jalan`, `kawasan_resapan`, `kode_zona` sub-zona seperti `P-1`, `R-3`) |
| `gate_hukum.keputusan_sistem_per_zona[]` | `status_keputusan` per zona untuk rekomendasi per zona |
| `gate_hukum.pemberitahuan` | dibawa apa adanya ke `catatan_global` |
| `impact_assessment` | tetap satu poin `dampak` (kawasan digabung) |

Pasangan zona ITBX dengan entri intensitasnya dicocokkan lewat **nama zona**, atau kode intensitas turunan kode ITBX
(`P-1` untuk `P`). Zona yang tidak punya entri intensitas, atau berstatus selain `MEMENUHI_SYARAT`/`MELAMPAUI_BATAS`,
menghasilkan poin intensitas berstatus `Tidak Dinilai`.

`tahapan.itbx` dan `tahapan.intensitas` (hasil agregat BE) **tidak dipakai** pada jalur multi-zona. Blok
`advisory` dari BE (pengaman bila AI mati) diabaikan.

## Keluaran (OutputL3)

```json
{
  "poin": [
    {"poin_id": "itbx_P", "kategori": "Klasifikasi Kegiatan (ITBX) - Zona Pertanian", "status": "I",
     "zona": {"nama_zona": "Zona Pertanian", "kode_zona": "P", "luas_m2": 623.55, "persentase": 50.4}, "...": "..."},
    {"poin_id": "intensitas_P", "kategori": "Intensitas Bangunan (KDB/KLB/KDH) - Zona Pertanian", "...": "..."},
    {"poin_id": "itbx_R", "...": "..."},
    {"poin_id": "intensitas_R", "...": "..."},
    {"poin_id": "dampak", "zona": null, "...": "satu untuk seluruh lahan"}
  ],
  "rekomendasi_per_zona": [
    {"nama_zona": "Zona Pertanian", "kode_zona": "P", "luas_m2": 623.55, "persentase": 50.4,
     "status_itbx": "I", "status_intensitas": "MEMENUHI_SYARAT", "status_keputusan": "LOLOS",
     "rekomendasi": "Setuju", "alasan": "...", "saran": "...", "poin_ids": ["itbx_P", "intensitas_P"],
     "low_confidence": false},
    {"nama_zona": "Zona Perumahan", "kode_zona": "R", "...": "..."}
  ]
}
```

- **`poin_id`** poin per zona berakhiran kode zona: `itbx_P`, `intensitas_P`, `itbx_R`, `intensitas_R`.
  Poin `dampak` tidak berakhiran apa pun. FE yang ingin mengelompokkan bisa memakai `zona.kode_zona`.
- **`PoinOutput.zona`** (baru, opsional) terisi hanya pada poin per zona; `null` pada `dampak` dan pada lahan satu zona.
- **`rekomendasi_per_zona`** (baru) berisi satu entri per zona, urutannya sama dengan `itbx_zones`. Daftar kosong
  `[]` untuk lahan satu zona.
  - `rekomendasi`: `Setuju` / `Setuju Bersyarat` / `Tidak Setuju`, diturunkan dari `status_keputusan` zona itu
    dengan aturan yang sama seperti gate keseluruhan (X -> Tidak Setuju; B/T/TB atau intensitas melampaui batas ->
    Setuju Bersyarat; selain itu Setuju). Dampak hidrologi tidak ikut, karena satu untuk seluruh lahan.
  - `alasan` dan `saran` dirangkai dari poin ITBX dan intensitas zona itu (tanpa panggilan LLM tambahan).
- **`rekomendasi_sistem`** (tingkat atas) tetap satu, dihitung dari `gate_hukum.final_gate_status` dan kategori dampak.
- **`narasi_rekomendasi.paragraf_gate_intensitas`** dan **`kesimpulan.langkah_berdampak`** menyebut tiap zona dengan
  namanya.
- **`catatan_global`** memuat `pemberitahuan` BE (zona apa saja dan luasnya).

## Hal lain yang berubah bersamaan

- Dampak berkategori **Rendah/Sedang** tidak lagi memakai `impact_assessment.rekomendasi_mitigasi`. BE sebelumnya
  mengirim blok itu juga untuk dampak Rendah dengan angka yang tidak bermakna (mis. lantai dasar maks 10.960,79 m²,
  RTH min 0 m², dimensi sumur 0 m³). Catatan "batas intensitas lebih ketat daripada syarat dampak" hanya dibuat bila
  dampaknya memang butuh mitigasi (Tinggi/Sangat Tinggi).

## Permintaan ke BE

1. Untuk dampak Rendah, `rekomendasi_mitigasi` sebaiknya `null` (atau tidak dikirim).
2. Bila satu zona berstatus ITBX `X`, mohon `intensitas_zones` zona itu tetap dikirim dengan `status` yang
   jelas (mis. `TIDAK_DIEVALUASI`). Tanpa entri, poin intensitas zona itu `Tidak Dinilai`.
3. Konfirmasi dasar angka P-1: KLB maksimum 1 (BE) vs 0,1 (Lampiran VI RDTR, resapan lokal).
