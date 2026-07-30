# Blueprint Adaptasi Reasoning Engine → Model L2 (Gate + Impact)
### Pre-Check Izin Bangunan Sleman · komponen kita = "L3 Advisory"

Cetak biru untuk mengubah reasoning engine dari model lama (8 indikator komposit) ke model L2 revisi: **Gate Hukum (2 tahap) + Skor Dampak Runoff**. Bukan rebuild — penyederhanaan + restrukturisasi I/O.

---

## 1. Model baru (ringkas)

- **Gate = 2 tahap short-circuit:** ITBX (kegiatan) → Intensitas (KDB/KLB/KDH). **LP2B, Sempadan, Banjir DIHAPUS** dari gate.
- **Outcome gate:** hanya **X → Tidak Lolos**. `I`+intensitas patuh → **Lolos**. `I`+pelanggaran intensitas atau `T/B/TB` → **Lolos Bersyarat**.
- **Impact = kelas tata guna lahan**, indeks = C_sesudah/C_sebelum, **skor INVERS (tinggi = dampak rendah)**, advisory (tak menggagalkan gate).
- **Peran kita (L3):** konsumsi status gate + skor dampak → hasilkan reasoning + sitasi + rekomendasi + **turunkan `rekomendasi_sistem`**.

---

## 2. Skema Input (`schemas.py` baru) — Pydantic

```python
from typing import Literal, Optional
from pydantic import BaseModel

class DasarHukum(BaseModel):
    dokumen: str; pasal: Optional[str] = None
    regulation_ref: Optional[str] = None; url: Optional[str] = None
    kutipan: Optional[str] = None

class Lokasi(BaseModel):
    koordinat: Optional[dict] = None
    rdtr_zone: Optional[str] = None; kode_sub_zona: Optional[str] = None  # mis. "R-3"
    luas_lahan_m2: Optional[float] = None

# ---- Gate ----
class TahapITBX(BaseModel):
    status: Literal["I","T","B","TB","X"]
    lolos: bool
    kbli_diusulkan: Optional[str] = None
    kegiatan_diusulkan: Optional[str] = None
    kegiatan_diizinkan: list[str] = []
    kegiatan_terbatas: list[str] = []
    kegiatan_bersyarat: list[str] = []
    kegiatan_terbatas_bersyarat: list[str] = []
    keterangan_ketentuan: list[str] = []      # sumber sitasi ITBX
    reason: Optional[str] = None              # gateBreakdown back-end (fakta)
    dasar_hukum: list[DasarHukum] = []
    # nuansa: deteksi fallback data-kosong & makna X (dari reason)

class ParamIntensitas(BaseModel):
    usulan: Optional[float] = None
    ambang_maks: Optional[float] = None       # KDB/KLB
    ambang_min: Optional[float] = None         # KDH
    memenuhi: Optional[bool] = None
    satuan: Optional[str] = None               # "persen" | "rasio"

class TahapIntensitas(BaseModel):
    status: Literal["MEMENUHI_SYARAT","MELAMPAUI_BATAS"]
    lolos: bool
    parameter: dict[str, ParamIntensitas]      # {"kdb","klb","kdh"}
    luas_tapak_m2: Optional[float] = None
    jumlah_lantai: Optional[int] = None
    luas_rth_usulan_m2: Optional[float] = None
    reason: Optional[str] = None
    dasar_hukum: list[DasarHukum] = []

class GateHukum(BaseModel):
    final_gate_status: Literal["Lolos","Lolos Bersyarat","Tidak Lolos"]
    decisive_stage: Optional[str] = None
    tahapan: dict                              # {"itbx": TahapITBX, "intensitas": TahapIntensitas}

# ---- Impact ----
class ImpactAssessment(BaseModel):
    dinilai: bool
    c_before: Optional[float] = None; c_after: Optional[float] = None
    runoff_change_index: Optional[float] = None
    impact_score: Optional[float] = None       # INVERS: tinggi = dampak rendah
    impact_category: Optional[str] = None       # Rendah|Sedang|Tinggi|Sangat Tinggi
    threshold_bands: Optional[dict] = None
    existing_surface_details: Optional[dict] = None
    proposed_surface_details: Optional[dict] = None
    c_coefficients: Optional[dict] = None
    luas_lahan_m2: Optional[float] = None
    data_confidence: Optional[str] = None
    limitations: Optional[str] = None

class Meta(BaseModel):
    data_confidence_keseluruhan: Optional[str] = None
    caveats: list[str] = []

class L2Assessment(BaseModel):                 # INPUT utuh dari back-end
    application_id: int
    application_number: Optional[str] = None
    timestamp: Optional[str] = None
    lokasi: Lokasi
    gate_hukum: GateHukum
    impact_assessment: ImpactAssessment
    meta: Meta = Meta()
```

---

## 3. Penurunan `rekomendasi_sistem` (fungsi DETERMINISTIK — logika kita)

Karena ini bagian kita, kami turunkan dari kombinasi gate × dampak (kode, bukan LLM):

```python
def turunkan_rekomendasi(gate_status: str, impact_category: str | None) -> str:
    if gate_status == "Tidak Lolos":
        return "Tidak Setuju"                       # X = pelanggaran hukum absolut
    impact_tinggi = impact_category in ("Tinggi", "Sangat Tinggi")
    if gate_status == "Lolos Bersyarat" or impact_tinggi:
        return "Setuju Bersyarat"                   # ada syarat/mitigasi
    return "Setuju"                                 # Lolos + dampak Rendah/Sedang
```

| Gate | Dampak | → rekomendasi_sistem |
|---|---|---|
| Tidak Lolos | apa pun | **Tidak Setuju** |
| Lolos Bersyarat | apa pun | **Setuju Bersyarat** |
| Lolos | Rendah/Sedang | **Setuju** |
| Lolos | Tinggi/Sangat Tinggi | **Setuju Bersyarat** (mitigasi runoff) |

*(Fungsi deterministik + unit test. KONFIRMASI aturan ini ke tim sebelum dikunci.)*

---

## 4. Adapter: Gate + Impact → `poin[]`

Petakan tiap aspek jadi satu poin reasoning:

| Poin | Sumber | Tipe rekomendasi | Catatan |
|---|---|---|---|
| **Kesesuaian Kegiatan (ITBX)** | `tahapan.itbx` | **kategorikal** (X → sarankan kegiatan diizinkan/terbatas/bersyarat dari back-end) | X = absolut (tak bisa ditweak); T/B/TB → tampilkan syarat dari `keterangan_ketentuan` |
| **Intensitas Ruang (KDB/KLB/KDH)** | `tahapan.intensitas` | **numerik** (hitung target dari `usulan` vs `ambang`) | MELAMPAUI_BATAS → Lolos Bersyarat, bisa diperbaiki |
| **Dampak Runoff** | `impact_assessment` | **numerik-mitigasi** (KDB↓, KDH↑, sumur resapan, kolam retensi) | skor INVERS; advisory |

Ringkasan output = **dua jalur**: (a) status gate (Lolos/Bersyarat/Tidak Lolos) + (b) skor/kategori dampak + `rekomendasi_sistem`. **Tanpa composite/level lama.**

---

## 5. Aturan Faithfulness KHUSUS model baru (untuk generator + guardrail)

1. **Skor dampak INVERS** — skor tinggi = dampak **RENDAH**. Reasoning tak boleh membalik ("skor 70 = risiko tinggi" ❌ → "dampak Sedang" ✓). **Guardrail: cek arah kategori vs skor.**
2. **ITBX fallback data-kosong** — bila `reason` menandakan "kolom matriks RDTR kosong/otomatis", status `I` itu **bukan kepatuhan sebenarnya** → set `low_confidence=True` + caveat eksplisit ("diloloskan otomatis karena data RDTR kosong"). Jangan klaim "kegiatan sesuai".
3. **Makna `X` ganda** — "dilarang" vs "tidak ditemukan di matriks" vs "di luar area RDTR". Ikuti `reason` back-end; jangan selalu bilang "dilarang".
4. **Caveat wajib diteruskan** — `meta.caveats` + `data_confidence` muncul di output, tidak disembunyikan.
5. **Fakta dari back-end** (`reason`/`gateBreakdown`, status, `pelanggaranDetail`) dikonsumsi sebagai **ground truth** lalu diperkaya; angka & verdict tak dihitung ulang LLM. `rekomendasi_sistem` dari fungsi §3 (kode).
6. **Sitasi hierarki:** `keterangan_ketentuan` + `dasar_hukum` back-end = anchor; **RAG** untuk grounding pasal RDTR & basis mitigasi. Guardrail: sitasi tak boleh kontradiksi anchor.

---

## 6. Yang DIBUANG dari kode saat ini

Hapus poin/handler/mock/test untuk indikator yang tak ada lagi:
- **LP2B, Sempadan, Resapan, Banjir** (sebagai indikator/poin).
- `kegiatan_data.py` (stopgap) — daftar kegiatan kini dari back-end (`kegiatan_diizinkan/terbatas/bersyarat`).
- Logika composite `skor_total`/level 0–100.
- Fakta spasial lama (`fakta_spasial{in_lp2b, banjir, resapan, ...}`) — diganti struktur gate.

## 7. Yang DIPAKAI ULANG (inti mesin selamat)

- `generator.py` (pola fact-injection — tinggal fakta baru).
- `guardrail.py` (+ tambah cek invers-skor & konsistensi caveat).
- `assemble.py` (restruktur output dua-jalur).
- `api/`, `llm_client.py`, retrieval/RAG, logging.

---

## 8. Langkah migrasi (untuk Claude Code, urut)

1. **Tulis ulang `schemas.py`** (§2) + `output` dua-jalur (ringkasan_gate, ringkasan_dampak, poin[], rekomendasi_sistem, kesimpulan). Test round-trip pakai 2 JSON contoh back-end.
2. **`adapter.py` baru** — `L2Assessment` → daftar poin internal (§4) + panggil `turunkan_rekomendasi` (§3).
3. **Sesuaikan `calculator.py`** — target intensitas dari `usulan`/`ambang`; mitigasi dampak (KDB/KDH → arah skor). Buang formula indikator lama.
4. **Sesuaikan `prompts.py`/`generator.py`** — fakta baru (status gate, pelanggaranDetail, kategori dampak invers, caveat). Hapus prompt indikator lama.
5. **Perkuat `guardrail.py`** — cek invers-skor, low_confidence fallback ITBX, konsistensi caveat.
6. **`assemble.py`** — rakit output dua-jalur + `rekomendasi_sistem`.
7. **Perbarui `eval/gold_set`** — kasus baru: ITBX X (tolak), T/B/TB (bersyarat), intensitas melampaui (bersyarat+numerik), dampak Tinggi (mitigasi), ITBX fallback-kosong (low_confidence). Jalankan `run_eval`.
8. Hapus test/kode indikator lama; pastikan hijau.

---

## 9. Butuh konfirmasi terakhir
- **Aturan `turunkan_rekomendasi` (§3)** — setujui matriksnya?
- **Enum casing** `impact_category` ("SEDANG" vs "Sedang") — normalkan di adapter.
- **`impact_score` band final** (masih "ilustratif" di dokumen) — pakai apa adanya dari back-end.

*Prinsip tetap: faithful dulu; angka & verdict dari back-end/kode; LLM hanya pelapis; skor dampak INVERS jangan dibalik.*
