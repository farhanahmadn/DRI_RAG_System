"""eval/statistik.py — selang kepercayaan & uji signifikansi untuk metrik evaluasi.

Kenapa perlu. Evaluasi retrieval pertama (commit `85abbda`) melaporkan angka telanjang: 22 topik,
satu kali jalan, tanpa ukuran ketidakpastian. Akibatnya selisih sebesar SATU topik dari 22 terbaca
seperti keunggulan — padahal tak bisa dibedakan dari derau. Untuk angka yang dipakai mendukung
pengajuan paten, tiap klaim keunggulan harus bisa menjawab "seberapa yakin?".

Dua hal yang disediakan modul ini:

1. **Bootstrap percentile CI** atas TOPIK. Jalur retrieval deterministik untuk indeks & query yang
   sama, jadi mengulang query yang sama menghasilkan angka identik — ketidakpastiannya datang dari
   **pencuplikan topik**, bukan derau antar-jalan. Bootstrap meresample topik, tepat memodelkan
   "bagaimana kalau kebetulan kita memilih 50 topik yang sedikit berbeda?".

2. **Wilcoxon signed-rank berpasangan.** Semua konfigurasi dinilai atas topik yang SAMA, jadi
   perbandingannya berpasangan. Dipilih non-parametrik karena skor per-topik (hit 0/1, nDCG, recall)
   jelas tidak normal — sebagian besar menumpuk di 0 dan 1. Ukuran efek dilaporkan sebagai
   rank-biserial correlation, supaya "signifikan" tidak tertukar dengan "besar".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SelangKepercayaan:
    rata: float
    bawah: float
    atas: float
    n: int

    def ringkas(self, persen: bool = False) -> str:
        if persen:
            return f"{self.rata:.0%} [{self.bawah:.0%}–{self.atas:.0%}]"
        return f"{self.rata:.3f} [{self.bawah:.3f}–{self.atas:.3f}]"


def bootstrap_ci(nilai: list[float], n_resample: int = 10_000, alpha: float = 0.05,
                 seed: int = 12345) -> SelangKepercayaan:
    """CI percentile dengan meresample TOPIK (bukan mengulang query).

    `seed` tetap supaya laporan bisa diproduksi ulang persis — syarat dasar reproduksibilitas.
    """
    n = len(nilai)
    if n == 0:
        return SelangKepercayaan(0.0, 0.0, 0.0, 0)
    rata = sum(nilai) / n
    if n == 1:
        return SelangKepercayaan(rata, rata, rata, 1)

    rng = random.Random(seed)
    rata_resample = []
    for _ in range(n_resample):
        total = 0.0
        for _ in range(n):
            total += nilai[rng.randrange(n)]
        rata_resample.append(total / n)
    rata_resample.sort()
    lo = rata_resample[int((alpha / 2) * n_resample)]
    hi = rata_resample[min(int((1 - alpha / 2) * n_resample), n_resample - 1)]
    return SelangKepercayaan(rata, lo, hi, n)


@dataclass(frozen=True)
class UjiBerpasangan:
    p: float
    n_beda: int            # jumlah topik yang skornya berbeda (yang seri dibuang, sesuai Wilcoxon)
    efek: float            # rank-biserial correlation, -1..1
    selisih_rata: float

    @property
    def signifikan(self) -> bool:
        return self.p < 0.05

    def ringkas(self) -> str:
        if self.n_beda == 0:
            return "identik di semua topik"
        arah = "unggul" if self.selisih_rata > 0 else "kalah"
        tanda = "signifikan" if self.signifikan else "TIDAK signifikan"
        return f"{arah} {self.selisih_rata:+.3f}, p={self.p:.3f} ({tanda}), efek={self.efek:+.2f}, n≠={self.n_beda}"


def wilcoxon_berpasangan(a: list[float], b: list[float]) -> UjiBerpasangan:
    """Wilcoxon signed-rank: apakah `a` berbeda dari `b` pada topik yang sama.

    Topik dengan selisih nol dibuang (konvensi Wilcoxon) — kalau SEMUA seri, tidak ada yang bisa
    diuji dan itu dilaporkan apa adanya, bukan dipaksakan jadi p-value.
    """
    if len(a) != len(b):
        raise ValueError(f"panjang tak sama: {len(a)} vs {len(b)}")
    selisih_semua = [x - y for x, y in zip(a, b)]
    selisih_rata = sum(selisih_semua) / len(selisih_semua) if selisih_semua else 0.0
    selisih = [d for d in selisih_semua if d != 0]
    if not selisih:
        return UjiBerpasangan(p=1.0, n_beda=0, efek=0.0, selisih_rata=0.0)

    try:
        from scipy.stats import wilcoxon as _wilcoxon

        hasil = _wilcoxon(selisih, alternative="two-sided", zero_method="wilcox")
        p = float(hasil.pvalue)
    except Exception:
        p = _p_normal_approx(selisih)

    # Rank-biserial: (jumlah rank positif - negatif) / total rank. Independen dari n, jadi bisa
    # dibandingkan antar-metrik.
    peringkat = _peringkat_rata(sorted(range(len(selisih)), key=lambda i: abs(selisih[i])), selisih)
    r_plus = sum(r for d, r in zip(selisih, peringkat) if d > 0)
    r_minus = sum(r for d, r in zip(selisih, peringkat) if d < 0)
    total = r_plus + r_minus
    efek = (r_plus - r_minus) / total if total else 0.0
    return UjiBerpasangan(p=p, n_beda=len(selisih), efek=efek, selisih_rata=selisih_rata)


def _peringkat_rata(urut_idx: list[int], selisih: list[float]) -> list[float]:
    """Peringkat |selisih| dengan rata-rata untuk nilai seri (tied ranks)."""
    peringkat = [0.0] * len(selisih)
    i = 0
    while i < len(urut_idx):
        j = i
        while j + 1 < len(urut_idx) and abs(selisih[urut_idx[j + 1]]) == abs(selisih[urut_idx[i]]):
            j += 1
        rata = (i + j) / 2 + 1
        for k in range(i, j + 1):
            peringkat[urut_idx[k]] = rata
        i = j + 1
    return peringkat


def _p_normal_approx(selisih: list[float]) -> float:
    """Cadangan kalau scipy tak tersedia — aproksimasi normal, tanpa koreksi kontinuitas."""
    n = len(selisih)
    peringkat = _peringkat_rata(sorted(range(n), key=lambda i: abs(selisih[i])), selisih)
    w = sum(r for d, r in zip(selisih, peringkat) if d > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return 1.0
    z = (w - mu) / sigma
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
