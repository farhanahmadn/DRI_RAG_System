import pytest
from pydantic import ValidationError

from app.schemas import (
    FaktaSpasial,
    IndikatorJejak,
    JejakAturanRequest,
    KesimpulanOutput,
    OutputPreCheck,
    PoinOutput,
    RekomendasiOutput,
    RingkasanOutput,
    SitasiOutput,
)


def test_jejak_aturan_request_valid_lp2b():
    request = JejakAturanRequest(
        skor_total=35.0,
        zona="LP2B",
        indikator=[
            IndikatorJejak(
                poin_id="LP2B-01",
                kategori="Lokasional",
                bobot=20.0,
                skor=20.0,
                kontribusi=20.0,
                nilai_input="dalam_lp2b",
                ambang="tidak_dalam_lp2b",
                operator="==",
                formula="in_lp2b == True",
                zona="LP2B",
                referensi_hukum=["UU No. 41 Tahun 2009 Pasal 44"],
                fakta_spasial=FaktaSpasial(in_lp2b=True, banjir=False, resapan=False),
                target_rekomendasi=None,
            )
        ],
    )

    dumped = request.model_dump()
    restored = JejakAturanRequest.model_validate(dumped)
    assert restored == request
    assert restored.indikator[0].fakta_spasial.in_lp2b is True


def test_output_pre_check_valid_roundtrip():
    output = OutputPreCheck(
        ringkasan=RingkasanOutput(skor_total=35.0, level="Tinggi", kalimat="Risiko tinggi karena lokasi berada di LP2B."),
        poin=[
            PoinOutput(
                poin_id="LP2B-01",
                kategori="Lokasional",
                status="melanggar",
                kontribusi=20.0,
                reasoning_pendek="Lokasi berada di kawasan LP2B.",
                reasoning_panjang="Berdasarkan jejak aturan, lokasi terindikasi berada di dalam LP2B sehingga dilarang dialihfungsikan kecuali memenuhi syarat kepentingan umum.",
                sitasi=[
                    SitasiOutput(
                        citation_id="c1",
                        dokumen="UU No. 41 Tahun 2009",
                        pasal="44",
                        halaman=21,
                        kutipan="Lahan Pertanian Pangan Berkelanjutan yang sudah ditetapkan dilarang dialihfungsikan.",
                        terverifikasi=True,
                    )
                ],
                rekomendasi=RekomendasiOutput(
                    tipe="lokasional",
                    target=None,
                    saran="Ajukan kajian kelayakan strategis bila ingin melanjutkan.",
                    disclaimer="Keputusan akhir ada pada pemerintah daerah.",
                ),
            )
        ],
        kesimpulan=KesimpulanOutput(
            langkah_berdampak=["Konsultasi dengan dinas terkait status LP2B."],
            catatan_lokasi="Lokasi berada di kawasan LP2B Sleman.",
        ),
    )

    dumped = output.model_dump()
    restored = OutputPreCheck.model_validate(dumped)
    assert restored == output


def test_rekomendasi_tipe_invalid_raises():
    with pytest.raises(ValidationError):
        RekomendasiOutput(tipe="bukan_tipe_valid", saran="x")
