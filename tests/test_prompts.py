from app.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import Chunk
from app.schemas import FaktaSpasial, IndikatorJejak


def _indikator(**overrides) -> IndikatorJejak:
    defaults = dict(
        poin_id="LP2B-01",
        kategori="Lokasional LP2B",
        bobot=20.0,
        skor=20.0,
        kontribusi=20.0,
        nilai_input="dalam_lp2b",
        ambang="tidak_dalam_lp2b",
        operator="==",
        formula="in_lp2b == True",
        zona="LP2B",
        referensi_hukum=["UU No. 41 Tahun 2009 Pasal 44"],
    )
    defaults.update(overrides)
    return IndikatorJejak(**defaults)


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="uu41-2009-p44",
        level="pasal",
        teks="Pasal 44: dilarang dialihfungsikan.",
        dokumen="UU No. 41 Tahun 2009",
        pasal="44",
        halaman=21,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_system_prompt_memuat_semua_aturan_wajib():
    assert "FINAL" in SYSTEM_PROMPT
    assert "fakta_spasial" in SYSTEM_PROMPT
    assert "citation_id" in SYSTEM_PROMPT
    assert "JANGAN mengarang" in SYSTEM_PROMPT
    assert "pelanggaran hukum" in SYSTEM_PROMPT
    assert "risiko alam" in SYSTEM_PROMPT
    assert "warga awam" in SYSTEM_PROMPT
    assert "STATUS" in SYSTEM_PROMPT
    assert "FAKTA hasil perhitungan kode" in SYSTEM_PROMPT
    assert "KLASIFIKASI" in SYSTEM_PROMPT
    assert "Kegiatan Diizinkan di Zona Ini" in SYSTEM_PROMPT


def test_build_user_prompt_memuat_ringkasan_jejak():
    indikator = _indikator()
    prompt = build_user_prompt(indikator, [], None)

    assert "LP2B-01" in prompt
    assert "Lokasional LP2B" in prompt
    assert "20.0" in prompt
    assert "dalam_lp2b" in prompt
    assert "tidak_dalam_lp2b" in prompt
    assert "==" in prompt
    assert "in_lp2b == True" in prompt


def test_build_user_prompt_hanya_fakta_spasial_yang_tidak_none():
    indikator = _indikator(
        fakta_spasial=FaktaSpasial(in_lp2b=True, banjir=False, resapan=None, nama_sungai=None)
    )
    prompt = build_user_prompt(indikator, [], None)

    assert "in_lp2b: True" in prompt
    assert "banjir: False" in prompt
    assert "resapan" not in prompt
    assert "nama_sungai" not in prompt


def test_build_user_prompt_tanpa_fakta_spasial_tidak_error():
    indikator = _indikator(fakta_spasial=None)
    prompt = build_user_prompt(indikator, [], None)
    assert "Fakta Spasial" not in prompt


def test_build_user_prompt_daftar_chunk():
    chunk = _chunk()
    prompt = build_user_prompt(_indikator(), [chunk], None)

    assert "citation_id=uu41-2009-p44" in prompt
    assert "UU No. 41 Tahun 2009" in prompt
    assert "Pasal 44" in prompt
    assert chunk.teks in prompt


def test_build_user_prompt_chunk_kosong_ada_peringatan():
    prompt = build_user_prompt(_indikator(), [], None)
    assert "jangan mengarang sitasi" in prompt


def test_build_user_prompt_target_none():
    prompt = build_user_prompt(_indikator(), [], None)
    assert "Tidak ada target numerik" in prompt


def test_build_user_prompt_target_terisi():
    prompt = build_user_prompt(_indikator(), [], {"target_maks": 600.0, "selisih": 50.0})
    assert "target_maks: 600.0" in prompt
    assert "selisih: 50.0" in prompt


def test_build_user_prompt_fakta_verdict_terisi():
    prompt = build_user_prompt(
        _indikator(), [], None, fakta_verdict="STATUS: MELEBIHI. Nilai aktual (90.0) > batas (80.0)."
    )
    assert "STATUS Perbandingan" in prompt
    assert "STATUS: MELEBIHI. Nilai aktual (90.0) > batas (80.0)." in prompt


def test_build_user_prompt_fakta_verdict_none_tidak_muncul():
    prompt = build_user_prompt(_indikator(), [], None, fakta_verdict=None)
    assert "STATUS Perbandingan" not in prompt


def test_build_user_prompt_chunk_dengan_istilah_kode():
    chunk = _chunk(
        id="rdtr-lampiran-vi-c1",
        pasal=None,
        istilah_kode="Lampiran VI",
        dokumen="Peraturan Daerah Kabupaten Sleman tentang RDTR",
        teks="KDB maksimum 80%.",
    )
    prompt = build_user_prompt(_indikator(), [chunk], None)
    assert "(Lampiran VI)" in prompt


def test_build_user_prompt_kegiatan_diizinkan_terisi():
    prompt = build_user_prompt(
        _indikator(), [], None, kegiatan_diizinkan=["Rumah toko (ruko) skala kecil", "Perdagangan eceran"]
    )
    assert "Kegiatan Diizinkan di Zona Ini" in prompt
    assert "Rumah toko (ruko) skala kecil" in prompt
    assert "Perdagangan eceran" in prompt


def test_build_user_prompt_kegiatan_diizinkan_kosong_tidak_muncul():
    prompt = build_user_prompt(_indikator(), [], None, kegiatan_diizinkan=None)
    assert "Kegiatan Diizinkan di Zona Ini" not in prompt

    prompt_kosong = build_user_prompt(_indikator(), [], None, kegiatan_diizinkan=[])
    assert "Kegiatan Diizinkan di Zona Ini" not in prompt_kosong
