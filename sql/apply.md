# Menerapkan / memperbarui skema DB

`schema.sql` otomatis dijalankan **sekali** oleh Postgres saat volume data masih kosong
(mekanisme `docker-entrypoint-initdb.d`). Untuk kerja sehari-hari:

## Pertama kali
```bash
docker compose up -d
docker compose ps          # tunggu STATUS = healthy
```
Skema langsung terpasang. Cek:
```bash
docker compose exec db psql -U rdtr -d rdtr -c "\dt"
docker compose exec db psql -U rdtr -d rdtr -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

## Setelah mengubah `schema.sql`
initdb TIDAK mengulang kalau volume sudah berisi. Dua pilihan:

Reset total (hapus semua data — aman saat dev):
```bash
docker compose down -v && docker compose up -d
```

Terapkan manual tanpa hapus data:
```bash
docker compose exec -T db psql -U rdtr -d rdtr < sql/schema.sql
```
(Semua statement `CREATE ... IF NOT EXISTS`, jadi idempoten untuk objek baru.)

## Catatan
- pgvector di image `pgvector/pgvector:pg16` ≥ 0.7 (mendukung HNSW). Divalidasi juga di pgvector 0.6.
- DB ini dipakai bersama reasoning-side (tabel `logs` milik reasoning). Tabel RAG di sini
  (`dokumen`, `chunks`, `matriks_kegiatan`, `intensitas_zona`) tidak bentrok nama.
- Pindah ke Supabase nanti: jalankan `schema.sql` yang sama di SQL editor Supabase, lalu ganti
  `DATABASE_URL` di `.env`. Tidak ada perubahan kode.
