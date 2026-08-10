# Setup Deploy Key + GitHub Secrets untuk DRISleman-ai

CI auto-deploy butuh SSH keypair (deploy key), bukan PAT. Alasan: scope repo-spesifik, least-privilege, rotate-friendly.

## Langkah 1 — Generate key di VPS (user `ml`)

Login ke VPS dulu:
```bash
ssh urbansolv.ml
```

Jalankan ini (paste satu-satu):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "ci-drisman-ai"
```

Verifikasi (private key ada):
```bash
ls -la ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
cat ~/.ssh/id_ed25519.pub   # COPY OUTPUT INI — langkah 2
```

## Langkah 2 — Tambah pubkey ke GitHub Deploy Keys

1. Buka https://github.com/MspUrbansolv/project-2026-07-DRISleman-ai/settings/keys/new
2. Title: `CI DRISleman-ai (VPS1 ml)`
3. Key: paste pubkey dari langkah 1
4. ✅ **Allow write access** (kalau mau auto-deploy dari CI; kalau read-only tidak perlu ini)
5. Klik **Add key**

## Langkah 3 — Tambah 3 Secrets di repo

https://github.com/MspUrbansolv/project-2026-07-DRISleman-ai/settings/secrets/actions → **New repository secret**

| Secret | Value |
|---|---|
| `ML_HOST` | `206.237.97.19` |
| `ML_USER` | `ml` |
| `ML_SSH_KEY` | output `cat ~/.ssh/id_ed25519` (private key, seluruh isi termasuk `-----BEGIN/END-----`) |

⚠️ Paste private key **utuh** (jangan terpotong). Cara aman pakai `gh CLI`:
```bash
gh secret set -R MspUrbansolv/project-2026-07-DRISleman-ai ML_SSH_KEY < ~/.ssh/id_ed25519
```

## Langkah 4 — Test workflow

Setelah secrets + deploy key masuk:
1. Commit kecil di repo (mis. tambah baris di README)
2. push ke `main`
3. cek tab **Actions** di GitHub → run pertama hijau = jalan ✅

Kalau merah, klik run gagal → lihat log → biasanya:
- `Permission denied (publickey)` → pubkey belum masuk atau `Allow write access` belum dicentang
- `Host key verification failed` → tambah `known_hosts`:
  ```yaml
  - name: Setup SSH known_hosts
    run: |
      mkdir -p ~/.ssh
      ssh-keyscan -H 206.237.97.19 >> ~/.ssh/known_hosts
  ```
  sebelum rsync step di workflow (kalau perlu — tapi kebanyakan VPS sudah punya `known_hosts` di runner pool)

## Rollback

Hapus deploy key: GitHub → Settings → Deploy keys → Remove  
Hapus secrets: Settings → Secrets and variables → Actions → Delete  
Hapus keypair di VPS: `ssh ml@VPS "rm ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub"`

## Kenapa bukan PAT?

| Aspek | Deploy Key (SSH) | PAT (Personal Access Token) |
|---|---|---|
| Scope | Repo ini saja | User/org-wide |
| Rotate | 1 repo | Semua repo user |
| Compromised blast | Kecil | Besar |
| Permission | Linux SSH key auth | HTTP Basic Auth |
| Cocok untuk | CI/CD spesifik repo | Scripts multi-repo |

→ Deploy key lebih aman. PAT jadi cadangan kalau ada halangan teknis.
