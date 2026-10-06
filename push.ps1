param (
    [string]$Message = "Deploy aplikasi klasifikasi berita",
    [string]$RepoUrl = "https://github.com/Nicoulass/aplikasi-klasifikasi-berita.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "    Memulai Proses Deploy Otomatis       " -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""

    # 1. Init repository bila belum ada
    Write-Host "1. Menyiapkan Git repository..." -ForegroundColor Yellow
    if (-not (Test-Path (Join-Path $PSScriptRoot ".git"))) {
        git init
        if ($LASTEXITCODE -ne 0) { throw "git init gagal" }
        git checkout -b $Branch
        if ($LASTEXITCODE -ne 0) { throw "git checkout gagal" }
    }
    Write-Host ""

    # 2. Set / perbaiki remote origin
    Write-Host "2. Memastikan remote origin menunjuk ke $RepoUrl..." -ForegroundColor Yellow
    $hasOrigin = (git remote) -contains "origin"
    if ($hasOrigin) { git remote set-url origin $RepoUrl } else { git remote add origin $RepoUrl }
    if ($LASTEXITCODE -ne 0) { throw "set remote gagal" }
    Write-Host ""

    # 3. Tambahkan semua perubahan
    Write-Host "3. Menambahkan file yang berubah ke Git..." -ForegroundColor Yellow
    git add .
    if ($LASTEXITCODE -ne 0) { throw "git add gagal" }
    Write-Host ""

    # 4. Buat commit hanya bila ada perubahan
    $staged = git diff --cached --name-only
    if ($staged) {
        Write-Host "4. Menyimpan perubahan (Commit) dengan pesan: '$Message'..." -ForegroundColor Yellow
        git commit -m $Message
        if ($LASTEXITCODE -ne 0) { throw "git commit gagal" }
    } else {
        Write-Host "4. Tidak ada perubahan baru, commit dilewati." -ForegroundColor Yellow
    }
    Write-Host ""

    # 5. Push ke GitHub
    Write-Host "5. Mengirim (Push) ke GitHub..." -ForegroundColor Yellow
    git push -u origin $Branch
    if ($LASTEXITCODE -ne 0) { throw "git push gagal. Cek kredensial GitHub (Git Credential Manager)." }
    Write-Host ""

    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "SELESAI! Perubahan berhasil dikirim." -ForegroundColor Green
    Write-Host "Repo: https://github.com/Nicoulass/aplikasi-klasifikasi-berita" -ForegroundColor Green
    Write-Host "Langkah berikutnya: deploy di https://share.streamlit.io" -ForegroundColor Green
    Write-Host "- Repository : Nicoulass/aplikasi-klasifikasi-berita" -ForegroundColor Green
    Write-Host "- Branch     : $Branch" -ForegroundColor Green
    Write-Host "- Main file  : app/streamlit_app.py" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
