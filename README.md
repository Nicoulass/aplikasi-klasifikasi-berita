# Aplikasi Klasifikasi Berita

Aplikasi Web Mining untuk mengklasifikasikan berita `sport` atau `finance` dari URL.

Pipeline: URL -> Trafilatura -> preprocessing + stopword -> Word2Vec Skip-gram (`sg=1`) -> document vector -> Gaussian Naive Bayes.

## Menjalankan

Gunakan Python 3.12 environment `enwebmining` yang sama untuk training dan aplikasi. Jangan memakai Python Laragon 3.10 untuk membuka model, karena file model dibuat dengan `scikit-learn==1.9.1`.

Dari folder aplikasi di PowerShell:

```powershell
& "..\enwebmining\Scripts\python.exe" -m pip install -r requirements.txt
& "..\enwebmining\Scripts\python.exe" train_model.py --data "..\output\berita.csv"
& "..\enwebmining\Scripts\python.exe" -m streamlit run app\streamlit_app.py
```

Aplikasi mencoba Trafilatura terlebih dahulu, kemudian fallback ke BeautifulSoup jika struktur halaman berita tidak dikenali. URL yang gagal dicatat bersama penyebabnya di `scraping_errors.log`.

Model disimpan di `models/`. Dataset lama tidak dihapus. URL yang gagal diambil dicatat di `scraping_errors.log`.

## Deployment

Unggah folder ini ke GitHub setelah training, pastikan `models/` ikut diunggah, lalu pilih `app/streamlit_app.py` sebagai entrypoint di Streamlit Community Cloud.
