from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import document_vector, model_paths, preprocess_text

st.set_page_config(page_title="Klasifikasi Berita", page_icon="News", layout="centered")
st.title("Klasifikasi Berita")
st.caption("Word2Vec Skip-gram + Gaussian Naive Bayes")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def extract_article(url: str) -> tuple[str | None, str]:
    """Ambil artikel dengan beberapa parser dan kembalikan (teks, sumber)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, "URL tidak valid; gunakan URL yang diawali http:// atau https://"

    html = None
    fetch_error = None

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "id-ID,id;q=0.9,en;q=0.8"},
            timeout=20,
            allow_redirects=True,
        )
        response.raise_for_status()
        html = response.text
    except requests.RequestException as error:
        fetch_error = f"request gagal: {error}"

    # Trafilatura biasanya paling baik untuk artikel yang HTML-nya standar.
    if html:
        article = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
        )
        if article and len(article.strip()) >= 120:
            return article.strip(), "trafilatura (requests)"

    # Coba downloader bawaan Trafilatura bila requests ditolak/redirect bermasalah.
    if not html:
        try:
            html = trafilatura.fetch_url(url)
            article = trafilatura.extract(html) if html else None
            if article and len(article.strip()) >= 120:
                return article.strip(), "trafilatura (fetch_url)"
        except Exception as error:
            fetch_error = f"trafilatura gagal: {error}"

    if not html:
        return None, fetch_error or "server tidak mengembalikan HTML"

    # Fallback untuk situs seperti Lambeturah yang tidak dikenali Trafilatura.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    selectors = [
        "article", "main", "[role='main']", ".article-content", ".post-content",
        ".entry-content", ".detail-content", ".post-detail", ".content-detail",
    ]
    candidates = soup.select(", ".join(selectors))
    if candidates:
        text = max((node.get_text(" ", strip=True) for node in candidates), key=len)
        if len(text) >= 120:
            return text, "BeautifulSoup (article container)"

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) >= 30)
    if len(text) >= 120:
        return text, "BeautifulSoup (paragraph fallback)"

    return None, f"HTML berhasil diambil tetapi isi artikel tidak ditemukan; {fetch_error or 'struktur halaman tidak dikenali'}"


@st.cache_resource
def load_models():
    paths = model_paths(PROJECT_ROOT)
    required = [paths["word2vec"], paths["classifier"], paths["encoder"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Model belum tersedia. Jalankan train_model.py terlebih dahulu.")
    return Word2Vec.load(str(paths["word2vec"])), joblib.load(paths["classifier"]), joblib.load(paths["encoder"])


try:
    word2vec, classifier, encoder = load_models()
except Exception as error:
    st.error(str(error))
    st.stop()

url = st.text_input("URL berita", placeholder="https://contoh.com/berita")
if st.button("Klasifikasikan", type="primary", use_container_width=True):
    if not url.strip():
        st.warning("Masukkan URL berita terlebih dahulu.")
        st.stop()

    with st.spinner("Mengambil dan membaca artikel..."):
        try:
            article, extraction_method = extract_article(url.strip())
        except Exception as error:
            article = None
            extraction_method = f"exception: {error}"

    if not article:
        error_line = f"{url.strip()} | {extraction_method}\n"
        with (PROJECT_ROOT / "scraping_errors.log").open("a", encoding="utf-8") as log:
            log.write(error_line)
        st.error(f"Artikel gagal diekstrak: {extraction_method}")
        st.info("URL dapat dibuka di browser, tetapi server atau struktur HTML-nya tidak cocok dengan parser.")
        st.stop()

    tokens = preprocess_text(article)
    if not tokens:
        st.error("Artikel berhasil diambil, tetapi tidak memiliki kata yang dapat diproses.")
        st.stop()

    vector = document_vector(word2vec, tokens).reshape(1, -1)
    prediction = classifier.predict(vector)
    probabilities = classifier.predict_proba(vector)[0]
    label = encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities) * 100)

    st.success(f"Kategori: {label.upper()}")
    st.metric("Keyakinan model", f"{confidence:.2f}%")
    st.caption(f"Ekstraksi berhasil menggunakan: {extraction_method}")
    st.json({name: f"{probability * 100:.2f}%" for name, probability in zip(encoder.classes_, probabilities)})
    with st.expander("Lihat isi artikel"):
        st.write(article[:5000])
