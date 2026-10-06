from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from gensim.models import Word2Vec

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "pada", "itu", "ini", "juga", "untuk",
    "dengan", "akan", "adalah", "ialah", "oleh", "karena", "agar", "sebagai",
    "para", "dalam", "serta", "atau", "masih", "sudah", "belum", "tidak", "bukan",
    "bisa", "dapat", "lebih", "saat", "ketika", "setelah", "sebelum", "selama",
    "melalui", "antara", "hingga", "kemudian", "lantas", "merupakan", "namun",
    "tetapi", "tapi", "terdapat", "ada", "adanya", "yaitu", "yakni", "terkait",
    "misalnya", "maupun", "seperti", "jika", "kalau", "maka", "harus", "hendak",
    "sangat", "paling", "semua", "setiap", "beberapa", "saya", "anda", "kami",
    "kita", "mereka", "dia", "ia", "kepada", "bagi", "pun", "saja", "mungkin",
    "lagi", "tersebut",
}


def preprocess_text(text: str) -> list[str]:
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return [word for word in text.split() if word not in STOPWORDS and len(word) > 2]


def document_vector(model: Word2Vec, tokens: Iterable[str]) -> np.ndarray:
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def model_paths(root: Path) -> dict[str, Path]:
    models = root / "models"
    return {"word2vec": models / "word2vec_skipgram.model", "classifier": models / "naive_bayes.pkl", "encoder": models / "label_encoder.pkl", "metadata": models / "model_metadata.pkl"}


def save_metadata(root: Path, metadata: dict) -> None:
    path = model_paths(root)["metadata"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata, path)
