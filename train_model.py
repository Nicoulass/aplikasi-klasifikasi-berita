from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from pipeline import document_vector, model_paths, preprocess_text, save_metadata


def train(data_path: Path, project_root: Path) -> None:
    df = pd.read_csv(data_path)
    required = {"Isi Berita", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom dataset kurang: {sorted(missing)}")
    df = df[["Isi Berita", "Label"]].dropna()
    df["Label"] = df["Label"].astype(str).str.lower().str.strip()
    if set(df["Label"].unique()) != {"sport", "finance"}:
        raise ValueError(f"Label harus sport dan finance, ditemukan: {sorted(df['Label'].unique())}")
    tokens = df["Isi Berita"].map(preprocess_text).tolist()
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(tokens, df["Label"].tolist(), test_size=0.2, random_state=42, stratify=df["Label"].tolist())
    word2vec = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4, sg=1, seed=42, epochs=100)
    encoder = LabelEncoder()
    y_train, y_test = encoder.fit_transform(train_labels), encoder.transform(test_labels)
    X_train = np.array([document_vector(word2vec, row) for row in train_tokens])
    X_test = np.array([document_vector(word2vec, row) for row in test_tokens])
    classifier = GaussianNB().fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    paths = model_paths(project_root)
    paths["word2vec"].parent.mkdir(parents=True, exist_ok=True)
    word2vec.save(str(paths["word2vec"]))
    joblib.dump(classifier, paths["classifier"])
    joblib.dump(encoder, paths["encoder"])
    save_metadata(project_root, {"classes": list(encoder.classes_), "vector_size": 100, "window": 5, "sg": 1, "train_rows": len(train_tokens), "test_rows": len(test_tokens), "dataset": str(data_path)})
    print(f"Data: {len(df)} baris ({len(train_tokens)} train, {len(test_tokens)} test)")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Model disimpan di: {paths['word2vec'].parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih Skip-gram + Gaussian Naive Bayes")
    parser.add_argument("--data", type=Path, default=Path("../output/berita.csv"))
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    train(args.data.resolve(), args.root.resolve())
