from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import project_root

def main() -> None:
    root = project_root()
    data_path = root / "data" / "processed" / "reviews_sentiments_sample.csv"
    df = pd.read_csv(data_path)

    # Keep only binary labels for the ML part (P/N), matching the report’s setup.
    df = df[df["Actual_Sentiment"].isin(["P", "N"])].copy()
    df = df.dropna(subset=["Review_Text"])
    df = df.drop_duplicates(subset="Review_Text", keep="first")
    df["Review_Text"] = df["Review_Text"].astype(str).str.lower()

    X = df["Review_Text"]
    y = df["Actual_Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=200)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_vec, y_train)

    y_pred = rf.predict(X_test_vec)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    labels = ["P", "N"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Random Forest (TF-IDF) — Confusion Matrix (sample)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix_rf_sample.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved figure -> {out_path}")

if __name__ == "__main__":
    main()
