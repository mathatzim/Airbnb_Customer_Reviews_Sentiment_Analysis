from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.utils import project_root

def main() -> None:
    root = project_root()
    data_path = root / "data" / "processed" / "reviews_sentiments_sample.csv"
    df = pd.read_csv(data_path)

    # Binary labels for comparison
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

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "SVM (RBF)": SVC(kernel="rbf"),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))

    results.sort(key=lambda x: x[1], reverse=True)
    print("Accuracy on sample split (higher is better):\n")
    for name, acc in results:
        print(f"- {name}: {acc:.4f}")

if __name__ == "__main__":
    main()
