from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import project_root

def main() -> None:
    root = project_root()
    data_path = root / "data" / "processed" / "reviews_sentiments_sample.csv"
    df = pd.read_csv(data_path)

    # Map labels to readable names (keeping NEU if present)
    y_true = df["Actual_Sentiment"].astype(str)
    y_pred = df["Predicted_Sentiment"].astype(str)

    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, labels=labels))

    # Plot confusion matrix
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Text2Data (Predicted) vs Annotator (Actual)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix_text2data_sample.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved figure -> {out_path}")

if __name__ == "__main__":
    main()
