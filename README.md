# Airbnb Customer Reviews – Sentiment Analysis (API vs ML)

Sentiment Analysis project using **Airbnb customer reviews**. Two approaches are compared:

1) **Text2Data Sentiment API** (3-class output: positive/neutral/negative)
2) **Machine Learning model**: TF‑IDF (1–3 grams) + **Random Forest** classifier

The original dataset consisted of **834** English reviews from **20** listings; **820** reviews were used in the API evaluation, and **817** reviews were used in the ML pipeline after cleaning.

## Results (from the report)

| Approach | Accuracy | Precision (Pos / Neg) | Recall (Pos / Neg) | F1 (Pos / Neg) |
|---|---:|---:|---:|---:|
| Text2Data API | **77.43%** | 81.26% / 71.73% | 81.09% / 71.95% | 81.17% / 71.83% |
| Random Forest (TF‑IDF) | **92.68%** | 93.87% / 90.90% | 93.87% / 90.90% | 93.87% / 90.90% |

These are the metrics reported in the coursework report.

## Data included in this GitHub-ready version

This repo includes **an anonymized sample** (200 rows) for reproducibility:

- `data/raw/extracted_reviews_sample.csv`
- `data/processed/reviews_sentiments_sample.csv`

The public sample removes user-identifying fields (e.g., name, profile info). The full anonymized dataset can be kept privately.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.evaluation.text2data_performance_evaluation
python -m src.models.random_forest_sentiment
python -m src.models.compare_models
```

Outputs (confusion matrices + plots) are saved to `outputs/figures/`.

## Repository structure

- `src/evaluation/` — evaluation scripts (API confusion matrix + metrics)
- `src/models/` — ML training & model comparisons (RF vs LR vs SVM)
- `data/` — sample datasets (raw + processed)
- `docs/report.docx` — original coursework report

## Notes on ethics / data

The report discusses ethical considerations for data mining and personal data.
For a public repo, it’s best practice to **avoid publishing personal identifiers** and to share only the minimum data needed to reproduce results (or share synthetic / sample data).

