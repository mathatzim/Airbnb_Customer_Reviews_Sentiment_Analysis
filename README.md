# Airbnb Customer Reviews – Sentiment Analysis (API vs ML)

Sentiment analysis project using **Airbnb customer reviews**. Two approaches are compared:

1. **Text2Data Sentiment API** (3-class output: positive / neutral / negative)
2. **Machine Learning**: TF-IDF (1–3 grams) + **Random Forest** classifier (plus model comparisons)

## Dataset
- **834** English reviews from **20** listings (original extraction)
- **820** reviews used in the API evaluation
- **817** reviews used in the ML pipeline after cleaning

This public repo includes **an anonymized sample (200 rows)** for reproducibility:
- `data/raw/extracted_reviews_sample.csv`
- `data/processed/reviews_sentiments_sample.csv`

## Results (from the coursework report)

| Approach | Accuracy | Precision (Pos / Neg) | Recall (Pos / Neg) | F1 (Pos / Neg) |
|---|---:|---:|---:|---:|
| Text2Data API | **77.43%** | 81.26% / 71.73% | 81.09% / 71.95% | 81.17% / 71.83% |
| Random Forest (TF-IDF) | **92.68%** | 93.87% / 90.90% | 93.87% / 90.90% | 93.87% / 90.90% |

## Quickstart

```bash
pip install -r requirements.txt
python -m src.evaluation.text2data_performance_evaluation
python -m src.models.random_forest_sentiment
python -m src.models.compare_models
