# Zeyro Assignment

Solution for the Zeyro ML Engineer take-home:
- synthetic dataset generation,
- feature engineering + EDA,
- click prediction model with evaluation,
- personalized nudge ranking with cold-start logic,
- drift monitoring design.

## Project Structure

- `src/generate_data.py` - creates synthetic user dataset (500 rows by default)
- `src/analyze_data.py` - EDA summary and feature insights
- `src/train_model.py` - preprocessing + model training + metrics
- `src/rank_nudges.py` - ranks 5 candidate nudges/user
- `data/users.csv` - generated dataset
- `artifacts/metrics.json` - model metrics and top feature coefficients
- `artifacts/eda_summary.json` - EDA output
- `artifacts/ranked_nudges.json` - sample ranked results
- `docs/walkthrough.md` - written assignment response

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run End-to-End

```bash
python src/generate_data.py --n-users 500 --seed 42 --output data/users.csv
python src/analyze_data.py --input data/users.csv --output artifacts/eda_summary.json
python src/train_model.py --input data/users.csv --model-output artifacts/nudge_model.joblib --metrics-output artifacts/metrics.json
python src/rank_nudges.py --input data/users.csv --model artifacts/nudge_model.joblib --output artifacts/ranked_nudges.json --sample-size 10
```

## Current Results (seed=42)

- Precision: `0.6500`
- Recall: `0.9420`
- F1: `0.7692`
- ROC-AUC: `0.7117`
- Threshold selected for best F1: `0.34`

Top learned drivers include:
- higher `onboarding_step_completed` (+),
- lower `last_app_open_days_ago` (-),
- linked bank account (+),
- income/city tier effects.

## Notes on Modeling Decisions

- Used logistic regression as a practical, interpretable baseline.
- Avoided data leakage by only using user-side signals available before prediction.
- Optimized F1 with recall emphasis to reduce false negatives for engagement nudges.

For the detailed write-up aligned with all PDF questions, see `docs/walkthrough.md`.
