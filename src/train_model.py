import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLS = [
    "age",
    "city_tier",
    "income_band",
    "last_app_open_days_ago",
    "onboarding_step_completed",
    "linked_bank_account",
    "sms_parsed_transactions_30d",
    "past_nudges_unique_count",
]
TARGET_COL = "nudge_clicked"


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["past_nudges_unique_count"] = (
        df["past_nudges_shown"].fillna("[]").str.count("nudge_").astype(float)
    )
    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "age",
        "last_app_open_days_ago",
        "onboarding_step_completed",
        "linked_bank_account",
        "sms_parsed_transactions_30d",
        "past_nudges_unique_count",
    ]
    categorical_features = ["city_tier", "income_band"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1200, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.25, 0.75, 101):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def top_coefficients(pipeline: Pipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    coef_pairs = sorted(
        [{"feature": feature, "coefficient": float(coef)} for feature, coef in zip(feature_names, coefs)],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )
    return coef_pairs[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nudge click prediction model.")
    parser.add_argument("--input", type=str, default="data/users.csv")
    parser.add_argument("--model-output", type=str, default="artifacts/nudge_model.joblib")
    parser.add_argument("--metrics-output", type=str, default="artifacts/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_engineer(args.input)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_probs = pipeline.predict_proba(X_train)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    threshold, train_f1 = select_threshold(y_train.to_numpy(), train_probs)
    y_pred = (test_probs >= threshold).astype(int)

    metrics = {
        "threshold_selected_for_f1": threshold,
        "train_f1_at_threshold": float(train_f1),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "top_predictive_features": top_coefficients(pipeline),
    }

    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_output}")


if __name__ == "__main__":
    main()
