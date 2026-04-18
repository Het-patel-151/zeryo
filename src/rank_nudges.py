import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


NUDGE_PRIOR = {
    "nudge_1": 0.52,
    "nudge_2": 0.49,
    "nudge_3": 0.58,
    "nudge_4": 0.56,
    "nudge_5": 0.44,
    "nudge_6": 0.60,
    "nudge_7": 0.47,
    "nudge_8": 0.54,
    "nudge_9": 0.51,
    "nudge_10": 0.46,
}

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


def parse_list_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or cell.strip() in {"", "[]"}:
        return []
    clean = cell.strip()[1:-1].strip()
    if not clean:
        return []
    return [item.strip().strip("'").strip('"') for item in clean.split(",")]


def rank_candidates(user_row: pd.Series, model) -> list[dict]:
    candidates = parse_list_cell(user_row["candidate_nudges"])
    past = set(parse_list_cell(user_row["past_nudges_shown"]))
    is_cold_start = len(past) == 0

    user_features = user_row.to_frame().T.copy()
    user_features["past_nudges_unique_count"] = float(len(past))
    base_prob = float(model.predict_proba(user_features[FEATURE_COLS])[:, 1][0])

    ranked = []
    for nudge in candidates:
        prior = NUDGE_PRIOR.get(nudge, 0.5)
        seen_penalty = -0.15 if nudge in past else 0.0
        hybrid_score = (0.35 * prior + 0.65 * base_prob) + seen_penalty
        if is_cold_start:
            # In cold start, rely more on priors to avoid overconfidence.
            hybrid_score = 0.7 * prior + 0.3 * base_prob
        ranked.append(
            {
                "nudge_id": nudge,
                "hybrid_score": round(float(hybrid_score), 4),
                "model_score": round(base_prob, 4),
                "prior": round(float(prior), 4),
            }
        )

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate nudges by relevance.")
    parser.add_argument("--input", type=str, default="data/users.csv")
    parser.add_argument("--model", type=str, default="artifacts/nudge_model.joblib")
    parser.add_argument("--output", type=str, default="artifacts/ranked_nudges.json")
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model = joblib.load(args.model)

    results = []
    for _, row in df.head(args.sample_size).iterrows():
        ranked = rank_candidates(row, model)
        results.append(
            {
                "user_id": row["user_id"],
                "cold_start": len(parse_list_cell(row["past_nudges_shown"])) == 0,
                "ranked_nudges": ranked,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote ranked nudges to {output_path}")
    print(json.dumps(results[:2], indent=2))


if __name__ == "__main__":
    main()
