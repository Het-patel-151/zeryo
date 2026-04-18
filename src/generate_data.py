import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUDGE_IDS = [f"nudge_{i}" for i in range(1, 11)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_users: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(loc=33, scale=10, size=n_users).round().astype(int), 18, 65)
    city_tier = rng.choice(["tier_1", "tier_2", "tier_3"], size=n_users, p=[0.35, 0.4, 0.25])
    income_band = rng.choice(["low", "mid", "high"], size=n_users, p=[0.4, 0.42, 0.18])
    last_app_open_days_ago = np.clip(rng.gamma(shape=2.4, scale=5.0, size=n_users), 0, 90).round(1)
    onboarding_step_completed = rng.choice([1, 2, 3, 4, 5], size=n_users, p=[0.08, 0.15, 0.24, 0.28, 0.25])
    linked_bank_account = rng.random(n_users) < (0.2 + 0.12 * onboarding_step_completed)
    linked_bank_account = linked_bank_account.astype(int)

    # Higher activity if onboarding is deeper and bank is linked.
    lam = 2.0 + 0.9 * onboarding_step_completed + 2.4 * linked_bank_account
    sms_parsed_transactions_30d = rng.poisson(lam=lam, size=n_users)

    # Inject missingness to make preprocessing realistic.
    income_missing = rng.random(n_users) < 0.08
    tx_missing = rng.random(n_users) < 0.06
    last_open_missing = rng.random(n_users) < 0.04
    income_band = pd.Series(income_band).mask(income_missing).values
    sms_parsed_transactions_30d = pd.Series(sms_parsed_transactions_30d).mask(tx_missing).values
    last_app_open_days_ago = pd.Series(last_app_open_days_ago).mask(last_open_missing).values

    # Personalization context columns.
    candidate_nudges = [rng.choice(NUDGE_IDS, size=5, replace=False).tolist() for _ in range(n_users)]
    shown_counts = rng.integers(0, 6, size=n_users)
    past_nudges_shown = []
    for count in shown_counts:
        if count == 0:
            past_nudges_shown.append([])
        else:
            past_nudges_shown.append(rng.choice(NUDGE_IDS, size=count, replace=True).tolist())

    # Label generation.
    income_effect = pd.Series(income_band).map({"low": -0.2, "mid": 0.15, "high": 0.3}).fillna(0.0).to_numpy()
    city_effect = pd.Series(city_tier).map({"tier_1": 0.15, "tier_2": 0.05, "tier_3": -0.12}).to_numpy()
    tx_filled = pd.Series(sms_parsed_transactions_30d).fillna(0).to_numpy()
    last_open_filled = pd.Series(last_app_open_days_ago).fillna(30).to_numpy()
    seen_before_penalty = np.array([len(set(x)) for x in past_nudges_shown]) * -0.08

    logit = (
        -0.8
        + 0.34 * onboarding_step_completed
        + 0.42 * linked_bank_account
        + 0.06 * np.minimum(tx_filled, 25)
        - 0.05 * np.minimum(last_open_filled, 40)
        + income_effect
        + city_effect
        + seen_before_penalty
        + rng.normal(0, 0.55, size=n_users)
    )
    click_probability = sigmoid(logit)
    nudge_clicked = (rng.random(n_users) < click_probability).astype(int)

    return pd.DataFrame(
        {
            "user_id": [f"user_{i:04d}" for i in range(1, n_users + 1)],
            "age": age,
            "city_tier": city_tier,
            "income_band": income_band,
            "last_app_open_days_ago": last_app_open_days_ago,
            "onboarding_step_completed": onboarding_step_completed,
            "linked_bank_account": linked_bank_account,
            "sms_parsed_transactions_30d": sms_parsed_transactions_30d,
            "past_nudges_shown": past_nudges_shown,
            "candidate_nudges": candidate_nudges,
            "nudge_clicked": nudge_clicked,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Zeyro users dataset.")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/users.csv")
    args = parser.parse_args()

    df = generate_dataset(n_users=args.n_users, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote dataset to {output_path} with shape={df.shape}")


if __name__ == "__main__":
    main()
