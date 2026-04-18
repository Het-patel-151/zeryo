import argparse
import json
from pathlib import Path

import pandas as pd


def parse_list_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or cell.strip() in {"", "[]"}:
        return []
    clean = cell.strip()[1:-1].strip()
    if not clean:
        return []
    return [item.strip().strip("'").strip('"') for item in clean.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick EDA and feature insights.")
    parser.add_argument("--input", type=str, default="data/users.csv")
    parser.add_argument("--output", type=str, default="artifacts/eda_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["past_nudges_unique_count"] = df["past_nudges_shown"].fillna("[]").apply(
        lambda x: len(set(parse_list_cell(x)))
    )

    summary = {
        "row_count": int(df.shape[0]),
        "click_rate": float(df["nudge_clicked"].mean()),
        "missingness": df.isna().mean().round(4).to_dict(),
        "click_rate_by_onboarding_step": df.groupby("onboarding_step_completed")["nudge_clicked"]
        .mean()
        .round(4)
        .to_dict(),
        "click_rate_linked_bank": df.groupby("linked_bank_account")["nudge_clicked"].mean().round(4).to_dict(),
        "click_rate_last_open_bucket": (
            df.assign(
                last_open_bucket=pd.cut(
                    df["last_app_open_days_ago"],
                    bins=[-1, 3, 7, 14, 30, 90],
                    labels=["0-3d", "4-7d", "8-14d", "15-30d", "31-90d"],
                )
            )
            .groupby("last_open_bucket", observed=False)["nudge_clicked"]
            .mean()
            .round(4)
            .to_dict()
        ),
        "click_rate_by_tx_bucket": (
            df.assign(
                tx_bucket=pd.cut(
                    df["sms_parsed_transactions_30d"],
                    bins=[-1, 2, 5, 9, 20, 100],
                    labels=["0-2", "3-5", "6-9", "10-20", "21+"],
                )
            )
            .groupby("tx_bucket", observed=False)["nudge_clicked"]
            .mean()
            .round(4)
            .to_dict()
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
