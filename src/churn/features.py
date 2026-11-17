"""Turn raw account + monthly usage data into one row of features per account."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FEATURE_COLUMNS = [
    "seat_adoption",
    "adoption_trend",
    "logins_per_user",
    "avg_features_used",
    "tickets_last_3m",
    "escalations_last_3m",
    "nps_score",
    "days_since_csm_touch",
    "onboarding_days",
    "tenure_months",
    "days_to_renewal",
]


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_csv(data_dir / "accounts.csv", parse_dates=["signup_date", "renewal_date"])
    usage = pd.read_csv(data_dir / "monthly_usage.csv")
    return accounts, usage


def build_features(accounts: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly usage into account-level features.

    Uses only usage data plus account attributes, so the same function works for
    scoring live accounts (no churn label needed).
    """
    usage = usage.sort_values(["account_id", "month"])
    months = sorted(usage["month"].unique())
    first3, last3 = months[:3], months[-3:]
    latest = months[-1]

    seats = accounts.set_index("account_id")["seats"]

    latest_rows = usage[usage.month == latest].set_index("account_id")
    early = usage[usage.month.isin(first3)].groupby("account_id")["active_users"].mean()
    recent = usage[usage.month.isin(last3)].groupby("account_id")

    feats = pd.DataFrame(index=accounts["account_id"])
    feats["seat_adoption"] = (latest_rows["active_users"] / seats).clip(upper=1.0)
    feats["adoption_trend"] = (recent["active_users"].mean() - early) / seats
    feats["logins_per_user"] = latest_rows["logins"] / latest_rows["active_users"].clip(lower=1)
    feats["avg_features_used"] = recent["features_used"].mean()
    feats["tickets_last_3m"] = recent["support_tickets"].sum()
    feats["escalations_last_3m"] = recent["escalations"].sum()

    acc = accounts.set_index("account_id")
    snapshot = pd.Timestamp(f"{latest}-01") + pd.offsets.MonthEnd(0)
    feats["nps_score"] = acc["nps_score"]
    feats["days_since_csm_touch"] = acc["days_since_csm_touch"]
    feats["onboarding_days"] = acc["onboarding_days"]
    feats["tenure_months"] = ((snapshot - acc["signup_date"]).dt.days / 30.44).round(1)
    feats["days_to_renewal"] = (acc["renewal_date"] - snapshot).dt.days

    return feats.fillna(0).reset_index()
