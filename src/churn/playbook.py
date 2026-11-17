"""Next-best-action recommendations a CSM can act on this week."""

from __future__ import annotations

import pandas as pd

ACTIONS = {
    "adoption": "Run a seat-activation campaign: share usage report, book a training for inactive users.",
    "engagement": "Usage is declining. Schedule a value review call and re-confirm the customer's goals.",
    "support": "High ticket/escalation load. Loop in support lead, do a root-cause review with the customer.",
    "sentiment": "Low NPS. Call the detractor within 48h, log feedback, agree on a fix plan.",
    "relationship": "No recent CSM contact. Book a check-in and update the success plan.",
}


def recommend(row: pd.Series) -> str:
    """One prioritised action per account, based on its weakest area and renewal timing."""
    action = ACTIONS[row["weakest_area"]]
    churn_prob = row.get("churn_probability", 0)
    if row["days_to_renewal"] <= 60 and (row["health_tier"] != "Healthy" or churn_prob >= 0.5):
        return f"URGENT (renews in {int(row['days_to_renewal'])}d): exec sponsor call + " + action
    return action


def priority_queue(scored: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Accounts a CSM team should work first: highest ARR at risk."""
    df = scored.copy()
    risk = df["churn_probability"] if "churn_probability" in df else (100 - df["health_score"]) / 100
    df["arr_at_risk_inr"] = (df["arr_inr"] * risk).round(-3).astype(int)
    df["next_best_action"] = df.apply(recommend, axis=1)
    cols = [
        "account_id", "company", "csm_owner", "plan", "arr_inr", "health_score", "health_tier",
        "days_to_renewal", "arr_at_risk_inr", "next_best_action",
    ]
    if "churn_probability" in df:
        cols.insert(6, "churn_probability")
    return df.sort_values("arr_at_risk_inr", ascending=False)[cols].head(top_n).reset_index(drop=True)
