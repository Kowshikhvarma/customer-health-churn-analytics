"""End-to-end run: data -> features -> health score -> churn model -> CSM priority list.

Usage:
    python -m churn.pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn import generate_data, model
from churn.features import DATA_DIR, build_features, load_raw
from churn.health_score import score_accounts, tier_summary
from churn.playbook import priority_queue


def score_all(data_dir: Path = DATA_DIR, models_dir: Path = model.MODELS_DIR) -> pd.DataFrame:
    accounts, usage = load_raw(data_dir)
    feats = build_features(accounts, usage)
    scored = score_accounts(feats)
    clf = model.load_model(models_dir)
    scored["churn_probability"] = model.predict_proba(clf, scored).round(3)
    return accounts.merge(scored, on=["account_id", "nps_score", "days_since_csm_touch", "onboarding_days"])


def main() -> None:
    if not (DATA_DIR / "accounts.csv").exists():
        accounts, usage = generate_data.generate()
        DATA_DIR.mkdir(exist_ok=True)
        accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
        usage.to_csv(DATA_DIR / "monthly_usage.csv", index=False)

    metrics = model.train()
    scored = score_all()
    scored.to_csv(DATA_DIR / "scored_accounts.csv", index=False)
    queue = priority_queue(scored)
    queue.to_csv(DATA_DIR / "csm_priority_queue.csv", index=False)

    print(f"Model: {metrics['model']}  |  hold-out ROC-AUC {metrics['test_roc_auc']}")
    print("\nHealth tier vs actual churn:")
    print(tier_summary(scored).to_string())
    print("\nTop 5 accounts to work this week:")
    print(queue.head(5)[["company", "health_tier", "arr_at_risk_inr", "next_best_action"]].to_string(index=False))


if __name__ == "__main__":
    main()
