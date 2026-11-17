"""Generate a realistic, fully synthetic B2B SaaS customer dataset.

Every account has a hidden "risk" level that drives its usage, support load,
sentiment and, finally, whether it churned. That gives the analytics and the
model a real signal to find, while keeping the data free of any client info.

Usage:
    python -m churn.generate_data --accounts 600 --seed 42
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

INDUSTRIES = ["Healthcare", "FinTech", "Retail", "EdTech", "Logistics", "Manufacturing", "IT Services"]
PLANS = {"Starter": (8_000, 25_000), "Growth": (25_000, 90_000), "Enterprise": (90_000, 350_000)}
CSMS = ["Asha", "Ravi", "Meera", "Arjun", "Divya"]
SNAPSHOT_DATE = date(2026, 10, 31)
MONTHS = 6

_PREFIX = [
    "Nova", "Blue", "Peak", "Bright", "Core", "Swift", "Prime", "Green", "Vertex", "Orbit",
    "Lumen", "Apex", "Indigo", "Zenith", "Cedar", "Harbor", "Quartz", "Summit", "Atlas", "Kite",
]
_CORE = ["Labs", "Systems", "Health", "Logistics", "Retail", "Learning", "Works", "Analytics", "Cloud", "Foods", "Mobility", "Finance"]
_TAIL = ["", " Technologies", " Solutions", " India", " Global"]


def _company_names(rng: np.random.Generator, n: int) -> list[str]:
    """Unique fictional company names (up to 1,200)."""
    pool = [f"{p} {c}{t}" for p in _PREFIX for c in _CORE for t in _TAIL]
    if n > len(pool):
        raise ValueError(f"Can generate at most {len(pool)} unique names")
    return list(rng.choice(pool, size=n, replace=False))


def generate(n_accounts: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accounts, monthly_usage) DataFrames."""
    rng = np.random.default_rng(seed)

    plans = rng.choice(list(PLANS), size=n_accounts, p=[0.5, 0.35, 0.15])
    risk = rng.beta(2, 5, size=n_accounts)  # hidden: most accounts are healthy

    accounts = pd.DataFrame(
        {
            "account_id": [f"ACC-{i:04d}" for i in range(1, n_accounts + 1)],
            "company": _company_names(rng, n_accounts),
            "industry": rng.choice(INDUSTRIES, size=n_accounts),
            "plan": plans,
            "csm_owner": rng.choice(CSMS, size=n_accounts),
        }
    )
    accounts["arr_inr"] = [
        int(round(rng.uniform(*PLANS[p]) * 12, -3)) for p in plans
    ]
    seat_base = {"Starter": 15, "Growth": 60, "Enterprise": 250}
    accounts["seats"] = [max(5, int(rng.normal(seat_base[p], seat_base[p] * 0.3))) for p in plans]

    tenure_days = rng.integers(60, 1100, size=n_accounts)
    accounts["signup_date"] = [SNAPSHOT_DATE - timedelta(days=int(d)) for d in tenure_days]
    accounts["renewal_date"] = [SNAPSHOT_DATE + timedelta(days=int(d)) for d in rng.integers(5, 365, n_accounts)]
    accounts["onboarding_days"] = np.clip(rng.normal(21 + 30 * risk, 7), 5, 120).round().astype(int)
    accounts["days_since_csm_touch"] = np.clip(rng.normal(10 + 50 * risk, 8), 0, 120).round().astype(int)

    # Monthly usage, trending down for risky accounts.
    rows = []
    for idx, acc in accounts.iterrows():
        r = risk[idx]
        start_adoption = np.clip(rng.normal(0.85 - 0.45 * r, 0.08), 0.05, 1.0)
        monthly_drift = -0.07 * r + rng.normal(0.005, 0.01)
        for m in range(MONTHS):
            month = (pd.Timestamp(SNAPSHOT_DATE) - pd.DateOffset(months=MONTHS - 1 - m)).to_period("M")
            adoption = float(np.clip(start_adoption + monthly_drift * m + rng.normal(0, 0.03), 0.02, 1.0))
            active = int(round(acc.seats * adoption))
            rows.append(
                {
                    "account_id": acc.account_id,
                    "month": str(month),
                    "active_users": active,
                    "logins": int(max(0, rng.poisson(max(active, 1) * (14 - 8 * r)))),
                    "features_used": int(np.clip(rng.normal(9 - 6 * r, 1.5), 1, 12)),
                    "support_tickets": int(rng.poisson(1 + 6 * r)),
                    "escalations": int(rng.poisson(0.05 + 1.2 * r**2)),
                }
            )
    usage = pd.DataFrame(rows)

    # Latest NPS response per account (-100..100 scale collapsed to 0..10 score).
    accounts["nps_score"] = np.clip(np.round(rng.normal(9 - 7 * risk, 1.4)), 0, 10).astype(int)

    # Churn outcome: logistic on hidden risk plus noise.
    logit = -4.2 + 8.5 * risk + rng.normal(0, 0.8, n_accounts)
    accounts["churned"] = (rng.random(n_accounts) < 1 / (1 + np.exp(-logit))).astype(int)

    return accounts, usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounts", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    accounts, usage = generate(args.accounts, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(args.out / "accounts.csv", index=False)
    usage.to_csv(args.out / "monthly_usage.csv", index=False)
    print(
        f"Wrote {len(accounts)} accounts and {len(usage)} usage rows to {args.out} "
        f"(churn rate {accounts.churned.mean():.1%})"
    )


if __name__ == "__main__":
    main()
