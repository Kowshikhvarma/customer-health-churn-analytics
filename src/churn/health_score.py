"""Transparent, rules-based customer health score (0-100).

A CSM team needs a score it can explain to an account owner in one sentence,
so this is deliberately a weighted average of five easy-to-read components
rather than a black box. The ML model in ``model.py`` is used alongside it to
catch risk the rules miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "adoption": 0.30,      # are they using the seats they pay for?
    "engagement": 0.20,    # is usage growing or shrinking?
    "support": 0.20,       # are they struggling / escalating?
    "sentiment": 0.20,     # what did they say in NPS?
    "relationship": 0.10,  # when did a CSM last talk to them?
}

TIERS = [(70, "Healthy"), (45, "At Risk"), (0, "Critical")]


def _scale(series: pd.Series, low: float, high: float) -> pd.Series:
    """Linearly map [low, high] to [0, 100] and clip."""
    return ((series - low) / (high - low) * 100).clip(0, 100)


def component_scores(features: pd.DataFrame) -> pd.DataFrame:
    comps = pd.DataFrame(index=features.index)
    comps["adoption"] = _scale(features["seat_adoption"], 0.2, 0.9)
    comps["engagement"] = _scale(features["adoption_trend"], -0.25, 0.05)
    support_penalty = features["tickets_last_3m"] * 4 + features["escalations_last_3m"] * 15
    comps["support"] = (100 - support_penalty).clip(0, 100)
    comps["sentiment"] = features["nps_score"] * 10
    comps["relationship"] = (100 - features["days_since_csm_touch"] * 1.5).clip(0, 100)
    return comps


def tier_for(score: float) -> str:
    for threshold, label in TIERS:
        if score >= threshold:
            return label
    return TIERS[-1][1]


def score_accounts(features: pd.DataFrame) -> pd.DataFrame:
    """Return features with component scores, overall health_score and tier."""
    comps = component_scores(features)
    weights = pd.Series(WEIGHTS)
    out = features.copy()
    for name in WEIGHTS:
        out[f"{name}_score"] = comps[name].round(1)
    out["health_score"] = (comps[list(WEIGHTS)] * weights).sum(axis=1).round(1)
    out["health_tier"] = out["health_score"].apply(tier_for)
    out["weakest_area"] = comps[list(WEIGHTS)].idxmin(axis=1)
    return out


def tier_summary(scored: pd.DataFrame, churn_col: str = "churned") -> pd.DataFrame:
    """Accounts and actual churn rate per tier: a quick check that the score works."""
    order = [label for _, label in TIERS]
    grp = scored.groupby("health_tier")
    summary = pd.DataFrame({"accounts": grp.size()})
    if churn_col in scored:
        summary["churn_rate"] = grp[churn_col].mean().round(3)
    return summary.reindex([t for t in order if t in summary.index])


def weights_are_valid() -> bool:
    return bool(np.isclose(sum(WEIGHTS.values()), 1.0))
