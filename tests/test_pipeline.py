from pathlib import Path

import pandas as pd
import pytest

from churn import db, model
from churn.features import FEATURE_COLUMNS, build_features
from churn.generate_data import generate
from churn.health_score import score_accounts, tier_for, tier_summary, weights_are_valid
from churn.playbook import priority_queue, recommend


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data")
    accounts, usage = generate(n_accounts=300, seed=7)
    accounts.to_csv(path / "accounts.csv", index=False)
    usage.to_csv(path / "monthly_usage.csv", index=False)
    return path


@pytest.fixture(scope="module")
def scored(dataset) -> pd.DataFrame:
    accounts = pd.read_csv(dataset / "accounts.csv", parse_dates=["signup_date", "renewal_date"])
    usage = pd.read_csv(dataset / "monthly_usage.csv")
    out = score_accounts(build_features(accounts, usage))
    return out.merge(accounts[["account_id", "company", "csm_owner", "plan", "arr_inr", "churned"]], on="account_id")


def test_generator_is_reproducible():
    a1, u1 = generate(50, seed=1)
    a2, u2 = generate(50, seed=1)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(u1, u2)
    assert a1.company.is_unique


def test_features_have_one_row_per_account(dataset):
    accounts = pd.read_csv(dataset / "accounts.csv", parse_dates=["signup_date", "renewal_date"])
    usage = pd.read_csv(dataset / "monthly_usage.csv")
    feats = build_features(accounts, usage)
    assert len(feats) == len(accounts)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert feats[FEATURE_COLUMNS].notna().all().all()
    assert feats.seat_adoption.between(0, 1).all()


def test_health_score_bounds_and_tiers(scored):
    assert weights_are_valid()
    assert scored.health_score.between(0, 100).all()
    assert set(scored.health_tier) <= {"Healthy", "At Risk", "Critical"}
    assert tier_for(85) == "Healthy" and tier_for(50) == "At Risk" and tier_for(10) == "Critical"


def test_health_score_separates_churners(scored):
    """Critical accounts should churn noticeably more than Healthy ones."""
    summary = tier_summary(scored)
    assert summary.loc["Critical", "churn_rate"] > summary.loc["Healthy", "churn_rate"] + 0.2


def test_model_beats_random(dataset, tmp_path):
    metrics = model.train(data_dir=dataset, models_dir=tmp_path)
    assert metrics["test_roc_auc"] > 0.6
    assert (tmp_path / "churn_model.joblib").exists()


def test_playbook_flags_urgent_renewals(scored):
    row = scored.iloc[0].copy()
    row["days_to_renewal"], row["health_tier"], row["weakest_area"] = 20, "Critical", "sentiment"
    assert recommend(row).startswith("URGENT")
    row["days_to_renewal"], row["health_tier"] = 200, "Healthy"
    assert not recommend(row).startswith("URGENT")


def test_priority_queue_sorted_by_risk(scored):
    queue = priority_queue(scored, top_n=10)
    assert len(queue) == 10
    assert queue.arr_at_risk_inr.is_monotonic_decreasing


def test_sql_kpis_run(dataset, tmp_path):
    conn = db.build_database(data_dir=dataset, db_path=tmp_path / "test.db")
    queries = db.named_queries()
    assert "gross_revenue_retention" in queries
    for name in queries:
        assert not db.run_query(conn, name).empty, name
