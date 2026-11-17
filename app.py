"""Streamlit dashboard for the Customer Health & Churn project.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from churn import generate_data, model  # noqa: E402
from churn.features import DATA_DIR  # noqa: E402
from churn.health_score import WEIGHTS, tier_summary  # noqa: E402
from churn.pipeline import score_all  # noqa: E402
from churn.playbook import priority_queue  # noqa: E402

TIER_COLORS = {"Healthy": "#2e9e6b", "At Risk": "#e0a526", "Critical": "#d64545"}

st.set_page_config(page_title="Customer Health & Churn", page_icon="📊", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, dict]:
    if not (DATA_DIR / "accounts.csv").exists():
        accounts, usage = generate_data.generate()
        DATA_DIR.mkdir(exist_ok=True)
        accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
        usage.to_csv(DATA_DIR / "monthly_usage.csv", index=False)
    if not (model.MODELS_DIR / "churn_model.joblib").exists():
        model.train()
    metrics = json.loads((model.MODELS_DIR / "metrics.json").read_text())
    return score_all(), metrics


def inr(x: float) -> str:
    """Format rupees in lakhs / crores."""
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    return f"₹{x / 1e5:.1f} L"


scored, metrics = load()

# ---- Sidebar filters -------------------------------------------------------
st.sidebar.header("Filters")
csms = st.sidebar.multiselect("CSM owner", sorted(scored.csm_owner.unique()))
plans = st.sidebar.multiselect("Plan", ["Starter", "Growth", "Enterprise"])
industries = st.sidebar.multiselect("Industry", sorted(scored.industry.unique()))
active_only = st.sidebar.checkbox("Hide accounts already churned", value=False)

view = scored.copy()
if csms:
    view = view[view.csm_owner.isin(csms)]
if plans:
    view = view[view.plan.isin(plans)]
if industries:
    view = view[view.industry.isin(industries)]
if active_only:
    view = view[view.churned == 0]

st.sidebar.markdown("---")
st.sidebar.caption("Health score weights")
st.sidebar.json(WEIGHTS)

# ---- Header KPIs -----------------------------------------------------------
st.title("📊 Customer Health & Churn Dashboard")
st.caption("Synthetic B2B SaaS portfolio. Health score + ML churn risk + next best action for every account.")

if view.empty:
    st.warning("No accounts match these filters.")
    st.stop()

total_arr = view.arr_inr.sum()
grr = 1 - view.loc[view.churned == 1, "arr_inr"].sum() / total_arr
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accounts", f"{len(view):,}")
c2.metric("Total ARR", inr(total_arr))
c3.metric("Gross revenue retention", f"{grr:.1%}")
c4.metric("Avg health score", f"{view.health_score.mean():.0f}/100")
c5.metric("Critical accounts", int((view.health_tier == "Critical").sum()))

tab_overview, tab_queue, tab_account, tab_model = st.tabs(
    ["Portfolio overview", "Priority queue", "Account drill-down", "Model performance"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            view, x="health_score", color="health_tier", nbins=30,
            color_discrete_map=TIER_COLORS, title="Health score distribution",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        summary = tier_summary(view).reset_index()
        fig = px.bar(
            summary, x="health_tier", y="churn_rate", color="health_tier", text_auto=".0%",
            color_discrete_map=TIER_COLORS, title="Actual churn rate by health tier",
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        by_csm = (
            view.groupby("csm_owner")
            .agg(accounts=("account_id", "count"), arr=("arr_inr", "sum"),
                 avg_health=("health_score", "mean"), churn_risk=("churn_probability", "mean"))
            .round(2).sort_values("arr", ascending=False).reset_index()
        )
        st.subheader("CSM book of business")
        st.dataframe(by_csm, hide_index=True, width="stretch")
    with right:
        fig = px.scatter(
            view, x="health_score", y="churn_probability", size="arr_inr", color="health_tier",
            color_discrete_map=TIER_COLORS, hover_name="company",
            title="Rules-based health vs ML churn probability",
        )
        st.plotly_chart(fig, width="stretch")

with tab_queue:
    st.subheader("Accounts to work this week (sorted by ARR at risk)")
    top_n = st.slider("How many accounts", 10, 100, 25, step=5)
    queue = priority_queue(view[view.churned == 0] if not active_only else view, top_n=top_n)
    st.dataframe(queue, hide_index=True, width="stretch")
    st.download_button("Download as CSV", queue.to_csv(index=False), "csm_priority_queue.csv", "text/csv")

with tab_account:
    company = st.selectbox("Choose an account", sorted(view.company))
    acc = view[view.company == company].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Health score", f"{acc.health_score:.0f}", acc.health_tier)
    b.metric("Churn probability", f"{acc.churn_probability:.0%}")
    c.metric("ARR", inr(acc.arr_inr))
    d.metric("Days to renewal", int(acc.days_to_renewal))
    comps = pd.DataFrame(
        {"component": list(WEIGHTS), "score": [acc[f"{k}_score"] for k in WEIGHTS]}
    )
    fig = px.line_polar(comps, r="score", theta="component", line_close=True, range_r=[0, 100],
                        title="Health components")
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, width="stretch")
    usage = pd.read_csv(DATA_DIR / "monthly_usage.csv")
    usage = usage[usage.account_id == acc.account_id]
    st.plotly_chart(
        px.line(usage, x="month", y=["active_users", "support_tickets"], markers=True,
                title="Monthly active users and tickets"),
        width="stretch",
    )

with tab_model:
    st.subheader(f"Selected model: `{metrics['model']}`")
    m1, m2, m3 = st.columns(3)
    m1.metric("Hold-out ROC-AUC", metrics["test_roc_auc"])
    m2.metric("Hold-out PR-AUC", metrics["test_pr_auc"])
    m3.metric("Base churn rate", f"{metrics['test_churn_rate']:.1%}")
    st.write("Cross-validated ROC-AUC per candidate:", metrics["cv_roc_auc"])
    imp = pd.Series(metrics["feature_importance"]).sort_values()
    st.plotly_chart(
        px.bar(imp, orientation="h", title="What drives churn (permutation importance)"),
        width="stretch",
    )
