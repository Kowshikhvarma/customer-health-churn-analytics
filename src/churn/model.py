"""Train and evaluate a churn prediction model.

Compares an interpretable Logistic Regression against a Random Forest with
5-fold cross-validation, keeps the better one, then reports hold-out metrics
and permutation feature importance.

Usage:
    python -m churn.model
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.features import DATA_DIR, FEATURE_COLUMNS, build_features, load_raw

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RANDOM_STATE = 42


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def training_frame(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    accounts, usage = load_raw(data_dir)
    feats = build_features(accounts, usage)
    df = feats.merge(accounts[["account_id", "churned"]], on="account_id")
    return df[FEATURE_COLUMNS], df["churned"]


def train(data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR) -> dict:
    X, y = training_frame(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {
        name: float(cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean())
        for name, model in candidate_models().items()
    }
    best_name = max(cv_scores, key=cv_scores.get)
    best = candidate_models()[best_name].fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    importance = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
    )
    top_features = (
        pd.Series(importance.importances_mean, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .round(4)
    )

    metrics = {
        "model": best_name,
        "cv_roc_auc": {k: round(v, 4) for k, v in cv_scores.items()},
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "feature_importance": top_features.to_dict(),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, models_dir / "churn_model.joblib")
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(models_dir: Path = MODELS_DIR) -> Pipeline:
    return joblib.load(models_dir / "churn_model.joblib")


def predict_proba(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(features[FEATURE_COLUMNS])[:, 1], index=features.index)


def main() -> None:
    m = train()
    print(f"Selected model: {m['model']}")
    print(f"CV ROC-AUC: {m['cv_roc_auc']}")
    print(f"Hold-out ROC-AUC: {m['test_roc_auc']} | PR-AUC: {m['test_pr_auc']}")
    print("Top drivers of churn:")
    for feat, val in list(m["feature_importance"].items())[:5]:
        print(f"  {feat:<22} {val:.4f}")


if __name__ == "__main__":
    main()
