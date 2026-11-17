"""Load the CSVs into SQLite and run the SQL KPI queries in ``sql/``.

Usage:
    python -m churn.db              # build data/customers.db and print every KPI query
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from churn.features import DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DB_PATH = DATA_DIR / "customers.db"


def build_database(data_dir: Path = DATA_DIR, db_path: Path = DB_PATH) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    pd.read_csv(data_dir / "accounts.csv").to_sql("accounts", conn, if_exists="append", index=False)
    pd.read_csv(data_dir / "monthly_usage.csv").to_sql("monthly_usage", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def named_queries(path: Path = SQL_DIR / "kpi_queries.sql") -> dict[str, str]:
    """Split a .sql file into {name: query} using ``-- name: <query_name>`` markers."""
    text = path.read_text()
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(named_queries()[name], conn)


def main() -> None:
    conn = build_database()
    pd.set_option("display.width", 160)
    for name, sql in named_queries().items():
        print(f"\n=== {name} ===")
        print(pd.read_sql_query(sql, conn).to_string(index=False))


if __name__ == "__main__":
    main()
