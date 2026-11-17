-- Schema for the customer success analytics database (SQLite / PostgreSQL compatible)

CREATE TABLE accounts (
    account_id            TEXT PRIMARY KEY,
    company               TEXT NOT NULL,
    industry              TEXT NOT NULL,
    plan                  TEXT NOT NULL CHECK (plan IN ('Starter', 'Growth', 'Enterprise')),
    csm_owner             TEXT NOT NULL,
    arr_inr               INTEGER NOT NULL,
    seats                 INTEGER NOT NULL,
    signup_date           DATE NOT NULL,
    renewal_date          DATE NOT NULL,
    onboarding_days       INTEGER NOT NULL,
    days_since_csm_touch  INTEGER NOT NULL,
    nps_score             INTEGER CHECK (nps_score BETWEEN 0 AND 10),
    churned               INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE monthly_usage (
    account_id       TEXT NOT NULL REFERENCES accounts(account_id),
    month            TEXT NOT NULL,          -- YYYY-MM
    active_users     INTEGER NOT NULL,
    logins           INTEGER NOT NULL,
    features_used    INTEGER NOT NULL,
    support_tickets  INTEGER NOT NULL,
    escalations      INTEGER NOT NULL,
    PRIMARY KEY (account_id, month)
);

CREATE INDEX idx_accounts_csm ON accounts (csm_owner);
CREATE INDEX idx_usage_month ON monthly_usage (month);
