-- Customer Success KPI queries.
-- Each query starts with a "-- name:" marker so it can be run from Python (churn/db.py).

-- name: gross_revenue_retention
-- Share of ARR kept after churn (no expansion counted).
SELECT
    ROUND(100.0 * SUM(CASE WHEN churned = 0 THEN arr_inr ELSE 0 END) / SUM(arr_inr), 1) AS grr_pct,
    SUM(arr_inr)                                          AS total_arr_inr,
    SUM(CASE WHEN churned = 1 THEN arr_inr ELSE 0 END)    AS churned_arr_inr,
    ROUND(100.0 * AVG(churned), 1)                        AS logo_churn_pct
FROM accounts;

-- name: churn_by_plan
SELECT
    plan,
    COUNT(*)                         AS accounts,
    ROUND(100.0 * AVG(churned), 1)   AS churn_pct,
    SUM(arr_inr)                     AS arr_inr
FROM accounts
GROUP BY plan
ORDER BY arr_inr DESC;

-- name: csm_portfolio
-- Book of business per CSM: workload and outcomes.
SELECT
    csm_owner,
    COUNT(*)                                   AS accounts,
    SUM(arr_inr)                               AS arr_inr,
    ROUND(AVG(nps_score), 1)                   AS avg_nps,
    ROUND(AVG(days_since_csm_touch), 1)        AS avg_days_since_touch,
    ROUND(100.0 * AVG(churned), 1)             AS churn_pct
FROM accounts
GROUP BY csm_owner
ORDER BY arr_inr DESC;

-- name: onboarding_speed_vs_churn
-- Do slow onboardings churn more? Bucket accounts by time-to-onboard.
SELECT
    CASE
        WHEN onboarding_days <= 21 THEN '1. <= 3 weeks'
        WHEN onboarding_days <= 35 THEN '2. 3-5 weeks'
        ELSE '3. > 5 weeks'
    END                                AS onboarding_bucket,
    COUNT(*)                           AS accounts,
    ROUND(100.0 * AVG(churned), 1)     AS churn_pct
FROM accounts
GROUP BY onboarding_bucket
ORDER BY onboarding_bucket;

-- name: nps_breakdown
-- Promoters 9-10, Passives 7-8, Detractors 0-6. NPS = %promoters - %detractors.
SELECT
    ROUND(100.0 * SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END) / COUNT(*), 1) AS promoters_pct,
    ROUND(100.0 * SUM(CASE WHEN nps_score BETWEEN 7 AND 8 THEN 1 ELSE 0 END) / COUNT(*), 1) AS passives_pct,
    ROUND(100.0 * SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END) / COUNT(*), 1) AS detractors_pct,
    ROUND(100.0 * (SUM(CASE WHEN nps_score >= 9 THEN 1 ELSE 0 END)
                 - SUM(CASE WHEN nps_score <= 6 THEN 1 ELSE 0 END)) / COUNT(*), 1) AS nps
FROM accounts;

-- name: adoption_trend
-- Platform-wide seat adoption by month (active users / purchased seats).
SELECT
    u.month,
    ROUND(100.0 * SUM(u.active_users) / SUM(a.seats), 1) AS seat_adoption_pct,
    SUM(u.support_tickets)                                AS tickets,
    SUM(u.escalations)                                    AS escalations
FROM monthly_usage u
JOIN accounts a ON a.account_id = u.account_id
GROUP BY u.month
ORDER BY u.month;

-- name: upcoming_renewals_at_risk
-- Active accounts renewing soonest with weak signals (low NPS or stale contact).
SELECT
    account_id,
    company,
    csm_owner,
    arr_inr,
    renewal_date,
    nps_score,
    days_since_csm_touch
FROM accounts
WHERE churned = 0
  AND (nps_score <= 6 OR days_since_csm_touch > 30)
ORDER BY renewal_date
LIMIT 10;
