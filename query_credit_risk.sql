-- ============================================================
-- Credit Risk: Data extraction & feature engineering
-- Dataset: German Credit (BigQuery, dataset `credit`)
-- ============================================================


-- ============================================================
-- SECTION 1: EXPLORATION
-- ============================================================

-- 1.1: Preview the first rows
SELECT *
FROM `progetto-xai.credit.german_credit`
LIMIT 20;


-- 1.2: Target distribution
SELECT
  Risk,
  COUNT(*) AS n
FROM `progetto-xai.credit.german_credit`
GROUP BY Risk;


-- 1.3: Default rate by loan purpose
SELECT
  Purpose,
  COUNT(*) AS total,
  COUNTIF(Risk = 'bad') AS bad_payers,
  ROUND(COUNTIF(Risk = 'bad') * 100.0 / COUNT(*), 1) AS pct_default
FROM `progetto-xai.credit.german_credit`
GROUP BY Purpose
ORDER BY pct_default DESC;


-- 1.4: Default rate by age band
SELECT
  CASE
    WHEN Age < 30 THEN '1_under_30'
    WHEN Age < 40 THEN '2_30_39'
    WHEN Age < 50 THEN '3_40_49'
    ELSE '4_50_plus'
  END AS age_band,
  COUNT(*) AS total,
  ROUND(COUNTIF(Risk = 'bad') * 100.0 / COUNT(*), 1) AS pct_default
FROM `progetto-xai.credit.german_credit`
GROUP BY age_band
ORDER BY age_band;


-- ============================================================
-- SECTION 2: Job codes 0/1/2/3 translated into readable descriptions
-- ============================================================

-- 2.1: Create the lookup table
CREATE OR REPLACE TABLE `progetto-xai.credit.job_lookup` AS
SELECT * FROM UNNEST([
  STRUCT(0 AS Job, 'unskilled_non_resident' AS job_desc),
  STRUCT(1, 'unskilled_resident'),
  STRUCT(2, 'skilled'),
  STRUCT(3, 'highly_skilled')
]);


-- 2.2: JOIN that enriches the data with the job description
SELECT g.Age, g.Job, j.job_desc, g.Risk
FROM `progetto-xai.credit.german_credit` g
LEFT JOIN `progetto-xai.credit.job_lookup` j
  ON g.Job = j.Job
LIMIT 20;


-- ============================================================
-- SECTION 3: FEATURE VIEW
-- ============================================================

-- This view is the analyst-facing feature table: it includes some readable,
-- redundant columns (age_band, job_desc, checking_missing) that are convenient
-- for SQL reporting/BI. The modeling script in Python keeps only ONE encoding of
-- each signal to avoid collinearity.
--
-- Two columns below (amount_vs_purpose_avg, high_amount) use cross-row statistics
-- computed over the WHOLE table. That is fine for exploratory SQL, but it would
-- leak test information into a model, so the Python pipeline DROPS them and
-- recomputes equivalents using training-set statistics only.
CREATE OR REPLACE VIEW `progetto-xai.credit.credit_features` AS
SELECT
  g.Age,
  g.Sex,
  g.Job,
  j.job_desc,
  g.Housing,
  g.`Saving accounts`   AS Saving_accounts,
  g.`Checking account`  AS Checking_account,
  g.`Credit amount`     AS Credit_amount,
  g.Duration,
  g.Purpose,
  -- TARGET 0/1 (1 = bad payer / default)
  CASE WHEN g.Risk = 'bad' THEN 1 ELSE 0 END AS default_flag,
  -- feature 1: age band  [reporting only — Python uses continuous Age instead]
  CASE
    WHEN g.Age < 30 THEN 'under_30'
    WHEN g.Age < 40 THEN '30_39'
    WHEN g.Age < 50 THEN '40_49'
    ELSE '50_plus'
  END AS age_band,
  -- feature 2: approximate monthly installment (amount / duration)  [row-local]
  ROUND(g.`Credit amount` / g.Duration, 1) AS amount_per_month,
  -- feature 3: missing checking account flag  [reporting only — Python uses the
  --            Checking_account 'missing' one-hot category instead]
  CASE WHEN g.`Checking account` = 'NA' THEN 1 ELSE 0 END AS checking_missing,
  -- feature 4: deviation of the amount from its purpose average  [LEAKY: global
  --            window stat — Python recomputes it on the train set only]
  ROUND(g.`Credit amount` - AVG(g.`Credit amount`) OVER (PARTITION BY g.Purpose), 0) AS amount_vs_purpose_avg,
  -- feature 5: credit amount as a multiple of the age  [row-local]
  ROUND(g.`Credit amount` / NULLIF(g.Age, 0), 1) AS amount_per_age,
  -- feature 6: long-term loan flag (duration above ~2 years)  [row-local]
  CASE WHEN g.Duration > 24 THEN 1 ELSE 0 END AS long_term_loan,
  -- feature 7: high amount flag, relative to the whole portfolio  [LEAKY: global
  --            quantile — Python recomputes it from the train 75th percentile]
  CASE
    WHEN g.`Credit amount` > (
      SELECT APPROX_QUANTILES(`Credit amount`, 4)[OFFSET(3)]
      FROM `progetto-xai.credit.german_credit`
    ) THEN 1 ELSE 0
  END AS high_amount,
  -- feature 8: has no savings info AND no checking info (thin financial profile)
  CASE
    WHEN g.`Saving accounts` = 'NA' AND g.`Checking account` = 'NA' THEN 1 ELSE 0
  END AS thin_file,
FROM `progetto-xai.credit.german_credit` g
LEFT JOIN `progetto-xai.credit.job_lookup` j
  ON g.Job = j.Job;


-- ============================================================
-- SECTION 4: CHECK
-- ============================================================

-- 4.1: Verify the view is populated and has the new columns
SELECT *
FROM `progetto-xai.credit.credit_features`
LIMIT 20;