CREDIT RISK INTELLIGENCE SYSTEM — Full Decision Log

THE BUSINESS PROBLEM
Predict whether a borrower will default on a LendingClub loan before the loan is issued. A lender has two costly mistakes — approving someone who defaults (loses money) and rejecting someone who would have paid (loses revenue). The model needs to minimize both.

DATASET

Started with 50,000 rows, 151 columns
Final clean dataset: 44,252 rows, 76 columns


TARGET VARIABLE

Binary 0/1 — 1 means default, 0 means paid off
20.96% default rate — class imbalance exists, will need to handle during modeling


COLUMNS DROPPED AND WHY
57 high-missing columns dropped — columns missing more than a threshold of data are unreliable. A model can't learn from data that isn't there, and imputing heavily missing columns introduces more noise than signal.
12 leakage columns dropped — these columns contain information that would only be available after the loan outcome is known. Using them would make the model look accurate in training but fail completely in production. Example: last_fico_range_high — this is the borrower's FICO score recorded after the loan was issued, possibly right before they defaulted. In production you only have the FICO score at origination.
8 identifier columns dropped — columns like id and member_id are unique per row. A model trained on them would memorize row numbers, not learn patterns. Zero predictive value.

OUTLIER TREATMENT AND WHY
annual_inc capped at $265,000 — extreme income outliers (one person earning $9M) would distort model training. Tree-based models handle outliers better than linear models, but capping at the 99th percentile is standard practice. Business reason: a $9M income borrower is not representative of LendingClub's customer base.
dti values above 100 nulled — debt-to-income ratio above 100 means debt exceeds income by more than 100%. Mathematically possible but almost certainly a data entry error. Nulling is safer than capping because these values are likely wrong, not just extreme.
revol_util capped at 100% — revolving utilization above 100% is impossible by definition. Values above 100 are data errors, capped at 100.
1 zero income row removed — a borrower with zero income is either a data error or an edge case so rare it adds no value to training.

EDA FINDINGS AND BUSINESS IMPLICATIONS
Default rate by loan grade:

Grade A: 5.4% → Grade G: 57%
Clean ascending pattern confirms target variable is correctly defined
Business implication: expected losses nearly triple from Grade A to Grade B at $15,000 average loan size. Grade is the single strongest categorical predictor.

FICO score distribution:

Defaulters cluster at lower FICO scores (660–680 range)
Heavy overlap between defaulters and non-defaulters
Business implication: FICO alone can't separate defaulters. Useful signal but insufficient standalone. Median FICO of defaulters is roughly 60–70 points lower than non-defaulters.

Income distribution:

Both groups peak at $40,000–$60,000
Very heavy overlap — weakest signal of the three plots
Business implication: income alone is a poor predictor of default. A high earner can still default due to high debt, job loss, or poor financial behavior. This is why DTI (debt-to-income ratio) will be more useful than raw income.

Correlation heatmap:

last_fico_range_high (0.70) and last_fico_range_low (0.60) — strongest correlations with target but leakage. Will not be used as model features.
int_rate (0.31) — strong legitimate predictor. LendingClub charges higher rates to riskier borrowers, creating a feedback signal.
fico_range_low, fico_mid, fico_range_high — perfectly correlated at 1.00 with each other. Keep only fico_range_low, drop the other two.
loan_amnt and funded_amnt — perfectly correlated at 1.00. Same signal twice. Drop funded_amnt, keep loan_amnt.
acc_open_past_24mths and num_tl_op_past_12m — correlated at 0.77. Both measure recent credit activity. Will revisit during feature engineering.