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
FEATURE ENGINEERING AND ENCODING DECISIONS

Grade → Ordinal Encoding (A=1 to G=7)
Ordinal encoding preserves the natural risk order assigned by LendingClub. One-hot encoding would treat all grades as equally distant categories, throwing away the credit risk ordering. A positive model coefficient now directly means higher grade = higher risk.

Term → Binary Encoding (36 months=0, 60 months=1)
Binary because there are only two values. 60 months encoded as 1 because longer loan term = higher default risk. This convention keeps the coefficient sign interpretable — positive coefficient means longer term = more risk.

Verification Status → One-Hot Encoding (drop_first=True)
No natural order exists between Verified, Source Verified, and Not Verified. One-hot encoding avoids imposing a fake ordering. drop_first=True drops Not Verified as the baseline category to avoid multicollinearity — if both dummies are 0, the model infers Not Verified automatically.

Earliest Credit Line → Credit History Months
Raw date is meaningless to the model. Converted to number of months between earliest credit line and December 2018 (end of dataset). Using a fixed reference date avoids data leakage — using today's date would add 8+ years of history that didn't exist when the loan was issued. Longer credit history = lower uncertainty = lower default risk.

addr_state → Dropped
50+ unique values would require 49 dummy columns, adding noise to the baseline model. Some states have too few loans for reliable pattern learning (data sparsity). Will revisit in later iterations using regional grouping.

Bool columns → Converted to int64
Logistic regression requires numeric input. All 18 boolean dummy columns cast to int64.

Final encoded dataset: 44,251 rows, 79 columns, all numeric (float64 + int64). Saved to data/processed/engineered_loans.parquet.

MODELING — LOGISTIC REGRESSION BASELINE

Train/test split: 80/20, stratified on target to preserve 20.96% default rate in both sets.
Feature scaling: StandardScaler fit on X_train only — prevents test set statistics leaking into scaling.

Results:
- LR no class balancing: Accuracy 0.80, Class 1 Recall 0.20, AUC 0.74
- LR class_weight=balanced: Accuracy 0.69, Class 1 Recall 0.66, Class 1 F1 0.47, AUC 0.74
- LR balanced threshold=0.30: Class 1 Recall 0.91, Class 1 Precision 0.27

Business decision: Use balanced model at 0.50 threshold as baseline. Recall 0.66 is acceptable
for a baseline — catching 66% of defaulters vs 20% justifies the drop in accuracy.
AUC of 0.74 is the benchmark all future models must beat.

Top predictors by coefficient magnitude:
1. int_rate (0.45) — absorbs grade signal due to correlation
2. term (0.36) — 60-month loans riskier
3. dti (0.17) — debt burden
4. mo_sin_old_rev_tl_op (-0.20) — long credit history reduces risk

Next: XGBoost — target AUC > 0.80, Class 1 Recall > 0.70
