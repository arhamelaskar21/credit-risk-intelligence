import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
EDA Steps 1-9: Feature analysis, imputation, correlation, visualization,
feature engineering, and data export.

Run this from the project root with the virtual environment activated.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# LOAD & REPRODUCE PREVIOUS CLEANING (from notebook cells 28-41)
# ============================================================
print("=" * 70)
print("LOADING DATA & REPRODUCING PREVIOUS CLEANING")
print("=" * 70)

DATA_PATH = r'C:\Users\Zakir Elaskar\credit-risk-system\credit-risk-system\data\raw\accepted_2007_to_2018Q4.csv.gz'
OUTPUT_DIR = r'C:\Users\Zakir Elaskar\credit-risk-system\credit-risk-system\data\processed'
PLOT_DIR = r'C:\Users\Zakir Elaskar\credit-risk-system\credit-risk-system\notebooks'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, compression='gzip', low_memory=False, nrows=50000)
print(f"Loaded: {df.shape}")

# Keep only conclusive loans
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)'])]
df['target'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)

# Drop >50% missing
missing_pct = (df.isnull().sum() / len(df)) * 100
cols_over_50 = missing_pct[missing_pct > 50].index.tolist()
df = df.drop(columns=cols_over_50)

# Drop leakage columns
leakage_cols = ['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
                'total_rec_prncp','total_rec_int','total_rec_late_fee',
                'recoveries','collection_recovery_fee','last_pymnt_d',
                'last_pymnt_amnt','last_credit_pull_d']
leakage_cols = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=leakage_cols)

# Drop useless columns
useless_cols = ['id','url','title','zip_code','pymnt_plan','policy_code','application_type']
useless_cols = [c for c in useless_cols if c in df.columns]
df = df.drop(columns=useless_cols)

# Fix outliers
income_cap = df['annual_inc'].quantile(0.99)
df['annual_inc'] = df['annual_inc'].clip(upper=income_cap)
df['dti'] = df['dti'].where(df['dti'] <= 100, other=None)
df['revol_util'] = df['revol_util'].where(df['revol_util'] <= 100, other=None)

print(f"After previous cleaning: {df.shape}")
print(f"Default rate: {df['target'].mean()*100:.2f}%\n")

# ============================================================
# STEP 1: FEATURE TYPE ANALYSIS
# ============================================================
print("=" * 70)
print("STEP 1: FEATURE TYPE ANALYSIS")
print("=" * 70)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")
print(f"\n--- Categorical columns and their unique value counts ---")
for col in cat_cols:
    nunique = df[col].nunique()
    print(f"  {col}: {nunique} unique values")
    if nunique < 15:
        print(f"    Values: {df[col].unique().tolist()}")

# ============================================================
# STEP 2: CLEAN TYPE-CONFUSED COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: CLEAN TYPE-CONFUSED COLUMNS")
print("=" * 70)

# Fix 'term'
df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(int)
print("term unique values after fix:", df['term'].unique())

# Fix 'emp_length'
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}
df['emp_length'] = df['emp_length'].map(emp_map)
print(f"\nemp_length missing after mapping: {df['emp_length'].isnull().sum()} "
      f"({df['emp_length'].isnull().mean()*100:.1f}%)")

# Fix 'earliest_cr_line'
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='mixed', errors='coerce')
print(f"\nearliest_cr_line sample:\n{df['earliest_cr_line'].head()}")

# ============================================================
# STEP 3: MISSING VALUE IMPUTATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: MISSING VALUE IMPUTATION")
print("=" * 70)

remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
print(f"Columns still with missing values: {len(remaining_missing)}\n")
print(remaining_missing)

# Strategy 1: MNAR columns
for mnar_col in ['mths_since_last_delinq', 'mths_since_recent_bc', 'mths_since_recent_inq']:
    if mnar_col in df.columns and df[mnar_col].isnull().sum() > 0:
        fill_val = df[mnar_col].max() * 2
        df[mnar_col] = df[mnar_col].fillna(fill_val)
        print(f"\n{mnar_col}: filled NaN with {fill_val} (MNAR — never happened)")

# Strategy 2: emp_length with median
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())

# Strategy 3: remaining numericals with median
num_cols_all = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols_all:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Strategy 4: remaining categoricals with mode
cat_cols_all = df.select_dtypes(include=['object']).columns
for col in cat_cols_all:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Handle datetime columns
for col in df.select_dtypes(include=['datetime64']).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

total_missing = df.isnull().sum().sum()
print(f"\n[OK] Total missing values remaining: {total_missing}")
print(f"Shape: {df.shape}")

# ============================================================
# STEP 4: CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: CORRELATION ANALYSIS")
print("=" * 70)

num_df = df.select_dtypes(include=['int64', 'float64'])
target_corr = num_df.corr()['target'].drop('target').sort_values(key=abs, ascending=False)

print("Top 20 features correlated with default:\n")
print(target_corr.head(20).to_string())

# Feature-to-feature correlation (redundancy)
print("\n\n--- Highly Correlated Feature Pairs (|r| > 0.85) ---\n")
corr_matrix = num_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for col in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, col] > 0.85:
            high_corr_pairs.append({
                'feature_1': idx,
                'feature_2': col,
                'correlation': round(upper.loc[idx, col], 3)
            })
high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
print(f"Found {len(high_corr_df)} pairs:\n")
print(high_corr_df.to_string(index=False))

# Heatmap
top_features = target_corr.head(15).index.tolist() + ['target']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap — Top Predictive Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[PLOT] Saved: correlation_heatmap.png")

# ============================================================
# STEP 5: DROP REDUNDANT FEATURES
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: DROP REDUNDANT FEATURES")
print("=" * 70)

redundant_cols = [
    'funded_amnt', 'funded_amnt_inv',
    'fico_range_high', 'last_fico_range_high',
    'num_sats', 'installment',
]
redundant_cols = [c for c in redundant_cols if c in df.columns]
df = df.drop(columns=redundant_cols)
print(f"Dropped {len(redundant_cols)} redundant features: {redundant_cols}")
print(f"Remaining columns: {df.shape[1]}")

# ============================================================
# STEP 6: UNIVARIATE / BIVARIATE VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: VISUALIZATIONS")
print("=" * 70)

# Plot 1: Default rates by categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

grade_default = df.groupby('grade')['target'].mean().sort_index() * 100
axes[0,0].bar(grade_default.index, grade_default.values,
              color=['#2ecc71','#27ae60','#f39c12','#e67e22','#e74c3c','#c0392b','#8e44ad'])
axes[0,0].set_title('Default Rate by Grade', fontsize=13)
axes[0,0].set_ylabel('Default Rate (%)')
axes[0,0].set_xlabel('Grade (A=best, G=worst)')

term_default = df.groupby('term')['target'].mean() * 100
axes[0,1].bar(term_default.index.astype(str), term_default.values, color=['#3498db','#e74c3c'])
axes[0,1].set_title('Default Rate by Loan Term', fontsize=13)
axes[0,1].set_ylabel('Default Rate (%)')
axes[0,1].set_xlabel('Term (months)')

home_default = df.groupby('home_ownership')['target'].mean().sort_values(ascending=False) * 100
axes[1,0].barh(home_default.index, home_default.values, color='#9b59b6')
axes[1,0].set_title('Default Rate by Home Ownership', fontsize=13)
axes[1,0].set_xlabel('Default Rate (%)')

purpose_default = df.groupby('purpose')['target'].mean().sort_values(ascending=False) * 100
top_purpose = purpose_default.head(8)
axes[1,1].barh(top_purpose.index, top_purpose.values, color='#1abc9c')
axes[1,1].set_title('Default Rate by Purpose (Top 8)', fontsize=13)
axes[1,1].set_xlabel('Default Rate (%)')

plt.suptitle('Default Rates Across Key Categorical Features', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'default_rates_categorical.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[PLOT] Saved: default_rates_categorical.png")

# Plot 2: Numerical distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
plot_features = ['int_rate', 'annual_inc', 'dti', 'fico_range_low', 'revol_util', 'loan_amnt']

for idx, feat in enumerate(plot_features):
    ax = axes[idx // 3, idx % 3]
    df[df['target'] == 0][feat].hist(bins=40, alpha=0.5, label='Good Loan', color='#2ecc71', ax=ax, density=True)
    df[df['target'] == 1][feat].hist(bins=40, alpha=0.5, label='Default', color='#e74c3c', ax=ax, density=True)
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylabel('Density')

plt.suptitle('Feature Distributions: Good Loans vs Defaults', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'distributions_by_target.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[PLOT] Saved: distributions_by_target.png")

# ============================================================
# STEP 7: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 70)

# Feature 1: Loan-to-Income Ratio
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# Feature 2: FICO Average
df['fico_avg'] = df['fico_range_low'] + 2

# Feature 3: Credit History Length
df['issue_d'] = pd.to_datetime(df['issue_d'], format='mixed', errors='coerce')
df['credit_history_months'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 30).round()

# Feature 4: Delinquency Recency Flag
if 'mths_since_last_delinq' in df.columns:
    median_delinq = df['mths_since_last_delinq'].median()
    df['ever_delinquent'] = (df['mths_since_last_delinq'] < median_delinq).astype(int)

print("New features created:")
new_features = ['loan_to_income', 'fico_avg', 'credit_history_months', 'ever_delinquent']
for f in new_features:
    if f in df.columns:
        print(f"  {f}: mean={df[f].mean():.4f}, min={df[f].min():.4f}, max={df[f].max():.4f}")

# Check if they're predictive
print("\nCorrelation with target (default):")
for f in new_features:
    if f in df.columns:
        corr = df[f].corr(df['target'])
        strength = "STRONG" if abs(corr) > 0.15 else "moderate" if abs(corr) > 0.05 else "weak"
        print(f"  {f:30s}: {corr:+.4f}  ({strength})")

# ============================================================
# STEP 8: DROP COLUMNS NO LONGER NEEDED
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: FINAL COLUMN CLEANUP")
print("=" * 70)

cols_to_drop_final = [
    'loan_status', 'issue_d', 'earliest_cr_line',
    'emp_title', 'hardship_flag',
]
cols_to_drop_final = [c for c in cols_to_drop_final if c in df.columns]
df = df.drop(columns=cols_to_drop_final)

print(f"Dropped {len(cols_to_drop_final)} final columns: {cols_to_drop_final}")
print(f"Final shape: {df.shape}")
print(f"\nColumn types:")
print(f"  Numerical: {len(df.select_dtypes(include=['int64','float64']).columns)}")
print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
print(f"\nRemaining categorical columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# ============================================================
# STEP 9: SAVE CLEANED DATA
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: SAVE CLEANED DATA")
print("=" * 70)

df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_loans.csv'), index=False)
df.to_parquet(os.path.join(OUTPUT_DIR, 'cleaned_loans.parquet'), index=False)

csv_size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, 'cleaned_loans.csv')) / 1e6
parquet_size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, 'cleaned_loans.parquet')) / 1e6

print(f"[OK] Saved cleaned data to: {OUTPUT_DIR}")
print(f"   CSV size:     {csv_size_mb:.1f} MB")
print(f"   Parquet size: {parquet_size_mb:.1f} MB")
print(f"   Shape: {df.shape}")
print(f"   Target distribution:\n{df['target'].value_counts().to_string()}")
print(f"   Default rate: {df['target'].mean()*100:.2f}%")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETE — Data is ready for modeling (notebook 02)")
print("=" * 70)
print(f"\nFinal columns ({df.shape[1]}):")
print(df.columns.tolist())
