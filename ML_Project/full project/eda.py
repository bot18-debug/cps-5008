import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for candidate in [
    os.path.join(SCRIPT_DIR, "customer_account_and_usage.csv"),
    os.path.join(SCRIPT_DIR, "..", "customer_account_and_usage.csv"),
]:
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break
else:
    raise FileNotFoundError("customer_account_and_usage.csv not found. Place it in the same folder as eda.py.")

df = pd.read_csv(CSV_PATH)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=150)
    plt.close()


print("=" * 60)
print("SECTION 1 - DATASET OVERVIEW")
print("=" * 60)

n_rows, n_cols = df.shape
churn_rate = df['Churn'].mean()
n_churners = int(df['Churn'].sum())
n_retained = n_rows - n_churners
imbalance_ratio = n_retained / n_churners

print(f"Rows: {n_rows:,}   Columns: {n_cols}")
print(f"Churners:  {n_churners:,}  ({churn_rate:.2%})")
print(f"Retained:  {n_retained:,}  ({1 - churn_rate:.2%})")
print(f"Class imbalance ratio (retained : churner): {imbalance_ratio:.1f}:1")
print("\nDtype summary:")
print(df.dtypes.value_counts())
print("\nFirst 3 rows:")
print(df.head(3).to_string())


print("\n" + "=" * 60)
print("SECTION 2 - MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found.")
else:
    missing_pct = (missing / n_rows * 100).round(2)
    missing_df = pd.DataFrame({"Count": missing, "Pct (%)": missing_pct})
    print(missing_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(missing.index, missing.values, color="#1f77b4")
    for i, (col, val) in enumerate(missing.items()):
        ax.text(i, val + 2, f"{val}\n({val/n_rows*100:.1f}%)",
                ha='center', va='bottom', fontsize=9)
    ax.set_title("Missing Values per Column", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Missing Values")
    ax.set_xlabel("")
    ax.set_xticks(range(len(missing)))
    ax.set_xticklabels(missing.index, rotation=15, ha='right')
    save("missing_values")
    print("\nChart saved: missing_values.png")


print("\n" + "=" * 60)
print("SECTION 3 - CHURN DISTRIBUTION")
print("=" * 60)
print(f"Non-churn: {n_retained:,} ({(n_retained/n_rows)*100:.2f}%)")
print(f"Churn:     {n_churners:,} ({churn_rate*100:.2f}%)")
print("NOTE: Accuracy is misleading here. Recall is the primary metric.")

fig, ax = plt.subplots(figsize=(6, 4))
counts = df['Churn'].value_counts().sort_index()
bars = ax.bar(['Retained (0)', 'Churned (1)'],
              counts.values,
              color=['#2196F3', '#FF5722'],
              edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}\n({val/n_rows*100:.1f}%)",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title(f"Churn Distribution  -  Overall churn rate: {churn_rate:.2%}",
             fontsize=12, fontweight='bold')
ax.set_ylabel("Customer Count")
ax.set_ylim(0, counts.max() * 1.15)
save("churn_distribution")
print("Chart saved: churn_distribution.png")


print("\n" + "=" * 60)
print("SECTION 4 - AGE DISTRIBUTION")
print("=" * 60)
print(df.groupby('Churn')['Age'].describe().round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='Churn', bins=20, alpha=0.6, ax=ax,
             palette={0: '#2196F3', 1: '#FF5722'})
ax.set_title("Age Distribution by Churn", fontsize=13, fontweight='bold')
ax.set_xlabel("Age")
ax.set_ylabel("Count")
save("age_churn")
print("Chart saved: age_churn.png")


print("\n" + "=" * 60)
print("SECTION 5 - CORRELATION ANALYSIS")
print("=" * 60)

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

churn_corr = (corr['Churn']
              .drop('Churn')
              .abs()
              .sort_values(ascending=False)
              .head(15))
print("Top 15 features correlated with Churn (absolute):")
print(churn_corr.round(4).to_string())

n = numeric_df.shape[1]
fig_size = (min(22, n * 0.42), min(18, n * 0.34))
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.3,
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Correlation Heatmap (All Numeric Features)",
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)
save("correlation_heatmap")
print("\nChart saved: correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(9, 5))
signed = corr['Churn'].drop('Churn').sort_values().tail(15)
colours = ['#FF5722' if v > 0 else '#2196F3' for v in signed.values]
ax.barh(signed.index, signed.values, color=colours)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Top 15 Feature Correlations with Churn\n(red = positive, blue = negative)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Pearson Correlation Coefficient")
save("churn_correlations_top15")
print("Chart saved: churn_correlations_top15.png")


print("\n" + "=" * 60)
print("SECTION 6 - CHURN RATE BY CATEGORICAL GROUP")
print("=" * 60)

categorical_cols = [
    'Gender', 'Region', 'Customer Type',
    'Tariff Type', 'Payment Plan', 'Meter Type',
]

for col in categorical_cols:
    if col not in df.columns:
        continue
    rates = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    counts = df.groupby(col)['Churn'].count()
    print(f"\nChurn rate by {col}:")
    for g in rates.index:
        print(f"  {g:20s}  rate={rates[g]:.3f}  n={counts[g]:,}")

    fig, ax = plt.subplots(figsize=(max(6, len(rates) * 1.4), 4))
    bars = ax.bar(rates.index, rates.values,
                  color='#1f77b4', edgecolor='white', linewidth=1)
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.1%}", ha='center', va='bottom', fontsize=9)
    ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel(col)
    ax.set_ylim(0, rates.max() * 1.2)
    ax.tick_params(axis='x', rotation=15)
    save(f"churn_by_{col.replace(' ', '_')}")
    print(f"  Chart saved: churn_by_{col.replace(' ', '_')}.png")


print("\n" + "=" * 60)
print("SECTION 7 - NUMERIC FEATURE BOXPLOTS vs CHURN")
print("=" * 60)

numeric_features = [
    'Age',
    'Late Payments',
    'Average Payment Delay (days)',
    'Calls Last Month',
    'Complaints Last Year',
    'App Logins',
]

for col in numeric_features:
    if col not in df.columns:
        continue
    group_stats = df.groupby('Churn')[col].describe()[['mean', '50%', 'std']]
    print(f"\n{col}:")
    print(group_stats.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='Churn', y=col, data=df, ax=ax,
                palette={0: '#2196F3', 1: '#FF5722'},
                width=0.4, flierprops=dict(marker='o', markersize=3, alpha=0.4))
    ax.set_xticklabels(['Retained (0)', 'Churned (1)'])
    ax.set_title(f"{col} by Churn", fontsize=12, fontweight='bold')
    ax.set_xlabel("")
    safe_name = col.replace(' ', '_').replace('(', '').replace(')', '')
    save(f"boxplot_{safe_name}")
    print(f"  Chart saved: boxplot_{safe_name}.png")


print("\n" + "=" * 60)
print("SECTION 8 - DIRECT DEBIT DEEP-DIVE")
print("=" * 60)

if 'Direct Debit' in df.columns:
    dd_churn = df.groupby('Direct Debit')['Churn'].mean()
    dd_count = df.groupby('Direct Debit')['Churn'].count()
    for k in dd_churn.index:
        label = "Uses DD" if k == 1 else "No DD"
        print(f"  {label}: churn rate = {dd_churn[k]:.3f}  (n={dd_count[k]:,})")

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['No Direct Debit', 'Uses Direct Debit']
    ax.bar(labels, dd_churn.values, color=['#FF5722', '#2196F3'],
           edgecolor='white', linewidth=1.2)
    for i, val in enumerate(dd_churn.values):
        ax.text(i, val + 0.003, f"{val:.1%}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title("Churn Rate - Direct Debit vs No Direct Debit",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, dd_churn.max() * 1.25)
    save("churn_by_direct_debit")
    print("Chart saved: churn_by_direct_debit.png")


print("\n" + "=" * 60)
print("SECTION 9 - MULTICOLLINEARITY RISK")
print("=" * 60)

elec_cols = [f'Electricity_Month_{i}' for i in range(1, 13)]
gas_cols = [f'Gas_Month_{i}' for i in range(1, 13)]
bill_cols = [f'Bill_Month_{i}' for i in range(1, 13)]

all_monthly = [c for c in elec_cols + gas_cols + bill_cols if c in df.columns]
if all_monthly:
    elec_present = [c for c in elec_cols if c in df.columns]
    gas_present = [c for c in gas_cols if c in df.columns]
    bill_present = [c for c in bill_cols if c in df.columns]

    for grp_name, cols in [('Electricity months', elec_present),
                            ('Gas months', gas_present),
                            ('Bill months', bill_present)]:
        if len(cols) > 1:
            sub = df[cols].corr().values
            upper = sub[np.triu_indices_from(sub, k=1)]
            print(f"  {grp_name}: avg inter-month correlation = {upper.mean():.3f}")

    print("  -> High multicollinearity confirms these 36 columns should be")
    print("     replaced with trend/recency summary features.")

    if elec_present:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[elec_present].corr(), cmap='Reds', annot=False,
                    linewidths=0.3, ax=ax,
                    xticklabels=[c.replace('Electricity_Month_', 'M') for c in elec_present],
                    yticklabels=[c.replace('Electricity_Month_', 'M') for c in elec_present])
        ax.set_title("Electricity Monthly Columns - Inter-feature Correlation",
                     fontsize=11, fontweight='bold')
        save("multicollinearity_electricity_months")
        print("Chart saved: multicollinearity_electricity_months.png")


print("\n" + "=" * 60)
print("SECTION 10 - DATA LEAKAGE RISK SUMMARY")
print("=" * 60)

risks = {
    "Direct Debit": "May reflect cancellation during churn process, not a prior signal.",
    "Monthly aggregate columns (3M Avg, Usage Change %)": "Redundant summaries of the 36 monthly columns - keeping both inflates model.",
    "Customer ID": "Unique identifier - memorises individuals, not a generalisable feature.",
    "SMOTE timing": "SMOTE must be applied AFTER train/test split to avoid synthetic test-set leakage.",
}
for k, v in risks.items():
    print(f"  [{k}]\n    -> {v}\n")


print("=" * 60)
print("EDA COMPLETE")
print(f"All charts saved to: {OUTPUT_DIR}")
print("=" * 60)