import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(output_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "customer_account_and_usage.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(script_dir, "..", "customer_account_and_usage.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("customer_account_and_usage.csv not found.")
df = pd.read_csv(csv_path)

print("="*60)
print("ENHANCED EXPLORATORY DATA ANALYSIS")
print("="*60)

print(f"Shape: {df.shape}")
churn_rate = df['Churn'].mean()
print(f"Churn rate: {churn_rate:.2%}")

majority = (df['Churn'] == 0).sum()
minority = (df['Churn'] == 1).sum()
imbalance_ratio = majority / minority
print(f"Imbalance ratio (non‑churn : churn): {imbalance_ratio:.1f}:1")
print("Accuracy would be misleading – we will optimise recall (identify churners).")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\nMissing values:\n{missing}")
    plt.figure(figsize=(10,4))
    missing.plot(kind='bar')
    plt.title("Missing values per column")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values.png"))
    plt.close()
else:
    print("\nNo missing values found.")

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title(f"Churn Distribution (Churn rate: {churn_rate:.2%})")
plt.savefig(os.path.join(output_dir, "churn_distribution.png"))
plt.close()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Age', hue='Churn', bins=20, alpha=0.6)
plt.title("Age Distribution by Churn")
plt.savefig(os.path.join(output_dir, "age_churn.png"))
plt.close()

numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
num_cols = numeric_df.shape[1]
figsize = (min(20, num_cols * 0.5), min(16, num_cols * 0.4))
plt.figure(figsize=figsize)
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap (Numeric Features)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

churn_corr = corr['Churn'].drop('Churn').abs().sort_values(ascending=False).head(10)
print(f"\nTop 10 features correlated with Churn:\n{churn_corr}")

categorical_cols = ['Gender', 'Region', 'Customer Type', 'Tariff Type', 'Payment Plan', 'Meter Type']
for col in categorical_cols:
    if col in df.columns:
        print(f"\nChurn rate by {col}:")
        print(df.groupby(col)['Churn'].mean().sort_values(ascending=False))
        plt.figure(figsize=(8,4))
        df.groupby(col)['Churn'].mean().plot(kind='bar')
        plt.title(f"Churn Rate by {col}")
        plt.ylabel("Churn Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"churn_by_{col}.png"))
        plt.close()

num_features = ['Age', 'Late Payments', 'Average Payment Delay (days)', 
                'Calls Last Month', 'Complaints Last Year', 'App Logins']
for col in num_features:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x='Churn', y=col, data=df)
        plt.title(f"{col} by Churn")
        plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.close()

print("\nAll EDA plots saved to 'graphs' folder.")