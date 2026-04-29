import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(output_dir, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "customer_account_and_usage.csv")
df = pd.read_csv(file_path)

print("="*60)
print("ENHANCED EXPLORATORY DATA ANALYSIS")
print("="*60)

print(f"\nShape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes.value_counts()}")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\nMissing values:\n{missing}")
    plt.figure(figsize=(10,4))
    missing.plot(kind='bar')
    plt.title("Missing values per column")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values.png"))
    plt.show()
else:
    print("\nNo missing values found.")

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title(f"Churn Distribution (Churn rate: {df['Churn'].mean():.2%})")
plt.savefig(os.path.join(output_dir, "churn_distribution.png"))
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Age', hue='Churn', bins=20, alpha=0.6)
plt.title("Age Distribution by Churn")
plt.savefig(os.path.join(output_dir, "age_churn.png"))
plt.show()

numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(14,12))
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

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
        plt.show()

num_features = ['Age', 'Late Payments', 'Average Payment Delay (days)', 
                'Calls Last Month', 'Complaints Last Year', 'App Logins']
for col in num_features:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x='Churn', y=col, data=df)
        plt.title(f"{col} by Churn")
        plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.show()

