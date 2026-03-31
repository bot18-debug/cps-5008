import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\singh\cps-5008\ML_Project\customer_account_and_usage.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()


sns.histplot(df['Age'], bins=20)
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()