import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"C:\Users\singh\cps-5008\ML_Project\customer_account_and_usage.csv")


df = df.drop("Customer ID", axis=1)


df = df.fillna(df.mean(numeric_only=True))


categorical_cols = [
    'Gender',
    'Region',
    'Customer Type',
    'Tenure Type',
    'Meter Type',
    'Tariff Type',
    'Payment Plan'
]

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


df.to_csv(r"C:\Users\singh\cps-5008\ML_Project\cleaned_data.csv", index=False)

print("Preprocessing complete. Cleaned data saved.")