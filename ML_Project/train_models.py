import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

print("="*60)
print("CUSTOMER CHURN PREDICTION")
print("="*60)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "customer_account_and_usage.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(script_dir, "..", "customer_account_and_usage.csv")
df = pd.read_csv(csv_path)

print(f"Shape: {df.shape}, Churn rate: {df['Churn'].mean():.4f}")

if 'Customer ID' in df.columns:
    df.drop('Customer ID', axis=1, inplace=True)

digital_cols = ['App Logins', 'Portal Logins', 'Email Clicks']
for col in digital_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

elec_cols = [f'Electricity_Month_{i}' for i in range(1,13)]
gas_cols = [f'Gas_Month_{i}' for i in range(1,13)]
bill_cols = [f'Bill_Month_{i}' for i in range(1,13)]

for cols, prefix in zip([elec_cols, gas_cols, bill_cols], ['elec', 'gas', 'bill']):
    df[f'{prefix}_trend'] = df[cols].diff(axis=1).mean(axis=1)
    df[f'{prefix}_month12'] = df[cols[-1]]

df.drop(columns=elec_cols + gas_cols + bill_cols, inplace=True)

redundant = ['Electricity 3M Avg', 'Gas 3M Avg', 'Total 3M Avg', 'Usage Change %', 
             'Average Monthly Usage', 'High Usage Flag']
df.drop(columns=[c for c in redundant if c in df.columns], inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

print("\n" + "="*60)
print("LOGISTIC REGRESSION (balanced class weight)")
print("="*60)
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"F1: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")

print("\n" + "="*60)
print("RANDOM FOREST + SMOTE (with GridSearchCV)")
print("="*60)
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 15],
    'clf__min_samples_split': [5, 10],
    'clf__class_weight': ['balanced']
}
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
print(f"Best params: {grid_rf.best_params_}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"F1: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

print("\n" + "="*60)
print("XGBOOST + SMOTE (with GridSearchCV)")
print("="*60)
xgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
])
param_grid_xgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.05, 0.1],
    'clf__scale_pos_weight': [5, 10]
}
grid_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
print(f"Best params: {grid_xgb.best_params_}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"F1: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
comparison = pd.DataFrame({
    'Model': ['Logistic Regression (balanced)', 'Random Forest + SMOTE', 'XGBoost + SMOTE'],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_xgb)],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_xgb)],
    'F1': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_xgb)],
    'ROC-AUC': [roc_auc_score(y_test, y_proba_lr), roc_auc_score(y_test, y_proba_rf), roc_auc_score(y_test, y_proba_xgb)]
})
print(comparison.to_string(index=False))

preprocessor_fitted = best_rf.named_steps['preprocessor']
cat_features = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, cat_features])
importances = best_rf.named_steps['clf'].feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [all_features[i] for i in indices])
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\n" + "="*60)
print("FAIRNESS ANALYSIS (Random Forest)")
print("="*60)
for col in ['Region', 'Gender', 'Customer Type']:
    if col in X_test.columns:
        print(f"\n{col}:")
        for val in X_test[col].unique():
            mask = X_test[col] == val
            if mask.sum() > 0 and y_test[mask].sum() > 0:
                rec = recall_score(y_test[mask], y_pred_rf[mask])
                print(f"  {val}: recall={rec:.3f} (n={mask.sum()}, churn={y_test[mask].sum()})")
            else:
                print(f"  {val}: no churn cases")

print("\nScript complete.")