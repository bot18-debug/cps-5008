import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              roc_auc_score, confusion_matrix,
                              roc_curve, precision_recall_curve,
                              ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(SCRIPT_DIR, "customer_account_and_usage.csv"),
    os.path.join(SCRIPT_DIR, "..", "customer_account_and_usage.csv"),
]:
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break
else:
    raise FileNotFoundError("customer_account_and_usage.csv not found.")

RANDOM_STATE = 42


def metrics(y_true, y_pred, y_proba):
    return {
        'Recall':    recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_true, y_proba),
    }


def print_metrics(name, m):
    print(f"  Recall:    {m['Recall']:.4f}")
    print(f"  Precision: {m['Precision']:.4f}")
    print(f"  F1:        {m['F1']:.4f}")
    print(f"  ROC-AUC:   {m['ROC-AUC']:.4f}")


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"{name}.png"), dpi=150)
    plt.close()
    print(f"  Chart saved: {name}.png")


print("=" * 60)
print("SECTION 1 - LOAD DATA")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}   Churn rate: {df['Churn'].mean():.4f}")


print("\n" + "=" * 60)
print("SECTION 2 - FEATURE ENGINEERING")
print("=" * 60)

if 'Customer ID' in df.columns:
    df.drop('Customer ID', axis=1, inplace=True)
    print("  Dropped: Customer ID")

digital_cols = ['App Logins', 'Portal Logins', 'Email Clicks']
for col in digital_cols:
    if col in df.columns:
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna(df[col].median())
        print(f"  Imputed {n_missing} missing values in '{col}' with median")

elec_cols = [f'Electricity_Month_{i}' for i in range(1, 13)]
gas_cols  = [f'Gas_Month_{i}'         for i in range(1, 13)]
bill_cols = [f'Bill_Month_{i}'        for i in range(1, 13)]

for raw_cols, prefix in [(elec_cols, 'elec'), (gas_cols, 'gas'), (bill_cols, 'bill')]:
    present = [c for c in raw_cols if c in df.columns]
    if present:
        df[f'{prefix}_trend'] = df[present].diff(axis=1).mean(axis=1)
        df[f'{prefix}_month12'] = df[present[-1]]
        df.drop(columns=present, inplace=True)
        print(f"  Engineered: {prefix}_trend, {prefix}_month12  (dropped {len(present)} raw columns)")

redundant = [
    'Electricity 3M Avg', 'Gas 3M Avg', 'Total 3M Avg',
    'Usage Change %', 'Average Monthly Usage', 'High Usage Flag'
]
dropped_redundant = [c for c in redundant if c in df.columns]
df.drop(columns=dropped_redundant, inplace=True)
if dropped_redundant:
    print(f"  Dropped redundant aggregate columns: {dropped_redundant}")

print(f"\n  Final feature count (before encoding): {df.shape[1] - 1} predictors + 1 target")
print(f"  Columns: {list(df.columns)}")


print("\n" + "=" * 60)
print("SECTION 3 - TRAIN / TEST SPLIT")
print("=" * 60)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
print(f"  Train churn rate: {y_train.mean():.4f}  |  Test churn rate: {y_test.mean():.4f}")


print("\n" + "=" * 60)
print("SECTION 4 - PREPROCESSING PIPELINE")
print("=" * 60)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Numeric features  ({len(numeric_cols)}): {numeric_cols}")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
])

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


print("\n" + "=" * 60)
print("MODEL 1 - LOGISTIC REGRESSION (Baseline)")
print("=" * 60)

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
    )),
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr  = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
m_lr       = metrics(y_test, y_pred_lr, y_proba_lr)
print_metrics("Logistic Regression", m_lr)

cm_lr = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = cm_lr.ravel()
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn:,}")
print(f"    False Positives: {fp:,}")
print(f"    False Negatives: {fn:,}")
print(f"    True Positives:  {tp:,}")


print("\n" + "=" * 60)
print("MODEL 2 - RANDOM FOREST + SMOTE (GridSearchCV)")
print("=" * 60)

rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])

param_grid_rf = {
    'clf__n_estimators':      [100, 200],
    'clf__max_depth':         [10, 15],
    'clf__min_samples_split': [5, 10],
    'clf__class_weight':      ['balanced'],
}

grid_rf = GridSearchCV(
    rf_pipeline, param_grid_rf,
    cv=cv_strategy, scoring='recall',
    n_jobs=-1, verbose=1, refit=True,
)
grid_rf.fit(X_train, y_train)
best_rf    = grid_rf.best_estimator_
y_pred_rf  = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
m_rf       = metrics(y_test, y_pred_rf, y_proba_rf)

print(f"\n  Best hyperparameters: {grid_rf.best_params_}")
print(f"  Best CV recall: {grid_rf.best_score_:.4f}")
print_metrics("Random Forest", m_rf)

cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm_rf.ravel()
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn:,}")
print(f"    False Positives: {fp:,}")
print(f"    False Negatives: {fn:,}")
print(f"    True Positives:  {tp:,}")


print("\n" + "=" * 60)
print("MODEL 3 - XGBOOST + SMOTE (GridSearchCV)")
print("=" * 60)

xgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ('clf', XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
    )),
])

param_grid_xgb = {
    'clf__n_estimators':     [100, 200],
    'clf__max_depth':        [3, 5],
    'clf__learning_rate':    [0.05, 0.1],
    'clf__scale_pos_weight': [5, 10],
}

grid_xgb = GridSearchCV(
    xgb_pipeline, param_grid_xgb,
    cv=cv_strategy, scoring='recall',
    n_jobs=-1, verbose=1, refit=True,
)
grid_xgb.fit(X_train, y_train)
best_xgb    = grid_xgb.best_estimator_
y_pred_xgb  = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
m_xgb       = metrics(y_test, y_pred_xgb, y_proba_xgb)

print(f"\n  Best hyperparameters: {grid_xgb.best_params_}")
print(f"  Best CV recall: {grid_xgb.best_score_:.4f}")
print_metrics("XGBoost", m_xgb)

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = cm_xgb.ravel()
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn:,}")
print(f"    False Positives: {fp:,}")
print(f"    False Negatives: {fn:,}")
print(f"    True Positives:  {tp:,}")


print("\n" + "=" * 60)
print("SECTION 8 - MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    'Model':     ['Logistic Regression (balanced)',
                  'Random Forest + SMOTE',
                  'XGBoost + SMOTE'],
    'Recall':    [m_lr['Recall'],    m_rf['Recall'],    m_xgb['Recall']],
    'Precision': [m_lr['Precision'], m_rf['Precision'], m_xgb['Precision']],
    'F1':        [m_lr['F1'],        m_rf['F1'],        m_xgb['F1']],
    'ROC-AUC':   [m_lr['ROC-AUC'],  m_rf['ROC-AUC'],  m_xgb['ROC-AUC']],
})
print(comparison.round(4).to_string(index=False))
comparison.to_csv(os.path.join(SCRIPT_DIR, 'model_comparison.csv'), index=False)
print("\n  Saved: model_comparison.csv")


print("\n" + "=" * 60)
print("SECTION 9 - CONFUSION MATRIX CHARTS")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
model_labels = [
    ('Logistic Regression\n(Baseline)', cm_lr),
    ('Random Forest\n+ SMOTE',          cm_rf),
    ('XGBoost\n+ SMOTE',                cm_xgb),
]
for ax, (title, cm) in zip(axes, model_labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Retained', 'Churned'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=11, fontweight='bold')

plt.suptitle("Confusion Matrices - All Three Models (Test Set)",
             fontsize=13, fontweight='bold', y=1.02)
save("confusion_matrices")


print("\n" + "=" * 60)
print("SECTION 10 - ROC CURVES")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 5))
for name, y_proba, m in [
    ("Logistic Regression", y_proba_lr,  m_lr),
    ("Random Forest",        y_proba_rf,  m_rf),
    ("XGBoost",              y_proba_xgb, m_xgb),
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {m['ROC-AUC']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline (AUC = 0.5)')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax.set_title("ROC Curves - Model Comparison", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
save("roc_curves")


print("\n" + "=" * 60)
print("SECTION 11 - THRESHOLD ANALYSIS (XGBoost)")
print("=" * 60)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_xgb)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, precisions[:-1], label='Precision', color='#2196F3', lw=2)
ax.plot(thresholds, recalls[:-1],    label='Recall',    color='#FF5722',  lw=2)
ax.axvline(0.3, color='green',  linestyle='--', lw=1.5, label='Threshold = 0.30 (recommended)')
ax.axvline(0.5, color='purple', linestyle='--', lw=1.5, label='Threshold = 0.50 (default)')
ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("XGBoost - Precision & Recall vs Decision Threshold",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
save("threshold_analysis")

recommended_threshold = 0.30
y_pred_xgb_t30 = (y_proba_xgb >= recommended_threshold).astype(int)
m_xgb_t30 = metrics(y_test, y_pred_xgb_t30, y_proba_xgb)
print(f"\n  XGBoost at threshold = {recommended_threshold}:")
print(f"    Recall:    {m_xgb_t30['Recall']:.4f}")
print(f"    Precision: {m_xgb_t30['Precision']:.4f}")
print(f"    F1:        {m_xgb_t30['F1']:.4f}")
cm_t30 = confusion_matrix(y_test, y_pred_xgb_t30)
print(f"    Confusion matrix:\n{cm_t30}")


print("\n" + "=" * 60)
print("SECTION 12 - FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

preprocessor_fitted = best_rf.named_steps['preprocessor']
cat_features_enc    = (preprocessor_fitted
                       .named_transformers_['cat']
                       .named_steps['onehot']
                       .get_feature_names_out(categorical_cols))
all_features = np.concatenate([numeric_cols, cat_features_enc])
importances  = best_rf.named_steps['clf'].feature_importances_

top_n    = 15
indices  = np.argsort(importances)[-top_n:]
top_names = [all_features[i] for i in indices]
top_vals  = importances[indices]

print("  Top 15 features by Random Forest importance:")
for name, val in zip(reversed(top_names), reversed(top_vals)):
    print(f"    {name:45s}  {val:.4f}")

fig, ax = plt.subplots(figsize=(10, 7))
colours = ['#FF5722' if 'direct_debit' in n.lower() or
                        'late' in n.lower() or
                        'payment' in n.lower() or
                        'complaint' in n.lower()
           else '#1f77b4' for n in top_names]
ax.barh(range(top_n), top_vals, color=colours, edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names, fontsize=9)
ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=10)
ax.set_title("Top 15 Feature Importances - Random Forest\n(orange = payment/behaviour features)",
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
save("feature_importance")


print("\n" + "=" * 60)
print("SECTION 13 - FAIRNESS / SEGMENT ANALYSIS")
print("=" * 60)

fairness_cols = ['Region', 'Gender', 'Customer Type', 'Tariff Type']

model_preds = [
    ("Logistic Regression", y_pred_lr),
    ("Random Forest",        y_pred_rf),
    ("XGBoost",              y_pred_xgb),
]

fairness_records = []

for col in fairness_cols:
    if col not in X_test.columns:
        continue
    print(f"\n  --- {col} ---")
    for model_name, y_pred in model_preds:
        print(f"  [{model_name}]")
        for val in sorted(X_test[col].unique()):
            mask = X_test[col] == val
            if mask.sum() == 0 or y_test[mask].sum() == 0:
                print(f"    {val:25s}  recall=N/A  (no churners in segment)")
                continue
            rec  = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            prec = precision_score(y_test[mask], y_pred[mask], zero_division=0)
            n_total = mask.sum()
            n_churn = int(y_test[mask].sum())
            print(f"    {val:25s}  recall={rec:.3f}  precision={prec:.3f}"
                  f"  n={n_total:,}  churners={n_churn}")
            fairness_records.append({
                'Segment_Column': col,
                'Segment_Value':  val,
                'Model':          model_name,
                'Recall':         round(rec, 4),
                'Precision':      round(prec, 4),
                'N_Total':        int(n_total),
                'N_Churners':     n_churn,
            })

fairness_df = pd.DataFrame(fairness_records)
fairness_df.to_csv(os.path.join(SCRIPT_DIR, 'fairness_analysis.csv'), index=False)
print("\n  Saved: fairness_analysis.csv")

region_fair = fairness_df[fairness_df['Segment_Column'] == 'Region']
if not region_fair.empty:
    pivot = region_fair.pivot(index='Segment_Value', columns='Model', values='Recall')
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind='bar', ax=ax, edgecolor='white',
               color=['#2196F3', '#FF9800', '#4CAF50'])
    ax.set_title("Recall by Region - All Three Models",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel("Recall")
    ax.set_xlabel("Region")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Model', fontsize=9)
    ax.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    save("fairness_region_recall")


print("\n" + "=" * 60)
print("SECTION 14 - ERROR ANALYSIS")
print("=" * 60)

for model_name, y_pred in model_preds:
    fn_mask = (y_test == 1) & (y_pred == 0)
    fp_mask = (y_test == 0) & (y_pred == 1)
    print(f"\n  [{model_name}]")
    print(f"    False Negatives (churners missed): {fn_mask.sum():,}")
    print(f"    False Positives (loyal flagged):   {fp_mask.sum():,}")
    if fn_mask.sum() > 0 and 'Region' in X_test.columns:
        fn_regions = X_test.loc[fn_mask, 'Region'].value_counts()
        print(f"    Missed churners by region:")
        for region, count in fn_regions.items():
            total_churn_in_region = int((y_test[X_test['Region'] == region]).sum())
            pct = count / total_churn_in_region * 100 if total_churn_in_region > 0 else 0
            print(f"      {region}: {count} missed  ({pct:.1f}% of churners in that region)")


print("\n" + "=" * 60)
print("SECTION 15 - QUANTIFIED BUSINESS IMPACT")
print("=" * 60)

total_annual_churners = int(df['Churn'].sum())
clv         = 500
retention_p = 0.30
cost_per_fp = 8

print(f"\n  Assumptions: CLV=£{clv}, retention rate={retention_p:.0%}, FP cost=£{cost_per_fp}")
print(f"  Total annual churners in dataset: {total_annual_churners:,}\n")

test_scale = len(y_test) / len(y)
for model_name, y_pred, y_proba in [
    ("Logistic Regression", y_pred_lr,  y_proba_lr),
    ("Random Forest",        y_pred_rf,  y_proba_rf),
    ("XGBoost",              y_pred_xgb, y_proba_xgb),
]:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    annual_tp = tp / test_scale
    annual_fp = fp / test_scale
    saved = annual_tp * retention_p * clv
    cost  = annual_fp * cost_per_fp
    net   = saved - cost
    print(f"  {model_name}:")
    print(f"    Churners identified / year (scaled): {annual_tp:,.0f}")
    print(f"    Estimated revenue saved:             £{saved:,.0f}")
    print(f"    Outreach cost (false positives):     £{cost:,.0f}")
    print(f"    Net annual benefit:                  £{net:,.0f}\n")


print("=" * 60)
print("TRAINING PIPELINE COMPLETE")
print("Output files:")
for fname in ['model_comparison.csv', 'fairness_analysis.csv',
              'confusion_matrices.png', 'roc_curves.png',
              'feature_importance.png', 'threshold_analysis.png',
              'fairness_region_recall.png']:
    full = os.path.join(SCRIPT_DIR, fname)
    status = "OK" if os.path.exists(full) else "NOT FOUND"
    print(f"  [{status}]  {fname}")
print("=" * 60)