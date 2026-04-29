
Copy

# imports.py
# Central imports file for CPS5008 - Customer Churn Prediction
# Student ID: 2411066
 
# ── Standard Library ──────────────────────────────────────────
import os
import warnings
warnings.filterwarnings('ignore')
 
# ── Data Handling ─────────────────────────────────────────────
import pandas as pd
import numpy as np
 
# ── Visualisation ─────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (required for script execution)
import matplotlib.pyplot as plt
import seaborn as sns
 
# ── Preprocessing ─────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
# ── Model Selection & Validation ──────────────────────────────
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
 
# ── Models ────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
 
# ── XGBoost ───────────────────────────────────────────────────
from xgboost import XGBClassifier
 
# ── Imbalanced Learning ───────────────────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
 
# ── Evaluation Metrics ────────────────────────────────────────
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)