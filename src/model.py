# =========================
# LOAN PREDICTION MODEL (CLEAN + SAFE)
# =========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/train.csv")

# =========================
# 2. DROP ID COLUMN FIRST
# =========================

df = df.drop("Loan_ID", axis=1)

# =========================
# 3. HANDLE MISSING VALUES (SAFE METHOD)
# =========================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# =========================
# 4. ENCODE CATEGORICAL COLUMNS
# =========================

categorical_cols = df.select_dtypes(include="object").columns

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# =========================
# 5. FINAL CHECK (IMPORTANT)
# =========================

print("Missing values check:")
print(df.isnull().sum())

# =========================
# 6. SPLIT DATA
# =========================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. SCALE FEATURES
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. TRAIN MODEL
# =========================

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# =========================
# 9. EVALUATE
# =========================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# 10. SAVE MODEL
# =========================

joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model + scaler saved successfully 🚀")