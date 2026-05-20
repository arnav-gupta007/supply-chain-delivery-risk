import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
# Update this path to where your CSV is stored!
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

# =========================
# SELECT FEATURES
# =========================
FEATURES = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate"
]

TARGET = "Late_delivery_risk"

df = df[FEATURES + [TARGET]]

# =========================
# SPLIT
# =========================
X = df[FEATURES]
y = df[TARGET]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# =========================
# PREPROCESSOR
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# =========================
# MODEL PIPELINE
# =========================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=150, random_state=42))
])

# =========================
# TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

print("Train acc:", pipeline.score(X_train, y_train))
print("Test acc:", pipeline.score(X_test, y_test))

# =========================
# SAVE MODEL
# =========================
# Saved to the current directory so app.py can easily find it
joblib.dump(pipeline, "model.pkl")

print("✅ Model saved as model.pkl")