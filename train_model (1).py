"""
Air Quality Category Prediction — Training Script
====================================================
Dataset: UCI Air Quality Data Set
https://archive.ics.uci.edu/dataset/360/air+quality

Download 'AirQualityUCI.csv' (or .xlsx) from the link above and place it
in this same folder before running this script.

This script:
1. Loads and cleans the raw sensor data
2. Engineers an AQI CATEGORY label (Good / Moderate / Poor / Severe) from
   CO and NO2 concentrations, since the raw dataset has no AQI column
3. Trains a Random Forest classifier
4. Saves the trained model + label encoder for later use in the app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
# Works with either the .csv or the .xlsx version of the dataset —
# just point FILENAME at whichever one you have.
FILENAME = "AirQualityUCI.xlsx"  # change to "AirQualityUCI.csv" if you have that instead

if FILENAME.endswith(".xlsx"):
    df = pd.read_excel(FILENAME)
else:
    # The official CSV is ';' separated and uses ',' as decimal separator.
    df = pd.read_csv(FILENAME, sep=";", decimal=",")

# Drop the two empty trailing columns the file sometimes ships with
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Raw shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------
# Missing sensor readings are encoded as -200 in this dataset
df = df.replace(-200, np.nan)

# Drop rows where the two pollutants we need for labeling are missing
df = df.dropna(subset=["CO(GT)", "NO2(GT)"])

# Fill remaining missing feature values with column median
feature_cols = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]
feature_cols = [c for c in feature_cols if c in df.columns]

for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------
# 3. ENGINEER AQI CATEGORY LABEL
# ---------------------------------------------------------
# NOTE: The UCI dataset does not include PM2.5/PM10, so an official
# EPA/CPCB AQI cannot be computed. This uses simplified breakpoints on
# CO (mg/m3) and NO2 (ug/m3) to create a 4-class proxy label.
# Swap in official breakpoints + PM2.5 data if you have it, for a more
# accurate real-world AQI classifier.

def co_category(co):
    if co < 1:
        return 0  # Good
    elif co < 2:
        return 1  # Moderate
    elif co < 10:
        return 2  # Poor
    else:
        return 3  # Severe

def no2_category(no2):
    if no2 < 40:
        return 0
    elif no2 < 80:
        return 1
    elif no2 < 180:
        return 2
    else:
        return 3

labels = ["Good", "Moderate", "Poor", "Severe"]

df["aqi_category"] = df.apply(
    lambda row: labels[max(co_category(row["CO(GT)"]), no2_category(row["NO2(GT)"]))],
    axis=1
)

print("\nClass distribution:")
print(df["aqi_category"].value_counts())

# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X = df[feature_cols]
y = df["aqi_category"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
print("\nFeature importances:")
print(importances.sort_values(ascending=False))

# ---------------------------------------------------------
# 7. SAVE MODEL + ENCODER + FEATURE LIST
# ---------------------------------------------------------
joblib.dump(model, "aqi_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("\nSaved: aqi_model.pkl, label_encoder.pkl, feature_cols.pkl")
