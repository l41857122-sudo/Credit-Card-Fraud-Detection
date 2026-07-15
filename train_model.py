import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -------------------------------
# Load Dataset
# -------------------------------

print("Loading Dataset...\n")

df = pd.read_csv("creditcard.csv")

print("First 5 Rows\n")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nClass Distribution\n")
print(df["Class"].value_counts())

# -------------------------------
# Features and Target
# -------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# Split Dataset
# -------------------------------

print("\nSplitting Dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Logistic Regression
# ==========================================================

print("\nTraining Logistic Regression...\n")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("Accuracy :", lr_accuracy)
print("Precision :", lr_precision)
print("Recall :", lr_recall)
print("F1 Score :", lr_f1)

print("\nClassification Report\n")
print(classification_report(y_test, lr_pred))

# ==========================================================
# Decision Tree
# ==========================================================

print("\nTraining Decision Tree...\n")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("Accuracy :", dt_accuracy)
print("Precision :", dt_precision)
print("Recall :", dt_recall)
print("F1 Score :", dt_f1)

print("\nClassification Report\n")
print(classification_report(y_test, dt_pred))

# ==========================================================
# Random Forest
# ==========================================================

print("\nTraining Random Forest...\n")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Accuracy :", rf_accuracy)
print("Precision :", rf_precision)
print("Recall :", rf_recall)
print("F1 Score :", rf_f1)

print("\nClassification Report\n")
print(classification_report(y_test, rf_pred))

# -------------------------------
# Model Comparison
# -------------------------------

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ],

    "Precision": [
        lr_precision,
        dt_precision,
        rf_precision
    ],

    "Recall": [
        lr_recall,
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        lr_f1,
        dt_f1,
        rf_f1
    ]

})

print("\nModel Comparison\n")
print(comparison)

# -------------------------------
# Save Best Model
# -------------------------------

print("\nRandom Forest gave the best performance.")
print("Saving the trained model...\n")

joblib.dump(rf, "model.pkl")

print("Model saved successfully as model.pkl")