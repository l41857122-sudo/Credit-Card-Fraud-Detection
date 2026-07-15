import pandas as pd

# Load Dataset
df = pd.read_csv("creditcard.csv")

# First 5 Rows
print("First 5 Rows:\n")
print(df.head())

# Dataset Information
print("\n==============================")
print("DATASET INFORMATION")
print("==============================")
print(df.info())

# Shape
print("\nDataset Shape:")
print(df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Class Distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Fraud Percentage
fraud = df["Class"].value_counts()[1]
genuine = df["Class"].value_counts()[0]

print(f"\nFraud Transactions : {fraud}")
print(f"Genuine Transactions : {genuine}")

print(f"\nFraud Percentage : {(fraud/(fraud+genuine))*100:.4f}%")
