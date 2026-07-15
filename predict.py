import joblib
import pandas as pd

print("Loading Trained Model...\n")

model = joblib.load("model.pkl")

print("Model Loaded Successfully!")

df = pd.read_csv("creditcard.csv")

while True:

    print("\n-----------------------------------")

    transaction = int(input("Enter Transaction Number : "))

    if transaction < 0 or transaction >= len(df):

        print("Invalid Transaction Number!")
        continue

    X = df.drop("Class", axis=1)

    sample = X.iloc[[transaction]]

    actual = df.iloc[transaction]["Class"]

    prediction = model.predict(sample)

    confidence = model.predict_proba(sample)

    confidence = confidence.max() * 100

    print("\nPrediction Result")

    if prediction[0] == 0:
        print("Prediction : Genuine Transaction")
    else:
        print("Prediction : Fraudulent Transaction")

    if actual == 0:
        print("Actual     : Genuine Transaction")
    else:
        print("Actual     : Fraudulent Transaction")

    print(f"Confidence : {confidence:.2f}%")

    again = input("\nCheck another transaction? (y/n) : ")

    if again.lower() != "y":
        break

print("\nThank You!")