
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Universal Bank Loan Dashboard", layout="wide")

st.title("📊 Personal Loan Prediction Dashboard")

df = pd.read_csv("UniversalBank.csv")

st.subheader("Dataset Overview")
st.write(df.head())

# Drop ID
df_model = df.drop(columns=["ID"])

X = df_model.drop("Personal Loan", axis=1)
y = df_model["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("📈 Model Performance")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# ROC Curve
st.subheader("ROC Curve")

plt.figure()

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
st.pyplot(plt)

# Confusion Matrix (Random Forest example)
st.subheader("Confusion Matrix (Random Forest)")
rf = models["Random Forest"]
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Upload test file
st.subheader("📂 Upload Test Data for Prediction")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    test = pd.read_csv(uploaded)
    preds = rf.predict(test)
    test["Predicted Personal Loan"] = preds
    st.write(test.head())

    csv = test.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
