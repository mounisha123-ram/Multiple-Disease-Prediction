# ckd_dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

st.set_page_config(page_title="CKD Dashboard", layout="wide")
st.title("🩺 Chronic Kidney Disease (CKD) Prediction")

# Sidebar navigation
section = st.sidebar.radio("Select Section", ["Prediction", "EDA", "Model Evaluation Results"])

# Load model and scaler
with open("kidney_rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("kidney_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_kidney_data.csv")

df = load_data()

# ---------------------- Prediction Section ---------------------- #
if section == "Prediction":
    st.header("🔍 CKD Prediction")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=100)
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")
    albumin = st.number_input("Albumin")
    sugar = st.number_input("Sugar")

    rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
    pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
    ba = st.selectbox("Bacteria", ["Not Present", "Present"])

    bgr = st.number_input("Blood Glucose Random")
    bu = st.number_input("Blood Urea")
    sc = st.number_input("Serum Creatinine")
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    hemo = st.number_input("Hemoglobin")
    pcv = st.number_input("Packed Cell Volume")
    wc = st.number_input("White Blood Cell Count")
    rc = st.number_input("Red Blood Cell Count")

    htn = st.selectbox("Hypertension", ["No", "Yes"])
    dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
    cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
    appet = st.selectbox("Appetite", ["Poor", "Good"])
    pe = st.selectbox("Pedal Edema", ["No", "Yes"])
    ane = st.selectbox("Anemia", ["No", "Yes"])

    # Mapping categorical values
    mapping = {
        'Normal': 0, 'Abnormal': 1,
        'Not Present': 0, 'Present': 1,
        'No': 0, 'Yes': 1,
        'Poor': 0, 'Good': 1
    }

    input_data = np.array([[age, bp, sg, albumin, sugar,
                            mapping[rbc], mapping[pc], mapping[pcc], mapping[ba],
                            bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                            mapping[htn], mapping[dm], mapping[cad],
                            mapping[appet], mapping[pe], mapping[ane]]])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to have CKD\nPrediction Confidence: {prob:.2f}")
        else:
            st.success(f"✅ Not likely to have CKD\nPrediction Confidence: {1 - prob:.2f}")

# ---------------------- EDA Section ---------------------- #
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    eda_option = st.selectbox("Select EDA Plot", [
        "Box Plots", "Histograms", "CKD Status Count",
        "EDA Features vs Status", "Correlation Matrix"
    ])

    if eda_option == "Box Plots":
        numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if "Status" in numerical_cols:
            numerical_cols.remove("Status")
        for col in numerical_cols:
            fig = px.box(df, y=col, title=f'Box Plot of {col}', points="all", template="plotly_white")
            st.plotly_chart(fig)

    elif eda_option == "Histograms":
        numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if "Status" in numerical_cols:
            numerical_cols.remove("Status")
        for col in numerical_cols:
            fig = px.histogram(df, x=col, title=f'Histogram of {col}', marginal="rug", nbins=30)
            st.plotly_chart(fig)

    elif eda_option == "CKD Status Count":
        st.bar_chart(df["Status"].value_counts())

    elif eda_option == "EDA Features vs Status":
        cat_features = ['Red_Blood_Cells', 'Pus_Cell', 'Hypertension', 'Diabetes_Mellitus', 'Appetite']
        for col in cat_features:
            fig, ax = plt.subplots()
            sns.countplot(x=col, hue="Status", data=df, palette="Set2", ax=ax)
            st.pyplot(fig)

    elif eda_option == "Correlation Matrix":
        corr = df.select_dtypes(include=["float64", "int64"]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ---------------------- Model Results Section ---------------------- #
elif section == "Model Evaluation Results":
    st.header("Model Used --> RandomForestClassifier")

    X_test = pd.read_csv("X_test_kidney_scaled.csv").values
    y_test = pd.read_csv("y_kidney_test.csv")['Status'].values

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("🔢 Accuracy", f"{acc*100:.2f}%")

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)
