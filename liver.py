# liver_dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,accuracy_score

st.set_page_config(page_title="Liver Disease Dashboard", layout="wide")
st.title("🧬 Liver Disease Analysis & Prediction")

# Sidebar navigation
section = st.sidebar.radio("Select Section", ["Prediction", "EDA", "Model Evaluation Results"])

# Load model and scaler
with open("liver_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("liver_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_liver_data.csv")
    return df

df = load_data()


# ---------------------- Prediction Section ---------------------- #
if section == "Prediction":
    st.header("🩺 Liver Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0)
    alk_phos = st.number_input("Alkaline Phosphotase", min_value=0)
    alt = st.number_input("Alamine Aminotransferase (ALT)", min_value=0)
    ast = st.number_input("Aspartate Aminotransferase (AST)", min_value=0)
    total_proteins = st.number_input("Total Proteins", min_value=0.0)
    albumin = st.number_input("Albumin", min_value=0.0)
    a_g_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0)

    gender_num = 1 if gender == "Male" else 0

    if st.button("Predict"):
        input_data = np.array([[age, gender_num, total_bilirubin, direct_bilirubin, alk_phos,
                                alt, ast, total_proteins, albumin, a_g_ratio]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to have Liver Disease.\nPrediction Confidence: {pred_proba:.2f}")
        else:
            st.success(f"✅ Not likely to have Liver Disease.\nPrediction Confidence: {1 - pred_proba:.2f}")

# ---------------------- EDA Section ---------------------- #
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    eda_option = st.selectbox("Select EDA Plot", [
        "Box Plots", "Histograms", "Liver Disease Status Count",
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

    elif eda_option == "Liver Disease Status Count":
        st.bar_chart(df["Status"].value_counts())

    elif eda_option == "EDA Features vs Status":
        # Gender vs Status
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Gender", hue="Status", data=df, palette="Set2", ax=ax1)
        ax1.set_title("Gender vs Liver Disease Status")
        st.pyplot(fig1)

        # Age Group vs Status
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Age_Group', hue='Status', data=df, palette='Set1', ax=ax2)
        ax2.set_title("Age Group vs Liver Disease Status")
        st.pyplot(fig2)
        df.drop('Age_Group', axis=1, inplace=True)

        # Protein Level vs Status
        df['Protein_Level'] = pd.cut(df['Total_Protiens'], bins=[0, 5.5, 6.5, 8.5],
                                     labels=['Low', 'Normal', 'High'])
        fig3, ax3 = plt.subplots()
        sns.countplot(x='Protein_Level', hue='Status', data=df, palette='coolwarm', ax=ax3)
        ax3.set_title("Protein Level vs Liver Disease Status")
        st.pyplot(fig3)
        df.drop('Protein_Level', axis=1, inplace=True)

    elif eda_option == "Correlation Matrix":
        corr_matrix = df.select_dtypes(include=["float64", "int64"]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ---------------------- Model Results Section ---------------------- #
elif section == "Model Evaluation Results":
    st.header("Model Used --> LogisticRegression")

    # Load test data and labels from saved files
    X_test_scaled = pd.read_csv("X_liver_scaled.csv").values  # numpy array needed
    y_test = pd.read_csv("y_liver_test.csv")['Status'].values

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric("🔢 Accuracy", f"{acc*100:.2f}%")

    # Show classification report
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve and AUC
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)
