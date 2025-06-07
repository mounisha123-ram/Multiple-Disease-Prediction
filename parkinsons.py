import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

st.set_page_config(page_title="Parkinson's Dashboard", layout="wide")
st.title("🧠 Parkinson's Disease Prediction & Analysis")

# Sidebar navigation
section = st.sidebar.radio("Select Section", ["Prediction", "EDA", "Model Evaluation Results"])

# Load model and scaler
with open("parkinsons_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("parkinsons_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_parkinsons_data.csv")

df = load_data()

# Prediction Section
if section == "Prediction":
    st.header("🔍 Parkinson's Prediction")

    feature_names = df.drop(columns=["status"]).columns.tolist()
    input_data = []

    st.subheader("Enter feature values")
    for feature in feature_names:
        val = st.number_input(f"{feature}:", value=0.0, format="%.5f")
        input_data.append(val)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to have Parkinson's Disease.\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Not likely to have Parkinson's Disease.\nProbability: {1 - prob:.2f}")

# EDA Section
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    eda_option = st.selectbox("Select EDA Plot", [
        "Box Plots", 
        "Histograms", 
        "Parkinson's Status Count", 
        "Correlation Matrix", 
        "EDA Features vs Status"
    ])

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if "status" in numeric_cols:
        numeric_cols.remove("status")

    if eda_option == "Box Plots":
        for col in numeric_cols:
            fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all", template="plotly_white")
            st.plotly_chart(fig)

    elif eda_option == "Histograms":
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Histogram of {col}", marginal="rug", nbins=30)
            st.plotly_chart(fig)

    elif eda_option == "Parkinson's Status Count":
        plt.figure(figsize=(10, 3))
        ax = sns.countplot(y='status', data=df, palette='pastel')
        plt.title("Parkinson's Status", fontsize=14)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Status", fontsize=12)

        for p in ax.patches:
            count = int(p.get_width())
            ax.text(p.get_width() + 1, p.get_y() + p.get_height() / 2, count, va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    elif eda_option == "Correlation Matrix":
        corr_matrix = df.corr()
        target_corr = corr_matrix['status'].sort_values(ascending=False)
        top_features = target_corr.head(13).index.tolist()
        if 'status' in top_features:
            top_features.remove('status')
        top_features = top_features[:13]
        top_features_with_status = top_features + ['status']
        small_corr_matrix = df[top_features_with_status].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(small_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix of Top Features with Status')
        st.pyplot(fig)


    elif eda_option == "EDA Features vs Status":
        # mdvp_fo_hz_ vs status (Boxplot)
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='status', y='mdvp_fo_hz_', data=df, palette='Set2')
        plt.title("Fundamental Frequency (mdvp_fo_hz_) by Status")
        plt.xlabel("Status (0 = Healthy, 1 = Parkinson's)")
        plt.ylabel("mdvp_fo_hz_")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

        # hnr vs status (Violin plot)
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='status', y='hnr', data=df, palette='cool')
        plt.title("Harmonic-to-Noise Ratio (hnr) by Status")
        plt.xlabel("Status (0 = Healthy, 1 = Parkinson's)")
        plt.ylabel("hnr")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

        # spread1 vs status (Boxplot)
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='status', y='spread1', data=df, palette='Set3')
        plt.title("Spread1 by Status")
        plt.xlabel("Status (0 = Healthy, 1 = Parkinson's)")
        plt.ylabel("spread1")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()


# Model Results Section
elif section == "Model Evaluation Results":
    st.header("Model Used --> RandomForestClassifier")

    try:
        X_test_scaled = pd.read_csv("X_test_parkinsons_scaled.csv")
        y_test_df = pd.read_csv("y_parkinsons_test.csv")

        if 'status' in y_test_df.columns:
            y_test = y_test_df['status'].values
        else:
            y_test = y_test_df.iloc[:, 0].values

        y_pred = model.predict(X_test_scaled.values)

        acc = accuracy_score(y_test, y_pred)

        st.metric("✅ Accuracy", f"{acc * 100:.2f}%")
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading test data or evaluating model: {e}")

