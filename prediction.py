import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

st.set_page_config(page_title="Health Prediction Dashboard", layout="wide")
st.title("🩺 Multi-Disease Prediction & Analysis")

# --- Sidebar: Select Disease ---
disease = st.sidebar.selectbox("Select Disease", ["Chronic Kidney Disease", "Parkinson's Disease","Liver Disease"])

# --- Load CKD Assets ---
@st.cache_data
def load_ckd_data():
    return pd.read_csv("cleaned_kidney_data.csv")

def load_ckd_model_scaler():
    with open("kidney_rf_model.pkl", "rb") as m:
        model = pickle.load(m)
    with open("kidney_scaler.pkl", "rb") as s:
        scaler = pickle.load(s)
    return model, scaler

# --- Load Parkinson's Assets ---
@st.cache_data
def load_parkinsons_data():
    return pd.read_csv("cleaned_parkinsons_data.csv")

def load_parkinsons_model_scaler():
    with open("parkinsons_rf_model.pkl", "rb") as m:
        model = pickle.load(m)
    with open("parkinsons_scaler.pkl", "rb") as s:
        scaler = pickle.load(s)
    return model, scaler

# --- Load Liver  Assets ---
@st.cache_data
def load_liver_data():
    return pd.read_csv("cleaned_liver_data.csv")

def load_liver_model_scaler():
    with open("liver_model.pkl", "rb") as m:
        model = pickle.load(m)
    with open("liver_scaler.pkl", "rb") as s:
        scaler = pickle.load(s)
    return model, scaler

# ------------------- CKD Dashboard ------------------- #
if disease == "Chronic Kidney Disease":
    section = st.sidebar.radio("CKD Section", ["Prediction", "EDA", "Model Evaluation Results"])
    df = load_ckd_data()
    model, scaler = load_ckd_model_scaler()

    if section == "Prediction":
        st.header("🔍 CKD Prediction")
        age = st.number_input("Age", 1, 100)
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

        mapping = {'Normal': 0, 'Abnormal': 1, 'Not Present': 0, 'Present': 1,
                   'No': 0, 'Yes': 1, 'Poor': 0, 'Good': 1}

        input_data = np.array([[age, bp, sg, albumin, sugar,
                                mapping[rbc], mapping[pc], mapping[pcc], mapping[ba],
                                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                mapping[htn], mapping[dm], mapping[cad],
                                mapping[appet], mapping[pe], mapping[ane]]])

        if st.button("Predict CKD"):
            scaled = scaler.transform(input_data)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ Likely to have CKD\nConfidence: {prob:.2f}")
            else:
                st.success(f"✅ Not likely to have CKD\nConfidence: {1 - prob:.2f}")

    elif section == "EDA":
        st.header("📊 CKD - EDA")
        eda_option = st.selectbox("Select Plot", [
            "Box Plots", "Histograms", "CKD Status Count",
            "EDA Features vs Status", "Correlation Matrix"
        ])
        if eda_option == "Box Plots":
            numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            if "Status" in numerical_cols:
                numerical_cols.remove("Status")
            for col in numerical_cols:
                fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        elif eda_option == "Histograms":
            num_cols = df.select_dtypes(include=["float64", "int64"]).drop("Status", axis=1, errors='ignore').columns
            for col in num_cols:
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram: {col}"))
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

    elif section == "Model Evaluation Results":
        st.header("CKD Model: RandomForestClassifier")
        try:
            X_test = pd.read_csv("X_test_kidney_scaled.csv").values
            y_test = pd.read_csv("y_kidney_test.csv")['Status'].values
            y_pred = model.predict(X_test)
            st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            st.subheader("--> Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("--> Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("--> ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title("ROC Curve")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------- Parkinson's Dashboard ------------------- #
elif disease == "Parkinson's Disease":
    section = st.sidebar.radio("Parkinson's Section", ["Prediction", "EDA", "Model Evaluation Results"])
    df = load_parkinsons_data()
    model, scaler = load_parkinsons_model_scaler()

    if section == "Prediction":
        st.header("🔍 Parkinson’s Prediction")
        input_data = []
        features = df.drop(columns=["status"]).columns.tolist()

        for f in features:
            val = st.number_input(f"{f}:", value=0.0)
            input_data.append(val)

        if st.button("Predict Parkinson's"):
            input_df = pd.DataFrame([input_data], columns=features)
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ Likely to have Parkinson's\nConfidence: {prob:.2f}")
            else:
                st.success(f"✅ Not likely to have Parkinson's\nConfidence: {1 - prob:.2f}")

    elif section == "EDA":
        st.header("📊 Parkinson’s - EDA")
        eda_option = st.selectbox("Select Plot", [
            "Box Plots", "Histograms", "Parkinson's Status Count",
            "Correlation Matrix", "EDA Features vs Status"
        ])

        numeric = df.select_dtypes(include=["float64", "int64"]).drop(columns=["status"])
        if eda_option == "Box Plots":
            numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            if "Status" in numerical_cols:
                numerical_cols.remove("Status")
            for col in numerical_cols:
                fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        elif eda_option == "Histograms":
            for col in numeric.columns:
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram: {col}"))
        elif eda_option == "Parkinson's Status Count":
            st.bar_chart(df["status"].value_counts())
        elif eda_option == "Correlation Matrix":
            corr = df.corr()
            top_features = corr['status'].abs().sort_values(ascending=False).head(13).index
            small_corr = df[top_features].corr()
            sns.heatmap(small_corr, annot=True, cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()
        elif eda_option == "EDA Features vs Status":
            sns.barplot(x='status', y='mdvp_fo_hz_', data=df)
            st.pyplot(plt.gcf())
            plt.clf()
            sns.barplot(x='status', y='hnr', data=df)
            st.pyplot(plt.gcf())
            plt.clf()
            sns.barplot(x='status', y='spread1', data=df)
            st.pyplot(plt.gcf())
            plt.clf()

    elif section == "Model Evaluation Results":
        st.header("Parkinson’s Model: RandomForestClassifier")
        try:
            X_test = pd.read_csv("X_test_parkinsons_scaled.csv")
            y_test_df = pd.read_csv("y_parkinsons_test.csv")
            y_test = y_test_df['status'] if 'status' in y_test_df else y_test_df.iloc[:, 0]

            y_pred = model.predict(X_test)
            st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            st.subheader("--> Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("--> Confusion Matrix")
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("--> ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error: {e}")
# ------------------- Liver Disease Dashboard ------------------- #
elif disease == "Liver Disease":
    section = st.sidebar.radio("Liver Section", ["Prediction", "EDA", "Model Evaluation Results"])
    df = load_liver_data()
    model, scaler = load_liver_model_scaler()

    if section == "Prediction":
        st.header("🔍 Liver Disease Prediction")
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

    elif section == "EDA":
        st.header("📊 Liver Disease - EDA")
        eda_option = st.selectbox("Select Plot", [
            "Box Plots", "Histograms", "Liver Disease Status Count",
            "Correlation Matrix", "EDA Features vs Status"
        ])

        numeric = df.select_dtypes(include=["float64", "int64"]).drop(columns=["Status"])
        if eda_option == "Box Plots":
            numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            if "Status" in numerical_cols:
                numerical_cols.remove("Status")
            for col in numerical_cols:
                fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        elif eda_option == "Histograms":
            for col in numeric.columns:
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram: {col}"))

        elif eda_option == "Liver Disease Status Count":
            sns.countplot(y='Status', data=df, palette='pastel')
            st.pyplot(plt.gcf())
            plt.clf()

        elif eda_option == "Correlation Matrix":
            corr = df.corr()
            top_features = corr['Status'].abs().sort_values(ascending=False).head(13).index
            small_corr = df[top_features].corr()
            sns.heatmap(small_corr, annot=True, cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()

        elif eda_option == "EDA Features vs Status":
            sns.countplot(x='Gender', hue='Status', data=df, palette='Set2')
            st.pyplot(plt.gcf())
            plt.clf()
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])
            sns.countplot(x='Age_Group', hue='Status', data=df, palette='Set1')
            st.pyplot(plt.gcf())
            plt.clf()
            df.drop('Age_Group', axis=1, inplace=True)
            df['Protein_Level'] = pd.cut(df['Total_Protiens'], bins=[0, 5.5, 6.5, 8.5], labels=['Low', 'Normal', 'High'])
            sns.countplot(x='Protein_Level', hue='Status', data=df, palette='coolwarm')
            st.pyplot(plt.gcf())
            plt.clf()
            df.drop('Protein_Level', axis=1, inplace=True)

    elif section == "Model Evaluation Results":
        st.header("Liver Disease Model: LogisticRegression")
        try:
            X_test = pd.read_csv("X_liver_scaled.csv")
            y_test_df = pd.read_csv("y_liver_test.csv")
            y_test = y_test_df['Status'] if 'Status' in y_test_df else y_test_df.iloc[:, 0]

            y_pred = model.predict(X_test)
            st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            st.subheader("--> Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("--> Confusion Matrix")
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("--> ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error: {e}")
