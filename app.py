import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load saved model components
means = joblib.load("models/gmm_means.pkl")
covariances = joblib.load("models/gmm_covariances.pkl")
weights = joblib.load("models/gmm_weights.pkl")
pca = joblib.load("models/gmm_pca.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Load original dataset
df = pd.read_csv("Employee-Attrition-uml.csv")
original_df = df.copy()
employee_names = df["EmployeeName"].tolist() if "EmployeeName" in df.columns else df.index.astype(str).tolist()

# Drop irrelevant columns
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours"], errors="ignore")

# Separate categorical and numerical columns
categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
numerical_cols = [col for col in df.columns if col not in categorical_cols]

# Apply saved LabelEncoders
for col in categorical_cols:
    le = label_encoders.get(col)
    if le:
        df[col] = le.transform(df[col])
    else:
        st.warning(f"Missing LabelEncoder for column: {col}")

# Standardize only numerical columns
df[numerical_cols] = scaler.transform(df[numerical_cols])

# Streamlit UI setup
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🧠 Employee Attrition & Retention Suggestion System")

# Employee selection
employee_input = st.selectbox("Select an Employee ID", employee_names)

if st.button("Predict Cluster and Suggest"):
    try:
        # Get selected employee index
        emp_idx = original_df[original_df["EmployeeName"] == employee_input].index[0] if "EmployeeName" in original_df.columns else int(employee_input)

        # Select only scaled numerical data (used in PCA)
        emp_data = df[numerical_cols].iloc[[emp_idx]]

        # Apply PCA
        emp_pca = pca.transform(emp_data)

        # GMM prediction
        probs = [weights[i] * multivariate_normal.pdf(emp_pca, mean=means[i], cov=covariances[i]) for i in range(len(means))]
        probs = np.array(probs).reshape(-1)
        cluster = np.argmax(probs)

        # Cluster labels and suggestions
        label_map = {
            0: "Willing to Stay",
            1: "Less Likely to Leave",
            2: "Most Likely to Leave",
            3: "Satisfied Performer"
        }

        suggestions = {
            0: "No immediate action needed. Continue engagement.",
            1: "Consider offering a promotion or reskilling opportunities.",
            2: "High risk of attrition. Increase salary, enhance role clarity, or flexible work options.",
            3: "Recognize achievements. Promote leadership programs."
        }

        # Output prediction and suggestion
        st.subheader("🧩 Prediction")
        st.success(f"Cluster Label: {label_map[cluster]}")

        st.subheader("💡 Suggested Action")
        st.info(suggestions[cluster])

    except Exception as e:
        st.error(f"An error occurred: {e}")
