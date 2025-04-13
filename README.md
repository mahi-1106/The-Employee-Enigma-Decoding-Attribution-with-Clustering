# 💼 The Employee Enigma: Decoding Attribution With Clustering 

A complete unsupervised learning pipeline to identify and interpret employee attrition risks using a hybrid clustering approach (KMeans + GMM), dimensionality reduction, and an interactive Streamlit dashboard for HR decision-making.

---

## 🚩 Problem Statement

Organizations often lack the tools to proactively identify which employees are likely to leave. Traditional retention methods are often reactive and based on intuition. This project provides a data-driven, AI-powered system that uncovers hidden patterns in employee behavior and clusters them based on attrition risk levels.

---

## 💡 Proposed Solution

An end-to-end unsupervised learning system that:

- Uses a hybrid clustering pipeline combining KMeans and GMM
- Applies PCA for dimensionality reduction and better clustering
- Labels clusters based on attrition behavior patterns
- Deploys a user-friendly dashboard with Streamlit for HR usage

---

## ⚙️ Technologies Used

| Technology          | Role                                                        |
|---------------------|-------------------------------------------------------------|
| Python              | Core language for data analysis and modeling                |
| Scikit-learn        | Preprocessing, PCA, clustering algorithms                   |
| Matplotlib & Seaborn| Visualizations and insights                                 |
| Streamlit           | Interactive web app for HR to explore attrition risk        |
| Joblib              | Save and load models, encoders, and scalers                 |
| Pandas & NumPy      | Data manipulation and numerical processing                  |

---

## 🔄 Project Workflow

### 🧩 1. Data Acquisition & Understanding

- Load HR dataset (`Employee-Attrition-uml.csv`)
- Perform exploratory data analysis (EDA) to understand features and relationships

### 🧼 2. Data Preprocessing

- Remove constant or irrelevant features (`EmployeeNumber`, `Over18`, etc.)
- Label encode categorical variables
- Standardize numerical features using `StandardScaler`

### 📉 3. Dimensionality Reduction

- Use Principal Component Analysis (PCA) to reduce dimensions while retaining maximum variance
- Improve clustering efficiency and visualization

### 🤖 4. Hybrid Clustering Approach

- **KMeans** used to estimate the optimal number of clusters (K)
  - Evaluate using Silhouette Score and Davies-Bouldin Index
- **GMM (Gaussian Mixture Model)** is applied after PCA for flexible, probabilistic clustering
  - Models arbitrary cluster shapes and soft cluster boundaries

### 🔍 5. Model Evaluation & Cluster Optimization

- Evaluate clusters using:
  - **Silhouette Score**
  - **Davies-Bouldin Index**
  - PCA-based cluster visualization
- Optimal clusters identified: **4**

### 🧠 6. Cluster Interpretation

- Label clusters based on behavioral traits and attrition likelihood:
  - 🔴 Most Likely to Leave
  - 🟠 Likely to Leave
  - 🟡 Neutral
  - 🟢 Willing to Stay

### 🚀 7. Deployment & Scalability

- Save models and preprocessing pipeline with `joblib`
- Deploy the solution via **Streamlit dashboard** with form-based predictions
- Users can input new employee data and get real-time cluster predictions

---

## 🖼️ Architecture Diagrams

### 🏗️ Overall Architecture
<img width="437" alt="image" src="https://github.com/user-attachments/assets/9f87b1a7-c283-4b20-8afd-2a5d01a54fb5" />

### ⚙️ Working Model Pipeline
<img width="422" alt="image" src="https://github.com/user-attachments/assets/2ded3955-1544-4569-bc2a-90cd58fb6819" />

---

## 🌐 Run the Dashboard

```bash
cd streamlit_Employee-Attrition
streamlit run app.py
