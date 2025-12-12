import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

st.title("Student Assessment Predictions Dashboard")

# ==============================
# Load Results CSVs
# ==============================
df_rq1 = pd.read_csv("results_rq1.csv")
df_rq2 = pd.read_csv("results_rq2.csv")
df_rq3 = pd.read_csv("results_rq3.csv")

# ==============================
# Load Models
# ==============================
# RQ1: Polynomial Regression
model_rq1 = pickle.load(open("model_RQ1_Polynomial_Regression_(deg=2).pkl", "rb"))
# RQ2 & RQ3: Multiple Linear Regression
model_rq2 = pickle.load(open("model_RQ2_Multiple_Linear_Regression.pkl", "rb"))
model_rq3 = pickle.load(open("model_RQ3_Multiple_Linear_Regression.pkl", "rb"))

# ==============================
# Plotting Function
# ==============================
def plot_actual_vs_predicted(csv_file, title):
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['Actual'], df['Predicted'], alpha=0.6, edgecolors='k')
    ax.plot([df['Actual'].min(), df['Actual'].max()],
            [df['Actual'].min(), df['Actual'].max()],
            'r--', lw=2)
    ax.set_xlabel("Actual Scores")
    ax.set_ylabel("Predicted Scores")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ==============================
# Sidebar for RQ selection
# ==============================
st.sidebar.title("Select Research Question")
rq_choice = st.sidebar.radio("Choose RQ", ("RQ1: S-I", "RQ2: S-II", "RQ3: Final"))

# ==============================
# RQ1: Midterm I
# ==============================
if rq_choice == "RQ1: S-I":
    st.header("RQ1: Predicting Midterm I (S-I)")
    st.dataframe(df_rq1)
    plot_actual_vs_predicted("preds_rq1.csv", "RQ1: Midterm I Predictions")

    st.subheader("Predict Midterm I Score")
    quiz1 = st.number_input("Enter Quiz 1 Score", min_value=0.0, max_value=100.0)
    quiz2 = st.number_input("Enter Quiz 2 Score", min_value=0.0, max_value=100.0)
    assignment1 = st.number_input("Enter Assignment 1 Score", min_value=0.0, max_value=100.0)
    assignment2 = st.number_input("Enter Assignment 2 Score", min_value=0.0, max_value=100.0)

    if st.button("Predict Midterm I"):
        # Use same feature order as during training
        features = [[quiz1, quiz2, assignment1, assignment2]]

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)  # scales only the single row

        # Polynomial transform
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features_poly = poly.fit_transform(features_scaled)

        predicted_mid1 = model_rq1.predict(features_poly)[0]
        st.success(f"Predicted Midterm I Score: {predicted_mid1:.2f}")

# ==============================
# RQ2: Midterm II
# ==============================
elif rq_choice == "RQ2: S-II":
    st.header("RQ2: Predicting Midterm II (S-II)")
    st.dataframe(df_rq2)
    plot_actual_vs_predicted("preds_rq2.csv", "RQ2: Midterm II Predictions")

    st.subheader("Predict Midterm II Score")
    mid1_score = st.number_input("Enter Midterm I Score", min_value=0.0, max_value=100.0)
    project_score = st.number_input("Enter Project Score", min_value=0.0, max_value=100.0)

    if st.button("Predict Midterm II"):
        features = [[mid1_score, project_score]]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        predicted_mid2 = model_rq2.predict(features_scaled)[0]
        st.success(f"Predicted Midterm II Score: {predicted_mid2:.2f}")

# ==============================
# RQ3: Final Exam
# ==============================
else:
    st.header("RQ3: Predicting Final Exam")
    st.dataframe(df_rq3)
    plot_actual_vs_predicted("preds_rq3.csv", "RQ3: Final Exam Predictions")

    st.subheader("Predict Final Exam Score")
    mid1_score = st.number_input("Enter Midterm I Score", min_value=0.0, max_value=100.0)
    mid2_score = st.number_input("Enter Midterm II Score", min_value=0.0, max_value=100.0)

    if st.button("Predict Final Exam"):
        features = [[mid1_score, mid2_score]]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        predicted_final = model_rq3.predict(features_scaled)[0]
        st.success(f"Predicted Final Exam Score: {predicted_final:.2f}")

# ==============================
# Optional Image / Logo
# ==============================
st.image("C:/Users/ABC/Downloads/my_image.png", caption="My Image", use_column_width=True)
