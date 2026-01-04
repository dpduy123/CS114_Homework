import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App Title
st.title("📊 Student Performance Analyzer")

# Upload CSV File
uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview", df.head())

    df = df.drop(columns=["Student"])
    #print(df.head())
    #print(df.dtypes) 
    
    # Select Features & Target
    target_column = st.selectbox("🎯 Select Target Column (e.g., Final Score)", df.columns)
    feature_columns = st.multiselect("📌 Select Feature Columns", df.columns, default=df.columns[:-1])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=23520362)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        st.write("### 📊 Model Performance")
        st.write(f"🔹 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"🔹 MSE: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"🔹 R² Score: {r2_score(y_test, y_pred):.2f}")

        # Feature Importance
        feature_importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
        st.write("### 🔥 Feature Importance")
        st.bar_chart(feature_importance)

        # Correlation Heatmap
        st.write("### 📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
