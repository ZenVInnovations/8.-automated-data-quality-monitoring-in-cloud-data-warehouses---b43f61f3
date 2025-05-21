import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

import pandas as pd

def load_data(data):
    return pd.read_csv(data)

data = load_data(r"C:\Users\ULLAS\OneDrive\Desktop\8.-automated-data-quality-monitoring-in-cloud-data-warehouses---b43f61f3-main\Diabetes Missing Data.csv")
print(data.head())

def check_null_values(df):
    return df.isnull().sum()

def check_duplicates(df):
    return df.duplicated().sum()

def detect_anomalies(df, columns):
    df_numeric = df[columns].select_dtypes(include=[np.number])
    if df_numeric.empty:
        return pd.DataFrame()

    df_numeric.fillna(df_numeric.median(), inplace=True)
    model = IsolationForest(contamination=0.05, random_state=42)
    df_numeric['anomaly'] = model.fit_predict(df_numeric)
    anomalies = df[df_numeric['anomaly'] == -1]
    return anomalies


st.title("📊 AI-Based Data Quality Monitoring")


try:
    data = load_data(r"C:\Users\ULLAS\OneDrive\Desktop\8.-automated-data-quality-monitoring-in-cloud-data-warehouses---b43f61f3-main\Diabetes Missing Data.csv")
    st.subheader("📌 Data Preview")
    st.dataframe(data.head())


    null_values = check_null_values(data)
    st.subheader("🛠 Missing Values")
    st.write(null_values)


    st.subheader("📊 Missing Values Visualization")
    fig, ax = plt.subplots()
    sns.barplot(x=null_values.index, y=null_values.values, ax=ax, palette="coolwarm")
    ax.set_ylabel("Count")
    ax.set_title("Missing Values Per Column")
    st.pyplot(fig)


    duplicates = check_duplicates(data)
    st.subheader("🛠 Duplicate Rows")
    st.write(f"Total Duplicates: {duplicates}")


    st.subheader("📊 Duplicate Rows Distribution")
    fig, ax = plt.subplots()
    labels = ["Unique Rows", "Duplicates"]
    values = [len(data) - duplicates, duplicates]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["skyblue", "red"])
    st.pyplot(fig)


    st.subheader("🔍 Select Columns for Anomaly Detection")
    selected_columns = st.multiselect("Choose Numeric Columns", data.select_dtypes(include=[np.number]).columns)
    
    if selected_columns:
        anomalies = detect_anomalies(data, selected_columns)
        st.subheader("🔴 Detected Anomalies")
        st.dataframe(anomalies)


        st.subheader("📊 Anomaly Detection Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=data[selected_columns], ax=ax, palette="Set2")
        ax.set_title("Box Plot for Outlier Detection")
        st.pyplot(fig)

except FileNotFoundError:
    st.error("File 'Diabetes Missing Data.csv' not found in the current folder.")
