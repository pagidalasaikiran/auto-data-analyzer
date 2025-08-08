import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Auto Data Analyzer", layout="wide")

st.title("📊 Auto Data Analyzer")
st.markdown("Upload your CSV or Excel file to automatically explore and visualize your data.")

# Upload
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

        st.subheader("📌 Data Preview")
        st.dataframe(df.head())

        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.subheader("📋 Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0] if missing.any() else "No missing values!")

        st.subheader("📐 Summary Statistics")
        st.dataframe(df.describe())

        numeric_df = df.select_dtypes(include='number')

        if numeric_df.shape[1] > 1:
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("📈 Column Distributions")
        cols = st.multiselect("Select numeric columns to plot", numeric_df.columns, default=numeric_df.columns[:2])
        for col in cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Please upload a file to start analysis.")
