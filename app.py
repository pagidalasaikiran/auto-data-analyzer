import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from dotenv import load_dotenv
from google import genai
import plotly.express as px
import plotly.figure_factory as ff
from openpyxl import Workbook

# -----------------------------
# LOAD GEMINI API
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI API KEY NOT FOUND. PLEASE ADD IT TO .env FILE")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Professional Data Analyzer", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:#1ABC9C;'>📊 PROFESSIONAL DATA ANALYZER</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 UPLOAD YOUR CSV FILE", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ ERROR READING CSV: {e}")
        st.stop()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    st.success("✅ CSV LOADED SUCCESSFULLY")

    # -----------------------------
    # TABS
    # -----------------------------
    tabs = st.tabs([
        "📋 BASIC DATA ANALYSIS & EXPLORATION",
        "📊 DASHBOARD",
        "🧹 DATA CLEANING",
        "🔄 DATA TRANSFORMATION",
        "📈 DATA VISUALIZATION",
        "🤖 GEMINI INSIGHTS",
        "⚡ MANUAL ANALYSIS",
        "💾 DOWNLOAD REPORT"
    ])

    # -----------------------------
    # 1. BASIC DATA ANALYSIS & EXPLORATION
    # -----------------------------
    with tabs[0]:
        st.header("📋 BASIC DATA ANALYSIS & EXPLORATION")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("**Column Attributes:**")
        column_info = pd.DataFrame({
            "COLUMN NAME": df.columns,
            "DATA TYPE": df.dtypes.values,
            "MISSING VALUES": df.isnull().sum().values,
            "UNIQUE VALUES": [df[col].nunique() for col in df.columns],
            "EXAMPLE VALUE": [
                df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else ""
                for col in df.columns
            ]
        })
        st.data_editor(column_info, hide_index=True, use_container_width=True, key="basic_analysis_editor")

        # Download CSV
        csv_buffer = io.StringIO()
        column_info.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Basic Analysis CSV",
            data=csv_buffer.getvalue(),
            file_name="basic_analysis.csv",
            mime="text/csv"
        )

    # -----------------------------
    # 2. DASHBOARD
    # -----------------------------
    with tabs[1]:
        st.header("📊 PROFESSIONAL DASHBOARD")
        total_rows, total_cols = df.shape
        numeric_count = len(numeric_cols)
        categorical_count = len(categorical_cols)
        missing_pct = round(df.isnull().sum().sum() / (total_rows*total_cols) * 100, 2)
        duplicate_count = df.duplicated().sum()
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Rows", total_rows)
        kpi2.metric("Total Columns", total_cols)
        kpi3.metric("Numeric Columns", numeric_count)
        kpi4.metric("Categorical Columns", categorical_count)
        kpi5.metric("Missing %", f"{missing_pct}% ({duplicate_count} duplicates)")
        
        st.divider()
        
        # Interactive Filters
        st.subheader("Filter Data")
        filtered_df = df.copy()
        if categorical_cols:
            with st.expander("Apply Categorical Filters"):
                for col in categorical_cols:
                    selected_vals = st.multiselect(f"{col} filter:", df[col].dropna().unique(), key=f"dashboard_filter_{col}")
                    if selected_vals:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
        st.write(f"Filtered Rows: {filtered_df.shape[0]}")

        # Numeric Summary & Charts
        if numeric_cols:
            st.subheader("Numeric Column Summary")
            for col in numeric_cols:
                mean_val, median_val = filtered_df[col].mean(), filtered_df[col].median()
                min_val, max_val = filtered_df[col].min(), filtered_df[col].max()
                std_val = filtered_df[col].std()
                with st.expander(f"{col} Summary"):
                    st.write(f"**Mean:** {mean_val:.3f}  |  **Median:** {median_val:.3f}  |  **Min:** {min_val}  |  **Max:** {max_val}  |  **Std Dev:** {std_val:.3f}")
                    fig_hist = px.histogram(filtered_df, x=col, nbins=30, title=f"{col} Distribution")
                    fig_box = px.box(filtered_df, y=col, title=f"{col} Boxplot")
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"dashboard_hist_{col}")
                    st.plotly_chart(fig_box, use_container_width=True, key=f"dashboard_box_{col}")

        # Categorical Counts
        if categorical_cols:
            st.subheader("Categorical Column Counts")
            for col in categorical_cols:
                count_df = filtered_df[col].value_counts().reset_index()
                count_df.columns = [col, "Count"]
                st.data_editor(count_df, hide_index=True, use_container_width=True, key=f"dashboard_cat_editor_{col}")
                fig_cat = px.bar(count_df, x=col, y="Count", title=f"{col} Distribution")
                st.plotly_chart(fig_cat, use_container_width=True, key=f"dashboard_cat_bar_{col}")

        # Correlation Heatmap
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = filtered_df[numeric_cols].corr()
            fig_corr = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale="Viridis", showscale=True)
            st.plotly_chart(fig_corr, use_container_width=True, key="dashboard_corr_heatmap")

        # Missing Values Heatmap
        st.subheader("Missing Values Heatmap")
        fig_missing = px.imshow(filtered_df.isnull().T, color_continuous_scale="Viridis", title="Missing Values Heatmap", labels={"x":"Rows","y":"Columns","color":"Missing"})
        st.plotly_chart(fig_missing, use_container_width=True, key="dashboard_missing_heatmap")

        # Download Filtered Data
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Filtered Data CSV",
            data=csv_buffer.getvalue(),
            file_name="dashboard_filtered_data.csv",
            mime="text/csv"
        )

    # -----------------------------
    # 3. DATA CLEANING
    # -----------------------------
    with tabs[2]:
        st.header("🧹 DATA CLEANING")
        if st.button("Remove Duplicates", key="clean_remove_dup"):
            df = df.drop_duplicates()
            st.success("✅ Duplicates Removed")
        if st.button("Fill Missing Values", key="clean_fill_na"):
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna("UNKNOWN", inplace=True)
            st.success("✅ Missing Values Filled")
        if st.button("Remove Outliers (IQR)", key="clean_remove_outliers"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25,0.75])
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            st.success("✅ Outliers Removed")
        cleaned_preview = df.head(10).copy()
        st.data_editor(cleaned_preview, hide_index=True, use_container_width=True, key="cleaned_data_editor")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("⬇️ Download Cleaned Data CSV", data=csv_buffer.getvalue(), file_name="cleaned_data.csv", mime="text/csv")

    # -----------------------------
    # 4. DATA TRANSFORMATION
    # -----------------------------
    with tabs[3]:
        st.header("🔄 DATA TRANSFORMATION")
        selected_col = st.selectbox("Select Column to Transform:", df.columns.tolist(), key="transform_col")
        target_type = st.selectbox("Convert To Type:", ["int", "float", "string", "category"], key="transform_type")
        if st.button("Apply Conversion", key="transform_apply"):
            try:
                df[selected_col] = df[selected_col].astype(target_type)
                st.success(f"✅ {selected_col} converted to {target_type}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        if numeric_cols:
            trans_col = st.selectbox("Select Numeric Column to Transform:", numeric_cols, key="transform_num_col")
            trans_method = st.selectbox("Transformation Method:", ["Normalize", "Log", "Square Root"], key="transform_method")
            if st.button("Apply Transformation to Numeric", key="transform_apply_numeric"):
                if trans_method=="Normalize":
                    df[trans_col] = (df[trans_col]-df[trans_col].min())/(df[trans_col].max()-df[trans_col].min())
                elif trans_method=="Log":
                    df[trans_col] = np.log1p(df[trans_col])
                elif trans_method=="Square Root":
                    df[trans_col] = np.sqrt(df[trans_col])
                st.success(f"✅ {trans_col} {trans_method} applied")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("⬇️ Download Transformed CSV", data=csv_buffer.getvalue(), file_name="transformed_data.csv", mime="text/csv")

    # -----------------------------
    # 5. DATA VISUALIZATION (Dashboard Style)
    # -----------------------------
    with tabs[4]:
        st.header("📈 DATA VISUALIZATION")
        # Numeric Columns
        if numeric_cols:
            st.subheader("Numeric Columns")
            for col in numeric_cols:
                with st.expander(f"{col} Summary & Charts"):
                    mean_val, median_val = df[col].mean(), df[col].median()
                    min_val, max_val = df[col].min(), df[col].max()
                    std_val = df[col].std()
                    st.write(f"**Mean:** {mean_val:.3f}  |  **Median:** {median_val:.3f}  |  **Min:** {min_val}  |  **Max:** {max_val}  |  **Std Dev:** {std_val:.3f}")
                    fig_hist = px.histogram(df, x=col, nbins=30, title=f"{col} Distribution")
                    fig_box = px.box(df, y=col, title=f"{col} Boxplot")
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"viz_hist_{col}")
                    st.plotly_chart(fig_box, use_container_width=True, key=f"viz_box_{col}")
        # Categorical Columns
        if categorical_cols:
            st.subheader("Categorical Columns")
            for col in categorical_cols:
                with st.expander(f"{col} Counts & Chart"):
                    count_df = df[col].value_counts().reset_index()
                    count_df.columns = [col, "Count"]
                    st.data_editor(count_df, hide_index=True, use_container_width=True, key=f"viz_cat_editor_{col}")
                    fig_cat = px.bar(count_df, x=col, y="Count", title=f"{col} Distribution")
                    st.plotly_chart(fig_cat, use_container_width=True, key=f"viz_cat_bar_{col}")

    # -----------------------------
    # 6. GEMINI INSIGHTS
    # -----------------------------
    with tabs[5]:
        st.header("🤖 GEMINI INSIGHTS")
        col_options = st.multiselect("Select Columns for Insights:", df.columns.tolist(), key="gemini_cols")
        analysis_df = df[col_options] if col_options else df
        if st.button("Generate Insights", key="gemini_generate"):
            with st.spinner("🔍 Gemini is analyzing..."):
                try:
                    data_snippet = analysis_df.head(20).to_string(index=False)
                    prompt = f"Analyze this dataset and give key insights:\n\n{data_snippet}"
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")

    # -----------------------------
    # 7. MANUAL ANALYSIS
    # -----------------------------
    with tabs[6]:
        st.header("⚡ MANUAL ANALYSIS")
        st.markdown("Perform operations manually on your dataset here.")
        manual_df = df.copy()

        # Filter Rows
        st.subheader("Filter Rows")
        col_filter = st.selectbox("Select Column:", manual_df.columns.tolist(), key="manual_filter_col")
        if manual_df[col_filter].dtype == object:
            val_filter = st.text_input("Enter value to filter", key="manual_filter_val")
            if st.button("Apply Text Filter", key="manual_apply_text_filter"):
                filtered_manual_df = manual_df[manual_df[col_filter].astype(str).str.contains(val_filter, case=False, na=False)]
                st.data_editor(filtered_manual_df, hide_index=True, use_container_width=True, key="manual_text_filtered")
        else:
            min_val, max_val = st.slider("Select range:", float(manual_df[col_filter].min()), float(manual_df[col_filter].max()), (float(manual_df[col_filter].min()), float(manual_df[col_filter].max())), key="manual_slider_filter")
            if st.button("Apply Numeric Filter", key="manual_apply_numeric_filter"):
                filtered_manual_df = manual_df[(manual_df[col_filter] >= min_val) & (manual_df[col_filter] <= max_val)]
                st.data_editor(filtered_manual_df, hide_index=True, use_container_width=True, key="manual_numeric_filtered")

        # Sort
        st.subheader("Sort Data")
        sort_col = st.selectbox("Select Column to Sort:", manual_df.columns.tolist(), key="manual_sort_col")
        sort_order = st.radio("Order:", ["Ascending", "Descending"], key="manual_sort_order")
        if st.button("Apply Sort", key="manual_apply_sort"):
            sorted_df = manual_df.sort_values(by=sort_col, ascending=(sort_order=="Ascending"))
            st.data_editor(sorted_df, hide_index=True, use_container_width=True, key="manual_sorted")

        # Add Column
        st.subheader("Add Column")
        new_col_name = st.text_input("New Column Name", key="manual_new_col_name")
        if st.button("Add Column with Zeros", key="manual_add_col"):
            if new_col_name:
                manual_df[new_col_name] = 0
                st.success(f"✅ Column '{new_col_name}' added")
                st.data_editor(manual_df, hide_index=True, use_container_width=True, key="manual_added_col")

    # -----------------------------
    # 8. DOWNLOAD REPORT
    # -----------------------------
    with tabs[7]:
        st.header("💾 DOWNLOAD REPORT")
        if st.button("Generate Excel Report", key="download_report_btn"):
            with st.spinner("📦 Creating Excel..."):
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Data', index=False)
                        summary = df.describe(include='all').T
                        summary.to_excel(writer, sheet_name='Summary', index=True)
                    output.seek(0)
                    st.download_button("⬇️ Download Report", data=output, file_name="professional_data_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"❌ ERROR GENERATING REPORT: {e}")

