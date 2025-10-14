![Python](https://img.shields.io/badge/Python-3.10+-green) ![Data Analysis](https://img.shields.io/badge/Streamlit-App-blue) 

# 📊 Professional Data Analyzer

**Professional Data Analyzer** is a powerful and interactive web application built with **Streamlit** that enables users to perform comprehensive data exploration, cleaning, transformation, visualization, manual analysis, and reporting—all in a single platform. This tool is ideal for data analysts, data scientists, students, and professionals who want quick insights from their datasets.

---
## 📁 Folder Structure
```
Auto-Data-Analyzer/
│
├─ app.py                  # Main Streamlit application
├─ requirements.txt        # Dependencies
├─ .env                    # Gemini API key (not in repo)
├─ README.md               # Project documentation
└─ assets/                 # Optional folder for screenshots, GIFs, icons
```
---

### 🔹 Key Features

### 1️⃣ CSV Upload
- Upload your datasets in CSV format effortlessly.
- Automatic detection of numeric and categorical columns.

### 2️⃣ Basic Data Analysis & Exploration
- View column types, missing values, unique counts, and example values.
- Quickly understand the dataset's structure.

### 3️⃣ Interactive Dashboard
- **Key Metrics**: Total rows, columns, missing values, numeric/categorical column counts.
- **Numeric Summary**: Mean, median, min, max, standard deviation.
- **Categorical Summary**: Value counts with interactive bar charts.
- **Correlation Heatmap** and **Missing Values Heatmap** for a visual overview.

### 4️⃣ Data Cleaning
- Remove duplicates with one click.
- Fill missing values automatically (numeric → 0, categorical → "UNKNOWN").
- Remove outliers using the IQR method.

### 5️⃣ Data Transformation
- Convert columns to different data types (`int`, `float`, `string`, `category`).
- Apply numeric transformations: normalization, log, or square root.

### 6️⃣ Data Visualization
- Interactive charts for numeric and categorical columns.
- Dashboard-style expandable summaries for easy exploration.

### 7️⃣ Gemini Insights
- Generate AI-powered insights on selected columns.
- Quick analysis using the **Gemini API** for key trends and patterns.
- Supports partial or full dataset insights for flexibility.

### 8️⃣ Manual Analysis
- Perform filters, sorting, add new columns, and interactive modifications.
- Supports both numeric and text-based filtering.
- Preview results in real-time.

### 9️⃣ Download Reports
- Export datasets at any stage (cleaned, transformed, or manually analyzed) in **CSV** or **Excel**.
- Full Excel reports include dataset and summary statistics.

---

## 🌈 Tech Stack

- **Python 3.10+**
- **Streamlit** – Web interface
- **Pandas & NumPy** – Data manipulation
- **Plotly** – Interactive visualizations
- **OpenPyXL** – Excel report generation
- **Python-dotenv** – Environment variable management

---

## ⚡ Installation & Setup

### 1️⃣ 🐙 Clone the repository:
```bash
git clone https://github.com/pagidalasaikiran/Auto-Data-Analyzer.git
cd Auto-Data-Analyzer
```

### 2️⃣ 🛠️ Create a virtual environment (optional but recommended):
```
python -m venv .venv
```
#Windows:
```
.venv\Scripts\activate
```
#Mac/Linux:
```
source .venv/bin/activate
```

### 3️⃣ 📦 Install dependencies:
```
pip install streamlit pandas numpy plotly openpyxl python-dotenv google-genai
```

### 4️⃣ 🔑 Add your Gemini API Key in a .env file:
```
GEMINI_API_KEY=your_api_key_here
```

### 5️⃣ 🚀 Run the Streamlit app:
```
streamlit run app.py
```
