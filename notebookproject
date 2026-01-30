digica/
│
├── app.py
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Analysis",
    layout="wide"
)

st.title("📦 Food Delivery Data Analysis")
st.caption("Converted from Jupyter Notebook to Streamlit App")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

train, test = load_data()

# -----------------------------
# DATA OVERVIEW
# -----------------------------
st.header("🔍 Dataset Overview")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Train Data")
    st.dataframe(train.head())

with col2:
    st.subheader("Test Data")
    st.dataframe(test.head())

st.write("Train shape:", train.shape)
st.write("Test shape:", test.shape)

# -----------------------------
# DATA CLEANING (RINGKASAN)
# -----------------------------
st.header("🧹 Data Cleaning")

train_clean = train.copy()
test_clean = test.copy()

# contoh handling missing value
missing = train_clean.isna().sum()
st.write("Missing Values:")
st.dataframe(missing[missing > 0])

# -----------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------
st.header("📊 Exploratory Data Analysis")

if "Delivery_person_Age" in train_clean.columns:
    fig, ax = plt.subplots()
    train_clean["Delivery_person_Age"].hist(bins=30, ax=ax)
    ax.set_title("Distribution of Delivery Person Age")
    st.pyplot(fig)

# -----------------------------
# FEATURE INSIGHT
# -----------------------------
st.header("📈 Business Insight")

st.markdown("""
Beberapa insight awal dari data:
- Distribusi umur driver mempengaruhi kecepatan delivery
- Variabel jarak dan kondisi cuaca berkontribusi terhadap ETA
- Data dapat dikembangkan untuk model prediksi waktu pengantaran
""")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Digica Project • Streamlit Deployment Ready")
streamlit
pandas
numpy
matplotlib
scikit-learn
