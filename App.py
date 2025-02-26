import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# Title
st.title("Automated Data Preprocessing")

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Handling Missing Values
    missing_strategy = st.selectbox("Select Missing Value Strategy", ["Mean", "Median", "Mode"])
    if st.button("Handle Missing Values"):
        imputer = SimpleImputer(strategy=missing_strategy.lower())
        df[df.select_dtypes(include=['number']).columns] = imputer.fit_transform(df.select_dtypes(include=['number']))
        st.write("### Processed Data", df.head())

    # One-Hot Encoding
    if st.button("Apply One-Hot Encoding"):
        enc = OneHotEncoder(sparse=False, drop='first')
        categorical_cols = df.select_dtypes(include=['object']).columns
        encoded_df = pd.DataFrame(enc.fit_transform(df[categorical_cols]))
        df = df.drop(columns=categorical_cols).join(encoded_df)
        st.write("### Encoded Data", df.head())

    # Feature Selection
    threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.01)
    if st.button("Apply Feature Selection"):
        selector = VarianceThreshold(threshold)
        df = df.loc[:, selector.fit(df).get_support()]
        st.write("### Selected Features", df.head())

    # Download Processed File
    st.download_button("Download CSV", df.to_csv(index=False), "processed_data.csv", "text/csv")

