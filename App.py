import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
        padding: 20px;
    }
    .stTitle {
        color: #1f4287;
        font-size: 3rem !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
        background: linear-gradient(120deg, #1f4287, #4070f4);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #4070f4;
        color: white;
        border-radius: 5px;
        padding: 10px 25px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1f4287;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .missing-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .recommendation {
        background-color: #e3f2fd;
        padding: 15px;
        border-left: 5px solid #4070f4;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Intelligent Data Preprocessing Suite")

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def get_ai_recommendation(column_name, dtype, missing_percentage):
    """AI recommendation for handling missing values"""
    if pd.api.types.is_numeric_dtype(dtype):
        if missing_percentage < 5:
            return "Mean imputation (Recommended: Low missing ratio, numeric data)"
        elif missing_percentage < 15:
            return "Median imputation (Recommended: Moderate missing ratio, handles outliers better)"
        else:
            return "KNN imputation (Recommended: High missing ratio, preserves relationships)"
    else:
        if missing_percentage < 10:
            return "Mode imputation (Recommended: Categorical data, low missing ratio)"
        else:
            return "Most frequent value or create 'Unknown' category (Recommended: High missing ratio in categorical data)"

if uploaded_file:
    # Initialize session state for processed dataframe if not exists
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = pd.read_csv(uploaded_file)
    
    df = st.session_state.processed_df
    
    # Missing Values Analysis
    st.markdown("<h2 style='color: #1f4287;'>Missing Values Analysis</h2>", unsafe_allow_html=True)
    
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_data,
        'Missing Percentage': missing_percentages
    })
    missing_info = missing_info[missing_info['Missing Values'] > 0]

    if not missing_info.empty:
        st.markdown("<div class='missing-card'>", unsafe_allow_html=True)
        st.write("### Columns with Missing Values:")
        for col in missing_info.index:
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>{col}</h4>
                <p>Missing Values: {missing_info.loc[col, 'Missing Values']}</p>
                <p>Missing Percentage: {missing_info.loc[col, 'Missing Percentage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Column Selection for Missing Value Treatment
        selected_columns = st.multiselect(
            "Select columns to handle missing values:",
            missing_info.index.tolist()
        )

        if selected_columns:
            for col in selected_columns:
                st.markdown(f"<h3 style='color: #1f4287;'>Handling {col}</h3>", unsafe_allow_html=True)
                
                # Store method selection in session state
                if f"method_{col}" not in st.session_state:
                    st.session_state[f"method_{col}"] = None

                # AI Recommendation
                recommendation = get_ai_recommendation(
                    col, 
                    df[col].dtype,
                    missing_info.loc[col, 'Missing Percentage']
                )
                st.markdown(f"""
                <div class='recommendation'>
                    <h4>AI Recommendation 🤖</h4>
                    <p>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

                # Method Selection
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    method = st.selectbox(
                        f"Choose method for {col}",
                        ["Mean", "Median", "Mode", "KNN"],
                        key=f"method_{col}"
                    )
                else:
                    method = st.selectbox(
                        f"Choose method for {col}",
                        ["Mode", "Create 'Unknown' category"],
                        key=f"method_{col}"
                    )

        # Process and Save Changes
        col1, col2 = st.columns([1, 1])
        process_button = col1.button("Process Missing Values", key="process")
        save_button = col2.button("Save Changes", key="save", disabled=not process_button)

        if process_button:
            temp_df = df.copy()
            for col in selected_columns:
                method = st.session_state[f"method_{col}"]
                if method == "Mean":
                    temp_df[col].fillna(temp_df[col].mean(), inplace=True)
                elif method == "Median":
                    temp_df[col].fillna(temp_df[col].median(), inplace=True)
                elif method == "Mode":
                    temp_df[col].fillna(temp_df[col].mode()[0], inplace=True)
                elif method == "Create 'Unknown' category":
                    temp_df[col].fillna("Unknown", inplace=True)
                elif method == "KNN":
                    st.warning("KNN imputation will be implemented in the next version")

            # Preview changes
            st.write("### Preview of Processed Data")
            st.write("Please review the changes and click 'Save Changes' to make them permanent.")
            st.write(temp_df.head())
            
            # Store temporary results
            st.session_state.temp_df = temp_df

        if save_button and 'temp_df' in st.session_state:
            # Save changes permanently
            st.session_state.processed_df = st.session_state.temp_df.copy()
            st.success("Changes saved successfully! You can now clear the selection and process other columns.")
            
            # Update missing info after saving
            missing_data = st.session_state.processed_df.isnull().sum()
            missing_percentages = (missing_data / len(st.session_state.processed_df)) * 100
            missing_info = pd.DataFrame({
                'Missing Values': missing_data,
                'Missing Percentage': missing_percentages
            })
            missing_info = missing_info[missing_info['Missing Values'] > 0]

    else:
        st.success("Your dataset has no missing values!")

    # Download Processed File
    st.download_button(
        "Download Processed Data",
        st.session_state.processed_df.to_csv(index=False),
        "processed_data.csv",
        "text/csv",
        key='download-csv'
    )

