import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Custom CSS - Reset and redefine all styles
st.markdown("""
<style>
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
        padding: 30px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title Styling */
    .stTitle {
        font-size: 3.2rem !important;
        text-align: center;
        padding: 25px 20px;
        background: linear-gradient(120deg, #1a237e, #1976d2);
        color: white;
        border-radius: 15px;
        margin-bottom: 35px;
        box-shadow: 0 10px 20px rgba(25, 118, 210, 0.2);
        position: relative;
        overflow: hidden;
    }

    /* Developer Banner */
    .developer-banner {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: glow 3s infinite alternate;
    }

    /* Missing Values Analysis */
    .missing-analysis-title {
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .missing-column-card {
        padding: 25px;
        margin: 15px 0;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .missing-column-card:hover {
        transform: translateY(-5px);
    }

    .column-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .column-name {
        color: #1976d2;
        font-size: 20px;
        margin: 0;
        font-weight: 600;
    }

    .missing-badge {
        background: linear-gradient(45deg, #1976d2, #2196f3);
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        font-size: 14px;
    }

    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        height: 10px;
        margin: 15px 0;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #1976d2, #2196f3);
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }

    .stats-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }

    .stats-label {
        color: #666;
        font-size: 14px;
    }

    .stats-value {
        color: #1976d2;
        font-weight: 600;
        font-size: 16px;
    }

    .dtype-badge {
        display: inline-block;
        padding: 5px 15px;
        background: #e3f2fd;
        border-radius: 15px;
        color: #1976d2;
        font-size: 14px;
        margin-top: 15px;
    }

    @keyframes glow {
        from {
            box-shadow: 0 0 10px #004e92, 0 0 20px #004e92;
        }
        to {
            box-shadow: 0 0 20px #004e92, 0 0 30px #004e92;
        }
    }

    /* File Upload Styling */
    .upload-container {
        position: relative;
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
        padding: 40px 30px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 20px 0;
        border: 2px dashed #1976d2;
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
        border-color: #2196f3;
    }

    .upload-content {
        margin-bottom: 25px;
    }

    .upload-icon {
        font-size: 40px;
        color: #1976d2;
        margin-bottom: 15px;
    }

    .upload-header {
        color: #1976d2;
        font-size: 1.5em;
        font-weight: 600;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .upload-text {
        color: #666;
        font-size: 1em;
        margin-bottom: 20px;
    }

    /* Custom Upload Button */
    .custom-upload-button {
        background: linear-gradient(45deg, #1976d2, #2196f3);
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.2);
        display: inline-block;
        margin-top: 10px;
    }

    .custom-upload-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.3);
    }

    /* Streamlit's default uploader modifications */
    .stFileUploader {
        padding-bottom: 1rem;
    }

    .stFileUploader > div {
        padding: 1rem;
    }

    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }

    /* Success message styling */
    .upload-success {
        margin-top: 15px;
        padding: 10px 20px;
        background: #4CAF50;
        color: white;
        border-radius: 10px;
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Missing Values Analysis Styling */
    .missing-value-title {
        text-align: center;
        color: #1976d2;
        margin: 20px 0;
        font-size: 24px;
        font-weight: 600;
    }

    .missing-value-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }

    .missing-value-card:hover {
        transform: translateY(-5px);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .column-name {
        color: #1976d2;
        font-size: 18px;
        font-weight: bold;
    }

    .missing-badge {
        background: #1976d2;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
    }

    .progress-bar-bg {
        background: #f0f0f0;
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }

    .progress-bar-fill {
        background: linear-gradient(90deg, #1976d2, #2196f3);
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }

    .stats-row {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }

    .percentage-label {
        color: #666;
    }

    .percentage-value {
        color: #1976d2;
        font-weight: bold;
    }

    .dtype-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
        margin-top: 10px;
    }

    /* Enhanced Button Styling */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 30px 0;
        padding: 10px;
    }

    .custom-button {
        background: linear-gradient(45deg, #1976d2, #2196f3);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.2);
        text-align: center;
        text-decoration: none;
        display: inline-block;
        min-width: 160px;
    }

    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.3);
    }

    .custom-button.process {
        background: linear-gradient(45deg, #1976d2, #2196f3);
    }

    .custom-button.save {
        background: linear-gradient(45deg, #43a047, #4caf50);
    }

    .custom-button.download {
        background: linear-gradient(45deg, #7b1fa2, #9c27b0);
    }

    .custom-button:disabled {
        background: #cccccc;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* Enhanced Button Styling */
    .stButton {
        display: inline-block;
    }

    .stButton > button {
        background: linear-gradient(45deg, #1976d2, #2196f3);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.2);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.3);
    }

    /* Process button */
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(45deg, #1976d2, #2196f3);
    }

    /* Save button */
    [data-testid="stButton"] > button:disabled {
        background: #cccccc;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #7b1fa2, #9c27b0);
    }

    /* Button container */
    [data-testid="column"] {
        padding: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# Developer Credit Banner
st.markdown("""
    <div class="developer-banner">
        <h2>Developed by Bilal Qaisar 👨‍💻</h2>
    </div>
""", unsafe_allow_html=True)

# Title with enhanced styling
st.markdown("""
    <div class="stTitle float-element">
        Data Preprocessing App
    </div>
""", unsafe_allow_html=True)

# Combined File Upload Section - Adjusted columns ratio
col1, col2, col3 = st.columns([0.5, 3, 0.5])  # Changed ratio to make middle column wider
with col2:  # Using the wider middle column
    upload_container = st.container()
    
    with upload_container:
        st.markdown("""
            <div class="upload-container" style="width: 100%; max-width: 800px; margin: 20px auto;">
                <div class="upload-content">
                    <div class="upload-icon">📁</div>
                    <div class="upload-header">Upload Your Dataset</div>
                    <div class="upload-text">Select your CSV file to begin analysis</div>
                </div>
        """, unsafe_allow_html=True)
        
        # File Upload with custom styling
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], 
                                       help="Upload your CSV file here")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.markdown("""
                <div class="upload-success">
                    ✅ File successfully uploaded!
                </div>
            """, unsafe_allow_html=True)

def local_css():
    st.markdown("""
    <style>
        .missing-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        
        .missing-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .column-title {
            color: #1976d2;
            font-size: 18px;
            font-weight: bold;
        }
        
        .missing-count {
            background: #1976d2;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
        }
        
        .progress-outer {
            background: #f0f0f0;
            height: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .progress-inner {
            background: linear-gradient(90deg, #1976d2, #2196f3);
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease-in-out;
        }
        
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        
        .dtype-tag {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS function at the start
local_css()

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
    
    # Add Data Overview Section
    st.markdown("<h2 style='text-align: center; color: #1976d2; margin: 20px 0;'>Data Overview 📊</h2>", unsafe_allow_html=True)
    
    # Data Shape
    st.write(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Data Info
    st.write("**Data Types and Non-Null Counts:**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # Quick Statistics
    st.write("**Quick Statistics:**")
    st.write(df.describe())
    
    # Correlation Matrix
    st.markdown("<h3 style='color: #1976d2;'>Correlation Matrix 📈</h3>", unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].corr(),
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig)
    
    # Data Distribution
    st.markdown("<h3 style='color: #1976d2;'>Data Distribution 📊</h3>", unsafe_allow_html=True)
    selected_column = st.selectbox("Select column for distribution plot:", df.columns)
    
    if df[selected_column].dtype in ['int64', 'float64']:
        fig = px.histogram(df, x=selected_column, nbins=30)
        st.plotly_chart(fig)
    else:
        fig = px.bar(df[selected_column].value_counts())
        st.plotly_chart(fig)
    
    # Outlier Detection
    st.markdown("<h3 style='color: #1976d2;'>Outlier Detection 🔍</h3>", unsafe_allow_html=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        selected_column_outlier = st.selectbox("Select column for outlier detection:", numeric_columns)
        fig = px.box(df, y=selected_column_outlier)
        st.plotly_chart(fig)
        
        # Calculate and display outlier statistics
        Q1 = df[selected_column_outlier].quantile(0.25)
        Q3 = df[selected_column_outlier].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[selected_column_outlier] < (Q1 - 1.5 * IQR)) | 
                     (df[selected_column_outlier] > (Q3 + 1.5 * IQR))][selected_column_outlier]
        
        st.write(f"Number of outliers detected: {len(outliers)}")
        if len(outliers) > 0:
            st.write("Outlier values:", outliers.values)
    
    # Data Cleaning Options
    st.markdown("<h3 style='color: #1976d2;'>Data Cleaning Options 🧹</h3>", unsafe_allow_html=True)
    
    # Duplicate Rows
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        if st.button("Remove Duplicate Rows"):
            df = df.drop_duplicates()
            st.session_state.processed_df = df
            st.success("Duplicate rows removed successfully!")
    
    # Handle Outliers
    if len(numeric_columns) > 0:
        st.write("**Handle Outliers**")
        outlier_column = st.selectbox("Select column for outlier treatment:", numeric_columns)
        outlier_method = st.selectbox("Select outlier handling method:", 
                                    ["None", "Remove", "Cap", "Replace with Mean"])
        
        if outlier_method != "None":
            Q1 = df[outlier_column].quantile(0.25)
            Q3 = df[outlier_column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[outlier_column] < (Q1 - 1.5 * IQR)) | (df[outlier_column] > (Q3 + 1.5 * IQR))
            
            if outlier_method == "Remove":
                df = df[~outlier_mask]
            elif outlier_method == "Cap":
                df.loc[df[outlier_column] < (Q1 - 1.5 * IQR), outlier_column] = Q1 - 1.5 * IQR
                df.loc[df[outlier_column] > (Q3 + 1.5 * IQR), outlier_column] = Q3 + 1.5 * IQR
            elif outlier_method == "Replace with Mean":
                df.loc[outlier_mask, outlier_column] = df[outlier_column].mean()
            
            st.session_state.processed_df = df
            st.success(f"Outliers in {outlier_column} handled using {outlier_method} method!")

    # Enhanced Stats Dashboard with Advanced Cards
    stats_html = f"""
        <style>
            .stats-dashboard {{
                background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
                padding: 25px;
                border-radius: 20px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
                margin: 20px 0;
                border: 1px solid rgba(25, 118, 210, 0.1);
            }}
            
            .stats-title {{
                text-align: center;
                color: #1976d2;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 25px;
                text-transform: uppercase;
                letter-spacing: 2px;
                background: linear-gradient(45deg, #1976d2, #2196f3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                padding: 10px;
            }}
            
            .stats-card {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid rgba(25, 118, 210, 0.1);
            }}
            
            .stats-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            }}
            
            .stats-label {{
                color: #666;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .stats-value {{
                color: #1976d2;
                font-size: 28px;
                font-weight: 600;
                margin: 0;
                background: linear-gradient(45deg, #1976d2, #2196f3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            .stats-icon {{
                font-size: 24px;
                margin-bottom: 10px;
                color: #1976d2;
            }}
        </style>
        
        <div class="stats-dashboard">
            <div class="stats-title">
                📊 Dataset Statistics
            </div>
            <div class="stats-grid">
                <div class="stats-card">
                    <div class="stats-icon">👥</div>
                    <div class="stats-label">Total Rows</div>
                    <div class="stats-value">{len(df)}</div>
                </div>
                <div class="stats-card">
                    <div class="stats-icon">📊</div>
                    <div class="stats-label">Total Columns</div>
                    <div class="stats-value">{len(df.columns)}</div>
                </div>
            </div>
        </div>
    """
    
    st.markdown(stats_html, unsafe_allow_html=True)
    
    # Track processed columns in session state
    if 'processed_columns' not in st.session_state:
        st.session_state.processed_columns = set()

    # Missing Values Analysis
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_data,
        'Missing Percentage': missing_percentages
    })
    missing_info = missing_info[missing_info['Missing Values'] > 0]

    if not missing_info.empty:
        st.markdown("<h2 style='text-align: center; color: #1976d2; margin: 20px 0;'>Missing Values Analysis 🔍</h2>", unsafe_allow_html=True)

        # Show only columns that haven't been processed
        remaining_columns = [col for col in missing_info.index if col not in st.session_state.processed_columns]
        
        for col in remaining_columns:
            missing_percent = missing_info.loc[col, 'Missing Percentage']
            missing_count = missing_info.loc[col, 'Missing Values']
            
            # Display column information
            st.write(f"### {col}")
            st.write(f"Missing Values: {missing_count}")
            st.write(f"Missing Percentage: {missing_percent:.1f}%")
            st.write(f"Data Type: {df[col].dtype}")
            st.progress(missing_percent/100)
            st.markdown("---")

        # Column Selection for Missing Value Treatment
        selected_columns = st.multiselect(
            "Select columns to handle missing values:",
            remaining_columns
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

        # Button layout
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            process_button = st.button("Process Missing Values", key="process_button", type="primary")
        with col2:
            save_button = st.button("Save Changes", key="save_button", disabled=not process_button)
        with col3:
            # Always show download button
            st.download_button(
                "Download Dataset",
                df.to_csv(index=False),
                "processed_data.csv",
                "text/csv",
                key='download-csv'
            )

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
            st.write(temp_df.head())
            
            # Store temporary results
            st.session_state.temp_df = temp_df

        if save_button and 'temp_df' in st.session_state:
            # Save changes permanently
            st.session_state.processed_df = st.session_state.temp_df.copy()
            # Add processed columns to the set
            st.session_state.processed_columns.update(selected_columns)
            st.success("Changes saved successfully! You can now process other columns or download the dataset.")
            
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

# Add particles background
st.markdown("""
<div id="tsparticles"></div>
<script src="https://cdn.jsdelivr.net/npm/tsparticles@1.37.5/dist/tsparticles.min.js"></script>
<script>
    tsParticles.load("tsparticles", {
        particles: {
            number: { value: 80 },
            color: { value: "#1976d2" },
            shape: { type: "circle" },
            opacity: { value: 0.5 },
            size: { value: 3 },
            move: {
                enable: true,
                speed: 2,
                direction: "none",
                random: false,
                straight: false,
                outModes: { default: "bounce" }
            },
            links: {
                enable: true,
                distance: 150,
                color: "#1976d2",
                opacity: 0.4,
                width: 1
            }
        }
    });
</script>
""", unsafe_allow_html=True)

# Enhanced Developer Banner with Glassmorphism
st.markdown("""
    <div class="developer-banner glass-card animated-border">
        <h2 class="neon-text">Developed by Bilal Qaisar 👨‍💻</h2>
    </div>
""", unsafe_allow_html=True)

