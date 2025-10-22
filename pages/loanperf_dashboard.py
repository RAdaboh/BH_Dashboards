import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Loan Performance Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
    .stMetric label {
        font-size: 14px;
        color: #666;
    }
    .stMetric div {
        font-size: 24px;
        font-weight: bold;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        st.info("🔄 Loading data from Google Drive...")
        
        # Try multiple methods to load the data
        methods = [
            # Method 1: Direct pandas read
            lambda: pd.read_csv('https://drive.google.com/uc?id=1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo'),
            lambda: pd.read_csv('https://drive.google.com/uc?id=1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6'),
            
            # Method 2: Using requests to get raw content
            lambda: load_via_requests('1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo'),
            lambda: load_via_requests('1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6'),
            
            # Method 3: Alternative URL format
            lambda: pd.read_csv(f'https://drive.google.com/uc?export=download&id=1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo'),
            lambda: pd.read_csv(f'https://drive.google.com/uc?export=download&id=1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6'),
        ]
        
        # Try loading disbursement data
        disbursement_df = None
        for i, method in enumerate(methods[::2]):  # Try every other method for disbursement
            try:
                st.info(f"🔄 Trying method {i+1} for disbursement data...")
                disbursement_df = method()
                if disbursement_df is not None and len(disbursement_df) > 0:
                    st.success(f"✅ Disbursement data loaded! Shape: {disbursement_df.shape}")
                    break
            except Exception as e:
                st.warning(f"⚠️ Method {i+1} failed: {e}")
                continue
        
        # Try loading repayment data
        repayment_df = None
        for i, method in enumerate(methods[1::2]):  # Try every other method for repayment
            try:
                st.info(f"🔄 Trying method {i+1} for repayment data...")
                repayment_df = method()
                if repayment_df is not None and len(repayment_df) > 0:
                    st.success(f"✅ Repayment data loaded! Shape: {repayment_df.shape}")
                    break
            except Exception as e:
                st.warning(f"⚠️ Method {i+1} failed: {e}")
                continue
        
        if disbursement_df is None or repayment_df is None:
            st.error("❌ Could not load data from any method")
            return None, None
        
        # Clean column names
        disbursement_df.columns = disbursement_df.columns.str.replace(' ', '_').str.lower()
        repayment_df.columns = repayment_df.columns.str.replace(' ', '_').str.lower()
        
        return disbursement_df, repayment_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None

def load_via_requests(file_id):
    """Load CSV via requests to get raw content"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                content = response.content.decode(encoding)
                df = pd.read_csv(StringIO(content))
                if len(df) > 0:
                    return df
            except:
                continue
        
        # If all encodings fail, try direct read
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.warning(f"Requests method failed: {e}")
        return None

# Load data with detailed progress
st.info("🔄 Starting data loading process...")
loan_disbursement_df, loan_repayment_df = load_data()

# Check if data loaded successfully
if loan_disbursement_df is None or loan_repayment_df is None:
    st.error("""
    ❌ Failed to load data. Possible reasons:
    
    1. **File not publicly accessible**: Make sure your Google Drive files are set to "Anyone with the link can view"
    2. **Incorrect file IDs**: Verify the file IDs in the URLs
    3. **Empty files**: Check that your CSV files contain actual data
    4. **File format issues**: Ensure files are valid CSV format
    
    **To fix this:**
    - Go to your Google Drive file
    - Click "Share" → "Change to anyone with the link" → "Viewer"
    - Copy the file ID from the URL (the part between '/d/' and '/view')
    - Verify the file has data by opening it locally
    """)
    st.stop()

# Display detailed data information
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Info")

if len(loan_disbursement_df) == 0:
    st.error("❌ Disbursement data is empty (0 rows)")
else:
    st.sidebar.write(f"**Disbursement Data:**")
    st.sidebar.write(f"- Rows: {len(loan_disbursement_df):,}")
    st.sidebar.write(f"- Columns: {len(loan_disbursement_df.columns)}")
    st.sidebar.write(f"- Memory: {loan_disbursement_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if len(loan_repayment_df) == 0:
    st.error("❌ Repayment data is empty (0 rows)")
else:
    st.sidebar.write(f"**Repayment Data:**")
    st.sidebar.write(f"- Rows: {len(loan_repayment_df):,}")
    st.sidebar.write(f"- Columns: {len(loan_repayment_df.columns)}")
    st.sidebar.write(f"- Memory: {loan_repayment_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Show actual data preview
with st.expander("🔍 Detailed Data Preview"):
    st.subheader("Disbursement Data")
    st.write(f"Shape: {loan_disbursement_df.shape}")
    st.write("Columns:", list(loan_disbursement_df.columns))
    st.dataframe(loan_disbursement_df.head(10))
    
    st.subheader("Repayment Data")
    st.write(f"Shape: {loan_repayment_df.shape}")
    st.write("Columns:", list(loan_repayment_df.columns))
    st.dataframe(loan_repayment_df.head(10))

# If data is empty, stop execution
if len(loan_disbursement_df) == 0 or len(loan_repayment_df) == 0:
    st.error("""
    ❌ One or both datasets are empty. 
    
    Please check:
    1. Your Google Drive files actually contain data
    2. The files are proper CSV format
    3. The file IDs are correct
    
    **Current file IDs being used:**
    - Disbursement: 1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo
    - Repayment: 1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6
    """)
    st.stop()

# Continue with the rest of your dashboard code...
# [The rest of your dashboard code from the previous version goes here]

# Helper function to find column names
def find_column(df, possible_names):
    """Find a column in dataframe from list of possible names"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

# Find key columns in disbursement data
disbursement_amount_col = find_column(loan_disbursement_df, 
                                    ['disbursement_amount', 'amount', 'loan_amount', 'disbursed_amount', 'principal'])
region_col_disb = find_column(loan_disbursement_df, 
                            ['region', 'region_x', 'location', 'area', 'branch_region'])
product_col = find_column(loan_disbursement_df, 
                        ['product_id', 'product', 'product_type', 'loan_product'])
branch_col = find_column(loan_disbursement_df, 
                       ['branch', 'branch_name', 'branch_code'])
year_col_disb = find_column(loan_disbursement_df, 
                          ['disbursement_year', 'year', 'disbursement_date', 'date'])

# Find key columns in repayment data
repayment_amount_col = find_column(loan_repayment_df, 
                                 ['amount_paid', 'repayment_amount', 'amount', 'paid_amount', 'payment_amount'])
region_col_repay = find_column(loan_repayment_df, 
                             ['region', 'region_x', 'location', 'area', 'branch_region'])
customer_id_col = find_column(loan_repayment_df, 
                            ['customer_id', 'customer', 'client_id'])
loan_number_col = find_column(loan_repayment_df, 
                            ['loan_number', 'loan_id', 'loan_no'])
outstanding_col = find_column(loan_repayment_df, 
                            ['outstanding_balance', 'outstanding', 'balance', 'remaining_balance'])
total_repayment_col = find_column(loan_repayment_df, 
                                ['total_repayment', 'total_due', 'expected_repayment'])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Disbursement Analysis", "Repayment Analysis", "Performance Metrics"])

# Main content
st.title("Loan Performance Analysis Dashboard")

# [Rest of your dashboard pages code...]
# Continue with the same page structure as before

if page == "Overview":
    st.header("Overview")
    
    if disbursement_amount_col:
        total_disbursed = loan_disbursement_df[disbursement_amount_col].sum()
        total_loans = len(loan_disbursement_df)
        avg_disbursement = loan_disbursement_df[disbursement_amount_col].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Loans Disbursed", f"GHC {total_disbursed:,.2f}")
        col2.metric("Number of Loans", f"{total_loans:,}")
        col3.metric("Average Disbursement", f"GHC {avg_disbursement:,.2f}")
    else:
        st.warning("Disbursement amount column not found.")
    
    st.write("---")
    
    st.subheader("Data Preview")
    tab1, tab2 = st.tabs(["Disbursement Data", "Repayment Data"])
    
    with tab1:
        st.dataframe(loan_disbursement_df.head(10))
    with tab2:
        st.dataframe(loan_repayment_df.head(10))

# Add other page implementations here...
# [Implement Disbursement Analysis, Repayment Analysis, Performance Metrics pages]

st.info("✅ Dashboard loaded successfully!")