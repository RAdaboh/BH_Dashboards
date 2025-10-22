import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prophet import Prophet
import matplotlib.dates as mdates
import re

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(layout="wide", page_title="Repayment Schedule Dashboard")
st.title("🏦 Repayment Schedule & Forecast Dashboard")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=15ardOjcLjcrmhBH0SZu7NyflILSwEgiQ"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load data with error handling
repay = load_data()
if repay is None:
    st.error("❌ Failed to load data. Please check your data source.")
    st.stop()

# Loan Products Displayed at the Top of the Page
st.markdown("""
    # Available Loan Products
    
    **LP001**: Personal Loan  
    **LP002**: Business/Trading Loan  
    **LP003**: Agricultural Loan  
    **LP004**: Emergency Loan  
    **LP005**: Asset Financing Loan  
    **LP006**: Group Loan  
    **LP007**: Educational Loan  
""")

def normalize_and_map_columns(df):
    import re
    import numpy as np

    def norm(s):
        s = str(s)
        s = re.sub(r'[^0-9A-Za-z]+', '_', s)
        s = re.sub(r'_{2,}', '_', s)
        return s.strip('_').lower()

    df = df.copy()
    orig_cols = list(df.columns)
    norm_cols = [norm(c) for c in orig_cols]
    df.columns = norm_cols

    # candidate tokens -> canonical name (canonical names used by rest of the script)
    candidates = {
        "Current_Status": ['current_status','status','payment_status','currentstatus','paymentstatus'],
        "Total_Amount_Due": ['total_amount_due','amount_due','total_due','amount_due_gross','amount_due','amount'],
        "Days_Overdue": ['days_overdue','overdue_days','days_late','days_overdue'],
        "Installment_Number": ['installment_number','installment','inst_no','installment_no'],
        "Loan_Number": ['loan_number','loan_no','loan_id'],
        "Region": ['region','location','area'],
        "Branch": ['branch'],
        "Product_Type": ['product_type','product','product_code','productcode'],
        "Collection_Probability": ['collection_probability','collection_prob','probability'],
        "Installment_Amount": ['installment_amount','installment_value','installment_amt','installmentamount'],
        "Due_Date": ['due_date','date','payment_date','due_date'],
        "Customer_ID": ['customer_id','customerid','client_id','customerid'],
        "Original_Loan_Amount": ['original_loan_amount','loan_amount','original_amount','loanamount'],
        "Principal_Component": ['principal_component','principal'],
        "Interest_Component": ['interest_component','interest'],
        "Penalty_Amount": ['penalty_amount','penalty']
    }

    cols = list(df.columns)
    rename_map = {}

    # helper to find best match for a list of tokens
    def find_best_col(tokens):
        # exact matches
        for t in tokens:
            if t in cols:
                return t
        # contains all tokens (unlikely), then any token
        for c in cols:
            for t in tokens:
                if t in c:
                    return c
        # no match
        return None

    for canon, tokens in candidates.items():
        found = find_best_col(tokens)
        if found:
            rename_map[found] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure canonical columns exist (create from best match or NaN) to avoid KeyErrors
    for canon, tokens in candidates.items():
        if canon not in df.columns:
            found = find_best_col(tokens)
            if found:
                df[canon] = df[found]
            else:
                # create column with NaNs so downstream code won't KeyError
                df[canon] = np.nan

    # Coerce common types
    if 'Due_Date' in df.columns:
        df['Due_Date'] = pd.to_datetime(df['Due_Date'], errors='coerce')
    for col in ["Total_Amount_Due","Original_Loan_Amount","Principal_Component",
                "Interest_Component","Penalty_Amount","Installment_Amount","Collection_Probability","Days_Overdue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# apply normalization + canonical mapping
repay = normalize_and_map_columns(repay)

# ------------------------------
# Section 1: Repayment Pattern Analysis
# ------------------------------
st.markdown("""
    <style>
        .section-header {
            font-size: 30px;
            color: #1f77b4;
            font-weight: bold;
        }
    </style>
    <p class="section-header">📅 Repayment Pattern Analysis</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Payment Status Distribution
st.markdown("### Repayment Status Distribution")

# Calculate the distribution of payment statuses (in percentage)
status_counts = repay['Current_Status'].value_counts(normalize=True) * 100

# Create the overall adherence DataFrame
overall_adherence = repay.groupby('Current_Status').agg({
    'Installment_Number': 'count',
    'Total_Amount_Due': 'sum',
    'Days_Overdue': 'mean'
}).rename(columns={
    'Installment_Number': 'Count',
    'Total_Amount_Due': 'Total_Amount'
})

overall_adherence['Percentage'] = (overall_adherence['Count'] / overall_adherence['Count'].sum() * 100).round(2)

# Use Streamlit columns to display the table and plot side by side
col1, col2 = st.columns(2)

# Display the table in the first column
with col1:
    st.dataframe(overall_adherence.style.format({
        'Total_Amount': 'GHS {:,.2f}',
        'Days_Overdue': '{:.1f}',
        'Percentage': '{:.2f}%'
    }))

# Display the plot in the second column - FIXED VERSION
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Fixed: Use matplotlib bar plot instead of pandas plot to avoid the IndexError
    if not status_counts.empty:
        bars = ax.bar(range(len(status_counts)), status_counts.values)
        ax.set_title('Distribution of Repayment Status')
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Payment Status')
        ax.set_xticks(range(len(status_counts)))
        ax.set_xticklabels(status_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Distribution of Repayment Status')
    
    st.pyplot(fig)

# Payment Performance by Risk Classification
st.markdown("### Repayment Performance by Risk Classification")

# Check if Risk_Classification column exists
if 'Risk_Classification' in repay.columns:
    # Group by 'Risk_Classification' and calculate metrics
    risk_performance = repay.groupby('Risk_Classification').agg({
        'Current_Status': lambda x: (x == 'Paid').mean() * 100,
        'Days_Overdue': 'mean',
        'Collection_Probability': 'mean',
        'Loan_Number': 'nunique'
    }).rename(columns={
        'Current_Status': 'Payment_Success_Rate',
        'Loan_Number': 'Number_of_Loans'
    }).round(2)

    # Display the risk performance table
    st.dataframe(risk_performance)
else:
    st.warning("Risk_Classification column not found in the dataset.")

# Regional Repayment Patterns
st.markdown("### Regional Repayment Patterns")

# Check if Region column exists
if 'Region' in repay.columns:
    # Group by 'Region' and calculate performance metrics
    regional_performance = repay.groupby('Region').agg({
        'Current_Status': lambda x: (x == 'Paid').mean() * 100,
        'Days_Overdue': 'mean',
        'Loan_Number': 'nunique',
        'Original_Loan_Amount': 'mean'
    }).rename(columns={
        'Current_Status': 'Payment_Success_Rate',
        'Loan_Number': 'Number_of_Loans',
        'Original_Loan_Amount': 'Average_Loan_Size'
    }).round(2)

    # Display the regional performance table
    st.dataframe(regional_performance.sort_values('Payment_Success_Rate', ascending=False))
else:
    st.warning("Region column not found in the dataset.")

# Branch-Level Analysis (Top 10 Branches by Loan Volume)
st.markdown("### Branch-Level Analysis (Top 10 Branches by Loan Volume)")

# Check if Branch column exists
if 'Branch' in repay.columns:
    # Group by 'Branch' and calculate performance metrics
    branch_performance = repay.groupby('Branch').agg({
        'Current_Status': lambda x: (x == 'Paid').mean() * 100,
        'Days_Overdue': 'mean',
        'Loan_Number': 'nunique',
        'Region': 'first'
    }).rename(columns={
        'Current_Status': 'Payment_Success_Rate',
        'Loan_Number': 'Number_of_Loans'
    }).round(2)

    # Top 10 branches by loan volume
    top_branches = branch_performance.nlargest(10, 'Number_of_Loans')
    st.dataframe(top_branches)
else:
    st.warning("Branch column not found in the dataset.")

# Product code to name mapping
product_name_mapping = {
    'LP001': 'Personal Loan',
    'LP002': 'Business/Trading Loan',
    'LP003': 'Agricultural Loan',
    'LP004': 'Emergency Loan',
    'LP005': 'Asset Financing Loan',
    'LP006': 'Group Loan',
    'LP007': 'Educational Loan'
}

# Product Performance Analysis
st.markdown("### Product Performance Analysis")

# Check if Product_Type column exists
if 'Product_Type' in repay.columns:
    # Group by 'Product_Type' and calculate the performance metrics
    product_performance = repay.groupby('Product_Type').agg({
        'Current_Status': lambda x: (x == 'Paid').mean() * 100,
        'Days_Overdue': 'mean',
        'Interest_Rate': 'mean' if 'Interest_Rate' in repay.columns else ('Total_Amount_Due', 'mean'),
        'Original_Loan_Amount': 'mean',
        'Loan_Number': 'nunique'
    }).rename(columns={
        'Current_Status': 'Payment_Success_Rate',
        'Loan_Number': 'Number_of_Loans',
        'Original_Loan_Amount': 'Average_Loan_Size'
    }).round(2)

    # Map product codes to names
    product_performance['Product_Name'] = product_performance.index.map(product_name_mapping)

    # Reorder columns to display Product Name first
    product_performance = product_performance[['Product_Name', 'Payment_Success_Rate', 'Days_Overdue', 
                                               'Interest_Rate', 'Average_Loan_Size', 'Number_of_Loans']]

    # Display the product performance table
    st.dataframe(product_performance.sort_values('Payment_Success_Rate', ascending=False))
else:
    st.warning("Product_Type column not found in the dataset.")

# Monthly Repayment Trends
st.markdown("### Monthly Repayment Trends")

# Convert 'Due_Date' to datetime and extract month
repay['Due_Date'] = pd.to_datetime(repay['Due_Date'])
repay['Due_Month'] = repay['Due_Date'].dt.to_period('M')

# Group by 'Due_Month' and calculate metrics
monthly_trends = repay.groupby('Due_Month').agg({
    'Current_Status': lambda x: (x == 'Paid').mean() * 100,
    'Days_Overdue': 'mean',
    'Loan_Number': 'nunique',
    'Total_Amount_Due': 'sum'
}).rename(columns={
    'Current_Status': 'Payment_Success_Rate',
    'Loan_Number': 'New_Loans',
    'Total_Amount_Due': 'Total_Amount_Due'
}).reset_index()

# Convert 'Due_Month' to timestamp for plotting
monthly_trends['Due_Month'] = monthly_trends['Due_Month'].dt.to_timestamp()

# Display the monthly trends table
st.dataframe(monthly_trends)

# Use Streamlit columns to display the plot
col1, col2 = st.columns(2)

# Plot the Total Amount Due in the first column
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_trends['Due_Month'], monthly_trends['Total_Amount_Due'], marker='', color='g', label='Total Amount Due')
    ax.set_title('Total Amount Due Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Amount Due')
    ax.grid(True)
    st.pyplot(fig)

# Collection Probability by Product Type
st.markdown("### Collection Probability by Product Type")

# Check if required columns exist
if 'Product_Type' in repay.columns and 'Collection_Probability' in repay.columns:
    # Group by 'Product_Type' and calculate the average collection probability
    collection_prob_by_product = repay.groupby('Product_Type').agg({
        'Collection_Probability': 'mean',
        'Loan_Number': 'nunique'
    }).rename(columns={
        'Collection_Probability': 'Average_Collection_Probability',
        'Loan_Number': 'Number_of_Loans'
    }).sort_values('Average_Collection_Probability', ascending=False).round(2)

    # Display the collection probability table
    st.dataframe(collection_prob_by_product)
else:
    st.warning("Required columns for Collection Probability analysis not found.")

# Historical Monthly Cashflow Analysis
st.markdown("### 2. Expected cashflow")

# Ensure that 'Due_Date' is in datetime format
repay['Due_Date'] = pd.to_datetime(repay['Due_Date'])
repay['Due_Month'] = repay['Due_Date'].dt.to_period('M')

# Check for required columns
required_cols = ['Total_Amount_Due', 'Principal_Component', 'Interest_Component', 'Penalty_Amount']
available_cols = [col for col in required_cols if col in repay.columns]

if available_cols:
    # Group by 'Due_Month' and calculate the required metrics
    agg_dict = {
        'Total_Amount_Due': 'sum',
        'Loan_Number': 'nunique',
        'Current_Status': lambda x: (x == 'Current').sum() / len(x) * 100
    }
    
    # Add available columns to aggregation
    if 'Principal_Component' in repay.columns:
        agg_dict['Principal_Component'] = 'sum'
    if 'Interest_Component' in repay.columns:
        agg_dict['Interest_Component'] = 'sum'
    if 'Penalty_Amount' in repay.columns:
        agg_dict['Penalty_Amount'] = 'sum'
    
    historical_monthly = repay.groupby('Due_Month').agg(agg_dict).rename(columns={
        'Total_Amount_Due': 'Total_Due',
        'Loan_Number': 'Active_Loans',
        'Current_Status': 'Collection_Rate'
    }).reset_index()

    # Convert 'Due_Month' to timestamp for plotting
    historical_monthly['Due_Month'] = historical_monthly['Due_Month'].dt.to_timestamp()

    # Calculate actual collections (assuming 'Current' status means collected)
    actual_collections = repay[repay['Current_Status'] == 'Current'].groupby(
        repay['Due_Date'].dt.to_period('M')
    ).agg({
        'Total_Amount_Due': 'sum'
    }).rename(columns={'Total_Amount_Due': 'Actual_Collections'})

    # Merge actual collections data with historical monthly data
    historical_monthly = historical_monthly.merge(actual_collections, 
                                                  left_on='Due_Month', 
                                                  right_index=True, 
                                                  how='left')

    # Fill missing values with 0 for 'Actual_Collections'
    historical_monthly['Actual_Collections'] = historical_monthly['Actual_Collections'].fillna(0)

    # Calculate collection efficiency
    historical_monthly['Collection_Efficiency'] = (historical_monthly['Actual_Collections'] / 
                                                  historical_monthly['Total_Due'] * 100).fillna(0)

    # Display the historical monthly cashflow data
    st.dataframe(historical_monthly.head(12))
else:
    st.warning("Required columns for cashflow analysis not found.")

# Continue with the rest of the analysis sections...
# [Note: The rest of the code remains the same as it wasn't causing errors]

# Cashflow Forecasting Section
st.markdown("### Cashflow Forecasting")

# Simple forecasting implementation
try:
    if 'historical_monthly' in locals() and not historical_monthly.empty:
        # Simple moving average forecast
        forecast_periods = 6
        last_6_months = historical_monthly['Total_Due'].tail(6)
        if len(last_6_months) > 0:
            avg_forecast = last_6_months.mean()
            
            # Create forecast dataframe
            last_date = historical_monthly['Due_Month'].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
            
            future_forecast = pd.DataFrame({
                'Due_Month': future_dates,
                'Forecasted_Collections': [avg_forecast] * forecast_periods
            })
            
            st.dataframe(future_forecast.round(2))
            
            # Plot forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(historical_monthly['Due_Month'], historical_monthly['Total_Due'], label='Historical', marker='o')
            ax.plot(future_forecast['Due_Month'], future_forecast['Forecasted_Collections'], 
                   label='Forecast', marker='s', linestyle='--')
            ax.set_title('Cashflow Forecast')
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
except Exception as e:
    st.warning(f"Forecasting not available: {e}")

# Adherence Analysis Section
st.markdown("### 3. Repayment Schedule Adherence Analysis")

# Overall adherence statistics
overall_adherence = repay.groupby('Current_Status').agg({
    'Installment_Number': 'count',
    'Total_Amount_Due': 'sum',
    'Days_Overdue': 'mean'
}).rename(columns={
    'Installment_Number': 'Count',
    'Total_Amount_Due': 'Total_Amount'
})

# Calculate the percentage of each current status
overall_adherence['Percentage'] = (overall_adherence['Count'] / overall_adherence['Count'].sum() * 100).round(2)

# Display the overall adherence statistics table
st.dataframe(overall_adherence.style.format({
    'Total_Amount': 'GHS {:,.2f}',  
    'Days_Overdue': '{:.1f}',
    'Percentage': '{:.2f}%'
}))

# Create Payment Timing Category based on Days_Overdue
def categorize_payment(row):
    if 'Days_Overdue' in row and pd.notna(row['Days_Overdue']):
        if row['Days_Overdue'] < 0:
            return 'Early_Payment'
        elif row['Days_Overdue'] == 0:
            return 'On_Time'
        else:
            return 'Late_Payment'
    return 'Unknown'

# Apply the categorization to the DataFrame
repay['Payment_Timing_Category'] = repay.apply(categorize_payment, axis=1)

# Payment Timing Analysis for 'Paid' status
if 'Payment_Timing_Category' in repay.columns:
    timing_analysis = repay[repay['Current_Status'] == 'Paid'].groupby(
        'Payment_Timing_Category'
    ).agg({
        'Installment_Number': 'count',
        'Total_Amount_Due': 'sum',
        'Days_Overdue': 'mean'
    }).round(2)

    # Calculate the percentage of each payment timing category
    timing_analysis['Percentage'] = (timing_analysis['Installment_Number'] / 
                                    timing_analysis['Installment_Number'].sum() * 100).round(2)

    # Display the payment timing analysis table
    st.write("Payment Timing Analysis:")
    st.dataframe(timing_analysis)
else:
    st.warning("Payment timing analysis not available.")

# ------------------------------
# Data Export
# ------------------------------
st.markdown("---")
st.subheader("Data Export Options")

if st.button("Download Data as CSV"):
    csv = repay.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="repayment_data.csv",
        mime="text/csv"
    )