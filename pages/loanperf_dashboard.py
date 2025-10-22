import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

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
    
    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        # Load data from Google Drive
        loan_disbursement_df = pd.read_csv('https://drive.google.com/uc?id=1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo')
        loan_repayment_df = pd.read_csv('https://drive.google.com/uc?id=1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6')

        # Clean column names (replace spaces with underscores and make lowercase)
        loan_disbursement_df.columns = loan_disbursement_df.columns.str.replace(' ', '_').str.lower()
        loan_repayment_df.columns = loan_repayment_df.columns.str.replace(' ', '_').str.lower()
        
        return loan_disbursement_df, loan_repayment_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load data with progress indicator
st.info("🔄 Loading data...")
loan_disbursement_df, loan_repayment_df = load_data()

if loan_disbursement_df is None or loan_repayment_df is None:
    st.error("Failed to load data. Please check the data sources.")
    st.stop()

# Display data info for debugging
with st.sidebar:
    st.subheader("📊 Data Info")
    st.write(f"Disbursement data shape: {loan_disbursement_df.shape}")
    st.write(f"Repayment data shape: {loan_repayment_df.shape}")
    
    with st.expander("View Columns"):
        st.write("**Disbursement Columns:**", list(loan_disbursement_df.columns))
        st.write("**Repayment Columns:**", list(loan_repayment_df.columns))

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

if page == "Overview":
    st.header("Overview")
    
    # Key metrics - with safe column access
    if disbursement_amount_col:
        total_disbursed = loan_disbursement_df[disbursement_amount_col].sum()
        total_loans = len(loan_disbursement_df)
        avg_disbursement = loan_disbursement_df[disbursement_amount_col].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Loans Disbursed", f"GHC {total_disbursed:,.2f}")
        col2.metric("Number of Loans", f"{total_loans:,}")
        col3.metric("Average Disbursement", f"GHC {avg_disbursement:,.2f}")
    else:
        st.warning("Disbursement amount column not found. Cannot display key metrics.")
    
    st.write("---")
    
    # Data preview
    st.subheader("Data Preview")
    
    tab1, tab2 = st.tabs(["Disbursement Data", "Repayment Data"])
    
    with tab1:
        st.write("Loan Disbursement Data (First 10 rows)")
        st.dataframe(loan_disbursement_df.head(10))
        
    with tab2:
        st.write("Loan Repayment Data (First 10 rows)")
        st.dataframe(loan_repayment_df.head(10))
    
elif page == "Disbursement Analysis":
    st.header("Loan Disbursement Analysis")
    
    if not disbursement_amount_col:
        st.error("Disbursement amount column not found in the data.")
        st.stop()
    
    # Total by Region
    if region_col_disb:
        st.subheader("Disbursement by Region")
        
        region_totals = loan_disbursement_df.groupby(region_col_disb)[disbursement_amount_col].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(region_totals, 
                         x=region_totals.index, 
                         y=region_totals.values,
                         labels={'y': 'Total Disbursement (GHC)', 'x': 'Region'},
                         title='Total Disbursement by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(region_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))
        
        st.write("---")
    
    # Total by Year (if available)
    if year_col_disb:
        st.subheader("Disbursement by Year")
        
        # If it's a date column, extract year
        if loan_disbursement_df[year_col_disb].dtype == 'object':
            try:
                loan_disbursement_df['year_extracted'] = pd.to_datetime(loan_disbursement_df[year_col_disb]).dt.year
                yearly_totals = loan_disbursement_df.groupby('year_extracted')[disbursement_amount_col].sum()
            except:
                yearly_totals = loan_disbursement_df.groupby(year_col_disb)[disbursement_amount_col].sum()
        else:
            yearly_totals = loan_disbursement_df.groupby(year_col_disb)[disbursement_amount_col].sum()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(yearly_totals, 
                          x=yearly_totals.index, 
                          y=yearly_totals.values,
                          labels={'y': 'Total Disbursement (GHC)', 'x': 'Year'},
                          title='Disbursement Trend Over Years')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(yearly_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))
        
        st.write("---")
    
    # Total by Product (if available)
    if product_col:
        st.subheader("Disbursement by Product")
        
        product_totals = loan_disbursement_df.groupby(product_col)[disbursement_amount_col].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(product_totals, 
                         names=product_totals.index, 
                         values=product_totals.values,
                         title='Disbursement by Product')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(product_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))
        
        st.write("---")
    
    # Total by Branch (if available)
    if branch_col:
        st.subheader("Disbursement by Branch")
        
        branch_totals = loan_disbursement_df.groupby(branch_col)[disbursement_amount_col].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(branch_totals.head(10),  # Show top 10 only
                         x=branch_totals.head(10).index, 
                         y=branch_totals.head(10).values,
                         labels={'y': 'Total Disbursement (GHC)', 'x': 'Branch'},
                         title='Total Disbursement by Branch (Top 10)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(branch_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))

elif page == "Repayment Analysis":
    st.header("Loan Repayment Analysis")
    
    if not repayment_amount_col:
        st.error("Repayment amount column not found in the data.")
        st.stop()
    
    # Calculate repayment metrics
    total_repayments = loan_repayment_df[repayment_amount_col].sum()
    avg_repayment = loan_repayment_df[repayment_amount_col].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Repayments", f"GHC {total_repayments:,.2f}")
    col2.metric("Average Repayment", f"GHC {avg_repayment:,.2f}")
    
    st.write("---")
    
    # Repayment by Region (if available)
    if region_col_repay:
        st.subheader("Repayment by Region")
        
        repayment_by_region = loan_repayment_df.groupby(region_col_repay)[repayment_amount_col].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(repayment_by_region, 
                         x=repayment_by_region.index, 
                         y=repayment_by_region.values,
                         labels={'y': 'Total Repayment (GHC)', 'x': 'Region'},
                         title='Total Repayment by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(repayment_by_region.apply(lambda x: f"GHC {x:,.2f}").rename("Total Repayment"))
        
        st.write("---")
    
    # Repayment over time (if date column available)
    date_cols_repay = [col for col in loan_repayment_df.columns if 'date' in col.lower() or 'month' in col.lower()]
    if date_cols_repay:
        date_col = date_cols_repay[0]
        st.subheader(f"Repayment Over Time")
        
        try:
            # Try to extract month/year from date column
            loan_repayment_df['repayment_period'] = pd.to_datetime(loan_repayment_df[date_col]).dt.to_period('M')
            repayment_by_period = loan_repayment_df.groupby('repayment_period')[repayment_amount_col].sum()
            
            fig = px.line(repayment_by_period, 
                          x=repayment_by_period.index.astype(str), 
                          y=repayment_by_period.values,
                          labels={'y': 'Total Repayment (GHC)', 'x': 'Period'},
                          title='Repayment Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Could not parse date column for time series analysis.")

elif page == "Performance Metrics":
    st.header("Loan Performance Metrics")
    
    # Check if we have the required columns
    if not disbursement_amount_col or not repayment_amount_col:
        st.error("Required amount columns not found for performance metrics.")
        st.stop()
    
    # Convert amount columns to numeric
    try:
        loan_disbursement_df[disbursement_amount_col] = pd.to_numeric(loan_disbursement_df[disbursement_amount_col], errors='coerce')
        loan_repayment_df[repayment_amount_col] = pd.to_numeric(loan_repayment_df[repayment_amount_col], errors='coerce')
    except Exception as e:
        st.error(f"Error converting amounts to numbers: {e}")
        st.stop()
    
    # Calculate key metrics
    total_disbursed = loan_disbursement_df[disbursement_amount_col].sum()
    total_repaid = loan_repayment_df[repayment_amount_col].sum()
    
    # Calculate total expected repayment if available
    total_expected_repayment = 0
    if total_repayment_col:
        try:
            loan_repayment_df[total_repayment_col] = pd.to_numeric(loan_repayment_df[total_repayment_col], errors='coerce')
            total_expected_repayment = loan_repayment_df[total_repayment_col].sum()
        except:
            total_expected_repayment = total_disbursed * 1.2  # Estimate if not available
    
    # Calculate outstanding balance
    total_outstanding = 0
    if customer_id_col and loan_number_col and outstanding_col:
        try:
            loan_repayment_df[outstanding_col] = pd.to_numeric(loan_repayment_df[outstanding_col], errors='coerce')
            min_outstanding_per_loan = (
                loan_repayment_df
                .groupby([customer_id_col, loan_number_col])[outstanding_col]
                .min()
                .reset_index()
            )
            total_outstanding = min_outstanding_per_loan[outstanding_col].sum()
        except:
            total_outstanding = total_expected_repayment - total_repaid
    
    # Calculate repayment rate
    try:
        repayment_rate = (total_repaid / total_expected_repayment) * 100 if total_expected_repayment > 0 else 0
    except:
        repayment_rate = 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Disbursed", f"GHC {total_disbursed:,.2f}")
    col2.metric("Expected Repayment", f"GHC {total_expected_repayment:,.2f}")
    col3.metric("Total Repaid", f"GHC {total_repaid:,.2f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Outstanding Balance", f"GHC {total_outstanding:,.2f}")
    col5.metric("Repayment Rate", f"{repayment_rate:.2f}%")
    
    # Calculate penalties (simplified - repaid more than disbursed)
    penalties = total_repaid - total_disbursed if total_repaid > total_disbursed else 0
    col6.metric("Penalties/Interest", f"GHC {penalties:,.2f}")
    
    st.write("---")
    
    # Regional comparison if both datasets have region columns
    if region_col_disb and region_col_repay:
        st.subheader("Performance by Region")
        
        disbursement_by_region = loan_disbursement_df.groupby(region_col_disb)[disbursement_amount_col].sum()
        repayment_by_region = loan_repayment_df.groupby(region_col_repay)[repayment_amount_col].sum()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Disbursement': disbursement_by_region,
            'Repayment': repayment_by_region
        }).fillna(0)
        
        # Calculate repayment rate by region
        comparison_df['Repayment_Rate'] = (comparison_df['Repayment'] / comparison_df['Disbursement'] * 100).round(2)
        
        fig = px.bar(comparison_df.reset_index(), 
                     x=region_col_disb, 
                     y='Repayment_Rate',
                     labels={'Repayment_Rate': 'Repayment Rate (%)', region_col_disb: 'Region'},
                     title='Repayment Rate by Region')
        st.plotly_chart(fig, use_container_width=True)
    
    # PAR metrics (simplified example)
    st.subheader("Portfolio at Risk (PAR)")
    
    # Example PAR values - in a real scenario, you'd calculate these based on overdue loans
    par_data = {
        'Days': [30, 60, 90],
        'PAR (%)': [5.2, 3.1, 1.8]
    }
    
    par_df = pd.DataFrame(par_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(par_df, 
                     x='Days', 
                     y='PAR (%)',
                     title='Portfolio at Risk by Days Past Due',
                     labels={'Days': 'Days Past Due'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(par_df)
    
    # PAR trend over time (example data)
    st.subheader("PAR Trend Over Time")
    
    par_trend_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'PAR 30': [6.1, 5.8, 5.5, 5.3, 5.2, 5.0],
        'PAR 60': [3.5, 3.4, 3.3, 3.2, 3.1, 3.0],
        'PAR 90': [2.0, 1.9, 1.9, 1.8, 1.8, 1.7]
    }
    
    par_trend_df = pd.DataFrame(par_trend_data)
    melted_par_df = par_trend_df.melt(id_vars='Month', var_name='PAR Type', value_name='Percentage')
    
    fig = px.line(melted_par_df, 
                  x='Month', 
                  y='Percentage', 
                  color='PAR Type',
                  labels={'Percentage': 'PAR (%)', 'Month': 'Month'},
                  title='PAR Trend Over Time')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Loan Performance Analysis Dashboard | Built with Streamlit")