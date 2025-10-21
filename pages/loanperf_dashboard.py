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

# Add this at the beginning of your script (after imports but before loading data)
st.markdown("""
<style>
    /* Main content text color */
    /*.css-18e3th9 {*/
      /*  color: #000000;*/
    }*/
    
    /* Sidebar text color */
    .css-1v3fvcr {
        color: #000000;
    }
    
    /* Metric labels */
    .stMetric label {
        color: #000000 !important;
    }
    
    /* Metric values */
    .stMetric div {
        color: #000000 !important;
    }
    
    /* Dataframe text */
    /*.dataframe { */
       /* color: #000000 !important; */
    /*}*/
    
    
    /* Chart text */
    .svg-container {
        color: #000000 !important;
    }
    
    /* Titles and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* General text */
    p, div {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        url1="https://drive.google.com/file/d/1p1u6r8lKoycS73UC3jGvihtMCqLAPqWo/view?usp=drive_link"
        loan_disbursement_df = pd.read_csv(url1)
        loan_repayment_df = pd.read_csv('https://drive.google.com/file/d/1o2O6aQS2gWiz5SJuZd_d1LoBPSrQfSU6/view?usp=drive_link')


        # Clean column names (replace spaces with underscores)
        loan_disbursement_df.columns = loan_disbursement_df.columns.str.replace(' ', '_')
        loan_repayment_df.columns = loan_repayment_df.columns.str.replace(' ', '_')
        
        return loan_disbursement_df, loan_repayment_df
    except FileNotFoundError:
        st.error("Data files not found. Please ensure the data files are in the correct location.")
        return None, None

loan_disbursement_df, loan_repayment_df = load_data()

if loan_disbursement_df is None or loan_repayment_df is None:
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Disbursement Analysis", "Repayment Analysis", "Performance Metrics"])

# Main content
st.title("Loan Performance Analysis Dashboard")


if page == "Overview":
    st.header("Overview")
    
    # Key metrics
    total_disbursed = loan_disbursement_df['Disbursement_Amount'].sum()
    total_loans = len(loan_disbursement_df)
    avg_disbursement = loan_disbursement_df['Disbursement_Amount'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans Disbursed", f"GHC {total_disbursed:,.2f}")
    col2.metric("Number of Loans", f"{total_loans:,}")
    col3.metric("Average Disbursement", f"GHC {avg_disbursement:,.2f}")
    
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
    
    # Total by Region
    st.subheader("Disbursement by Region")
    
    region_totals = loan_disbursement_df.groupby('Region')['Disbursement_Amount'].sum().sort_values(ascending=False)
    
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
    if 'Disbursement_Year' in loan_disbursement_df.columns:
        st.subheader("Disbursement by Year")
        
        yearly_totals = loan_disbursement_df.groupby('Disbursement_Year')['Disbursement_Amount'].sum()
        
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
    
    # Total by Product (if available)
    if 'Product_ID' in loan_disbursement_df.columns:
        st.subheader("Disbursement by Product")
        
        product_totals = loan_disbursement_df.groupby('Product_ID')['Disbursement_Amount'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(product_totals, 
                         names=product_totals.index, 
                         values=product_totals.values,
                         title='Disbursement by Product')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(product_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))
    
    # Total by Branch (if available)
    if 'Branch' in loan_disbursement_df.columns:
        st.subheader("Disbursement by Branch")
        
        branch_totals = loan_disbursement_df.groupby('Branch')['Disbursement_Amount'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(branch_totals, 
                         x=branch_totals.index, 
                         y=branch_totals.values,
                         labels={'y': 'Total Disbursement (GHC)', 'x': 'Branch'},
                         title='Total Disbursement by Branch')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(branch_totals.apply(lambda x: f"GHC {x:,.2f}").rename("Total Disbursement"))
    
    
    #####HERE######
elif page == "Repayment Analysis":
    st.header("Loan Repayment Analysis")
    
    # Check what columns are available in repayment data
    #st.write("Available columns in repayment data:", list(loan_repayment_df.columns))
    
    # Check if there's any amount column that could represent repayments
    amount_columns = [col for col in loan_repayment_df.columns if 'amount' in col.lower() or 'repayment' in col.lower()]
    
    if not amount_columns:
        st.warning("No repayment amount column found in the dataset. Please check your data.")
        st.write("Common column names to check for repayment amounts:")
        st.write("- Repayment_Amount")
        st.write("- Amount")
        st.write("- Payment_Amount")
        st.write("- Repaid_Amount")
    else:
        # Use the first amount column found
        repayment_col = amount_columns[0]
        #st.subheader(f"Total Repayments (using column: {repayment_col})")
        
        total_repayments = loan_repayment_df[repayment_col].sum()
        avg_repayment = loan_repayment_df[repayment_col].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Total Repayments", f"GHC {total_repayments:,.2f}")
        col2.metric("Average Repayment", f"GHC {avg_repayment:,.2f}")
        
        st.write("---")
        
        # Repayment by Region (if available)
        if 'Region' in loan_repayment_df.columns:
            st.subheader("Repayment by Region")
            
            repayment_by_region = loan_repayment_df.groupby('Region')[repayment_col].sum().sort_values(ascending=False)
            
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
        
        # Repayment by Year (if available)
        year_cols = [col for col in loan_repayment_df.columns if 'year' in col.lower() or 'date' in col.lower()]
        if year_cols:
            year_col = year_cols[0]
            st.subheader(f"Repayment by {year_col}")
            
            repayment_by_year = loan_repayment_df.groupby(year_col)[repayment_col].sum()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.line(repayment_by_year, 
                              x=repayment_by_year.index, 
                              y=repayment_by_year.values,
                              labels={'y': 'Total Repayment (GHC)', 'x': year_col},
                              title=f'Repayment Trend Over {year_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(repayment_by_year.apply(lambda x: f"GHC {x:,.2f}").rename("Total Repayment"))

elif page == "Performance Metrics":
    st.header("Loan Performance Metrics")
    
    # Check what columns are available in both datasets
    #st.write("Available columns in disbursement data:", list(loan_disbursement_df.columns))
    #st.write("Available columns in repayment data:", list(loan_repayment_df.columns))
    
    # Find potential amount columns
    disb_amount_cols = [col for col in loan_disbursement_df.columns if 'amount' in col.lower() or 'disbursement' in col.lower()]
    repay_amount_cols = [col for col in loan_repayment_df.columns if 'amount' in col.lower() or 'repayment' in col.lower() or 'paid' in col.lower()]
    
    if not disb_amount_cols or not repay_amount_cols:
        st.warning("Required amount columns not found in one or both datasets.")
        st.write("Please ensure your datasets contain columns representing disbursement and repayment amounts.")
    else:
        # Use the most relevant amount columns found
        disb_col = 'Disbursement_Amount'  # We know this exists from your data
        repay_col = 'Amount_Paid'  # Using Amount_Paid instead of Amount_Due
        
        # Convert amount columns to numeric, handling any errors
        try:
            loan_disbursement_df[disb_col] = pd.to_numeric(loan_disbursement_df[disb_col], errors='coerce')
            loan_repayment_df[repay_col] = pd.to_numeric(loan_repayment_df[repay_col], errors='coerce')
        except Exception as e:
            st.error(f"Error converting amounts to numbers: {e}")
            st.stop()
        
        #st.subheader("Key Performance Metrics")
        #st.write(f"Using disbursement column: {disb_col}")
        #st.write(f"Using repayment column: {repay_col}")
        
        # Calculate totals after converting to numeric
        total_disbursed = loan_disbursement_df[disb_col].sum()
        total_repaid = loan_repayment_df[repay_col].sum()
        total_repayment_sum = loan_disbursement_df['Total_Repayment'].sum()

        min_outstanding_per_loan = (
            loan_repayment_df
            .groupby(['Customer_ID', 'Loan_Number'])['Outstanding_Balance']
            .min()
            .reset_index()
        )

        # Sum up all the minimum outstanding balances
        total_min_outstanding_balance = min_outstanding_per_loan['Outstanding_Balance'].sum()

        
        # Simple repayment rate (this is a simplified calculation)
        try:
            repayment_rate = (total_repaid / total_disbursed) * 100 if total_disbursed > 0 else 0
        except Exception as e:
            st.error(f"Error calculating repayment rate: {e}")
            repayment_rate = 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Disbursed", f"GHC {total_disbursed:,.2f}")
        col2.metric("Expected Repayment", f"GHC {total_repayment_sum:,.2f}")
        col3.metric("Total Repaid", f"GHC {total_repaid:,.2f}")

    
        
        
        st.write("---")

        ###############################
        col1, col2, col3 = st.columns(3)
        col1.metric("Outstanding balance", f"GHC {total_min_outstanding_balance:,.2f}")
        col2.metric("Penalties Paid", f"GHC {(total_repaid - total_disbursed):,.2f}" if total_repaid > total_disbursed else "GHC 0.00")
        col3.metric("Repayment Rate", f"{repayment_rate:.2f}%")
        #col4.metric("Penalties Paid", f"GHC {(total_repaid - total_disbursed):,.2f}" if total_repaid > total_disbursed else "GHC 0.00")
        
        st.write("---")

        # Repayment vs Disbursement by Region (if both datasets have Region)
        region_col_disb = 'Region' if 'Region' in loan_disbursement_df.columns else None
        region_col_repay = 'Region' if 'Region' in loan_repayment_df.columns else None
        
        if region_col_disb and region_col_repay:
            st.subheader("Repayment vs Disbursement by Region")
            
            disbursement_by_region = loan_disbursement_df.groupby(region_col_disb)[disb_col].sum()
            repayment_by_region = loan_repayment_df.groupby(region_col_repay)[repay_col].sum()
            
            # Create a combined DataFrame
            comparison_df = pd.DataFrame({
                'Disbursement': disbursement_by_region,
                'Repayment': repayment_by_region
            }).reset_index()
            
            # Melt for plotting
            melted_df = comparison_df.melt(id_vars='Region', var_name='Type', value_name='Amount')
            
            fig = px.bar(melted_df, 
                         x='Region', 
                         y='Amount', 
                         color='Type',
                         barmode='group',
                         labels={'Amount': 'Amount (GHC)', 'Region': 'Region'},
                         title='Disbursement vs Repayment by Region')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Region column not found in one or both datasets - cannot show regional comparison.")
        
        ######HERE#####

        
        # Example: Assume loans are overdue if repayment is less than disbursement
        # You'll need to implement your actual PAR calculation based on your data structure
        par_30 = 5.2  # Example value - replace with your calculation
        par_60 = 3.1  # Example value - replace with your calculation
        par_90 = 1.8  # Example value - replace with your calculation
        
        col1, col2, col3 = st.columns(3)
        col1.metric("PAR 30 Days", f"{par_30}%")
        col2.metric("PAR 60 Days", f"{par_60}%")
        col3.metric("PAR 90 Days", f"{par_90}%")
        
        # PAR trend chart (example)
        par_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'PAR 30': [6.1, 5.8, 5.5, 5.3, 5.2, 5.0],
            'PAR 60': [3.5, 3.4, 3.3, 3.2, 3.1, 3.0],
            'PAR 90': [2.0, 1.9, 1.9, 1.8, 1.8, 1.7]
        }
        
        par_df = pd.DataFrame(par_data)
        melted_par_df = par_df.melt(id_vars='Month', var_name='PAR Type', value_name='Percentage')
        
        fig = px.line(melted_par_df, 
                      x='Month', 
                      y='Percentage', 
                      color='PAR Type',
                      labels={'Percentage': 'PAR (%)', 'Month': 'Month'},
                      title='PAR Trend Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("Required columns not available for performance metrics calculation.")

# Add some styling
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
</style>
""", unsafe_allow_html=True)