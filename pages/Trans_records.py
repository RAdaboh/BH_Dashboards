import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="BrightHorizon Transaction Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .section-header {font-size: 20px; color: #2ca02c; border-bottom: 2px solid #2ca02c; padding-bottom: 5px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

load_css()

# Load data
@st.cache_data
def load_data():
    transactions_file_id = '1qfvEooRGvqJQyGxRWT7QjhnSRf9gT2Oc'
    customer_file_id = '1T6vNJOlk43to_uWtgva4dAEb6hSncVZc'

    transactions_url = f'https://drive.google.com/uc?export=download&id={transactions_file_id}'
    customer_url = f'https://drive.google.com/uc?export=download&id={customer_file_id}'

    transactions_df = pd.read_csv(transactions_url)
    customer_df = pd.read_csv(customer_url)
    
    # Convert date columns
    transactions_df['Transaction_Date'] = pd.to_datetime(transactions_df['Transaction_Date'])
    
    # Extract time features
    transactions_df['Year'] = transactions_df['Transaction_Date'].dt.year
    transactions_df['Quarter'] = transactions_df['Transaction_Date'].dt.quarter
    transactions_df['Month'] = transactions_df['Transaction_Date'].dt.month
    transactions_df['Month_Name'] = transactions_df['Transaction_Date'].dt.month_name()
    transactions_df['Day_of_Week'] = transactions_df['Transaction_Date'].dt.day_name()
    transactions_df['Week_of_Month'] = (transactions_df['Transaction_Date'].dt.day - 1) // 7 + 1
    transactions_df['Hour'] = transactions_df['Transaction_Date'].dt.hour
    
    return transactions_df, customer_df

# Load data
transactions_df, customer_df = load_data()

# Merge customer and transaction data
@st.cache_data
def merge_data(transactions_df, customer_df):
    merged_data = transactions_df.merge(
        customer_df, 
        left_on='Customer_ID', 
        right_on='Customer ID', 
        how='left'
    )
    return merged_data

merged_data = merge_data(transactions_df, customer_df)

# Sidebar for filters
st.sidebar.header("Filters")
year_options = sorted(transactions_df['Year'].unique())
selected_years = st.sidebar.multiselect("Select Years", year_options, default=year_options)

channel_options = transactions_df['Channel'].unique()
selected_channels = st.sidebar.multiselect("Select Channels", channel_options, default=channel_options)

# Filter data based on selections
filtered_df = transactions_df[
    (transactions_df['Year'].isin(selected_years)) & 
    (transactions_df['Channel'].isin(selected_channels))
]

# Main dashboard
st.title("📊 BrightHorizon Transaction Analytics Dashboard")
st.markdown("---")

# Key Metrics
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_transactions = len(filtered_df)
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    total_amount = filtered_df['Transaction_Amount'].sum()
    st.metric("Total Amount", f"${total_amount:,.2f}")

with col3:
    avg_transaction = filtered_df['Transaction_Amount'].mean()
    st.metric("Avg. Transaction", f"${avg_transaction:,.2f}")

with col4:
    unique_customers = filtered_df['Customer_ID'].nunique()
    st.metric("Unique Customers", f"{unique_customers:,}")

st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Time Analysis", 
    "💳 Channel & Payment", 
    "👥 Customer Insights", 
    "🏦 Branch Analysis", 

    "📋 Executive Summary"
])

# Tab 1: Time Analysis
with tab1:
    st.header("Time-Based Transaction Patterns")
    
    # Quarterly Analysis - Aggregated Bar Chart
    st.subheader("Quarterly Trends (All Years Combined)")
    quarterly_agg = filtered_df.groupby('Quarter').size().reset_index(name='Count')
    quarterly_agg['Quarter'] = quarterly_agg['Quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    fig = px.bar(quarterly_agg, x='Quarter', y='Count', title='Total Transactions by Quarter')
    fig.update_traces(texttemplate='%{y:,}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Analysis
    st.subheader("Monthly Patterns")
    monthly_data = filtered_df.groupby(['Year', 'Month', 'Month_Name']).size().reset_index(name='Transaction_Count')
    monthly_avg = monthly_data.groupby(['Month', 'Month_Name'])['Transaction_Count'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('Month')
    
    fig = px.bar(monthly_avg, x='Month_Name', y='Transaction_Count', 
                 title='Average Monthly Transaction Volume')
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of Week Analysis
    st.subheader("Day of Week Patterns")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_data = filtered_df.groupby('Day_of_Week').size().reindex(weekday_order)
    
    fig = px.bar(x=weekday_data.index, y=weekday_data.values, 
                 title='Transaction Volume by Day of Week',
                 labels={'x': 'Day of Week', 'y': 'Number of Transactions'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly Analysis
    st.subheader("Hourly Patterns")
    hourly_data = filtered_df.groupby('Hour').size()
    
    fig = px.line(x=hourly_data.index, y=hourly_data.values, 
                  title='Transaction Count by Hour of Day',
                  labels={'x': 'Hour of Day', 'y': 'Number of Transactions'})
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Channel & Payment Analysis
with tab2:
    st.header("Channel and Payment Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel analysis
        st.subheader("Channel Performance")
        channel_data = filtered_df.groupby('Channel').agg({
            'Transaction_ID': 'count',
            'Transaction_Amount': 'sum'
        }).reset_index()
        
        fig = px.bar(channel_data, x='Channel', y='Transaction_ID', 
                     title='Transaction Volume by Channel')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(channel_data, x='Channel', y='Transaction_Amount', 
                     title='Transaction Amount by Channel')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment method analysis
        st.subheader("Payment Method Performance")
        payment_data = filtered_df.groupby('Payment_Method').agg({
            'Transaction_ID': 'count',
            'Transaction_Amount': 'sum'
        }).reset_index()
        
        fig = px.pie(payment_data, values='Transaction_ID', names='Payment_Method', 
                     title='Transaction Distribution by Payment Method')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(payment_data, x='Payment_Method', y='Transaction_Amount', 
                     title='Transaction Amount by Payment Method')
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel-Payment combo
    st.subheader("Channel and Payment Method Combination")
    channel_payment_combo = filtered_df.groupby(['Channel', 'Payment_Method'])['Transaction_ID'].count().unstack().fillna(0)
    
    fig = px.imshow(channel_payment_combo, 
                    title='Transaction Volume by Channel and Payment Method',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Customer Insights
with tab3:
    st.header("Customer Behavior Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer transaction frequency
        st.subheader("Customer Transaction Frequency")
        transactions_per_customer = filtered_df.groupby('Customer_ID')['Transaction_ID'].count()
        
        fig = px.histogram(transactions_per_customer, nbins=50,
                           title='Distribution of Transactions per Customer',
                           labels={'value': 'Number of Transactions', 'count': 'Number of Customers'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:

        # Debit vs Credit analysis
        st.subheader("Debit vs Credit Analysis")
        debit_count = (filtered_df['Debit_Amount'] > 0).sum()
        credit_count = (filtered_df['Credit_Amount'] > 0).sum()
        
        fig = px.pie(values=[debit_count, credit_count], 
                    names=['Debit Transactions', 'Credit Transactions'],
                    title='Debit vs Credit Transactions')
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Branch Analysis
with tab4:
    st.header("Branch Performance Analysis")
    
    # Identify top branches
    top_branches = filtered_df['Branch'].value_counts().head(5).index.tolist()
    branch_filter = st.multiselect("Select Branches", filtered_df['Branch'].unique(), default=top_branches)
    
    if branch_filter:
        branch_data = filtered_df[filtered_df['Branch'].isin(branch_filter)]
        
        # Branch performance
        branch_performance = branch_data.groupby('Branch').agg({
            'Transaction_ID': 'count',
            'Transaction_Amount': 'sum',
            'Customer_ID': 'nunique'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(branch_performance, x='Branch', y='Transaction_ID',
                         title='Transaction Volume by Branch')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(branch_performance, x='Branch', y='Transaction_Amount',
                         title='Transaction Amount by Branch')
            st.plotly_chart(fig, use_container_width=True)
        
        # Time patterns by branch
        st.subheader("Time Patterns by Branch")
        time_metric = st.selectbox("Select Time Metric", ['Hour', 'Day_of_Week', 'Month_Name'])
        
        branch_time_data = branch_data.groupby(['Branch', time_metric]).size().reset_index(name='Count')
        
        fig = px.line(branch_time_data, x=time_metric, y='Count', color='Branch',
                      title=f'Transaction Patterns by {time_metric.replace("_", " ")}')
        st.plotly_chart(fig, use_container_width=True)



# Tab 5: Executive Summary
with tab5:
    st.header("Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Insights")
        
        # Peak periods
        hourly_data = filtered_df.groupby('Hour').size()
        peak_hour = hourly_data.idxmax()
        
        weekday_data = filtered_df.groupby('Day_of_Week').size()
        peak_day = weekday_data.idxmax()
        
        monthly_data = filtered_df.groupby('Month_Name').size()
        peak_month = monthly_data.idxmax()
        
        st.info(f"""
        **Peak Periods:**
        - Busiest hour: {peak_hour}:00
        - Busiest day: {peak_day}
        - Busiest month: {peak_month}
        """)
        
        # Channel performance
        channel_data = filtered_df.groupby('Channel').agg({
            'Transaction_ID': 'count',
            'Transaction_Amount': 'sum'
        })
        top_channel = channel_data['Transaction_ID'].idxmax()
        top_channel_amount = channel_data['Transaction_Amount'].idxmax()
        
        st.info(f"""
        **Channel Performance:**
        - Most transactions: {top_channel}
        - Highest value: {top_channel_amount}
        """)
    
    with col2:
        st.subheader("Recommendations")
        
        st.success("""
        **Staffing Recommendations:**
        - Increase staff during peak hours (especially around {peak_hour}:00)
        - Ensure adequate coverage on {peak_day}s
        - Plan leaves during quieter periods
        
        **Operational Recommendations:**
        - Optimize resources for {top_channel} channel
        - Focus fraud detection efforts on high-risk channels and times
        - Consider promotions during slower periods to boost transaction volume
        """)
    
    # Performance trends
    st.subheader("Performance Trends")
    yearly_data = filtered_df.groupby('Year').agg({
        'Transaction_ID': 'count',
        'Transaction_Amount': 'sum'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add transaction count
    fig.add_trace(
        go.Bar(x=yearly_data['Year'], y=yearly_data['Transaction_ID'], name="Transaction Count"),
        secondary_y=False,
    )
    
    # Add transaction amount
    fig.add_trace(
        go.Scatter(x=yearly_data['Year'], y=yearly_data['Transaction_Amount'], name="Transaction Amount"),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Yearly Performance Trends"
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Transaction Amount ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**BrightHorizon Analytics Dashboard** | Created with Streamlit")