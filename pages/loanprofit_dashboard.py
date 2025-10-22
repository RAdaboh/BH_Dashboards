import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Loan Profitability Dashboard")
st.title("Loan Product Profitability Dashboard")

# --- DATA LOADING AND CACHING ---

@st.cache_data
def load_data():
    try:
        # Try loading from Parquet if available (faster)
        parquet_url = "https://drive.google.com/uc?id=1nRU2R4u_ohjhjqjz6xtuwAl3HBbdmrRr&export=download"
        st.info("🔄 Loading data from Parquet...")
        df = pd.read_parquet(parquet_url)
        st.success(f"✅ Successfully loaded data from Parquet! Shape: {df.shape}")
        return df
    except Exception as e:
        st.warning(f"⚠️ Parquet loading failed: {e}")
        # Fallback to CSV if Parquet fails
        try:
            csv_url = "https://drive.google.com/uc?id=1-XTNFwFpEID7avG8uHToGVqxA0Vg9rfM&export=download"
            st.info("🔄 Loading data from CSV...")
            df = pd.read_csv(csv_url)
            st.success(f"✅ Successfully loaded data from CSV! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
            return pd.DataFrame()

@st.cache_data
def preprocess_data(df):
    # Compute derived columns
    df['Total_Fees_Earned'] = df['Processing_Fee'] + df['Insurance_Fee'] + df['Penalty_Paid']
    df['Revenue'] = df['Interest_Paid'] + df['Total_Fees_Earned']
    df['Total_Default'] = df.apply(lambda row: row['Outstanding_Balance'] if row['Delinquent'] == 1 else 0, axis=1)

    if 'Disbursement_Year' not in df.columns:
        df['Disbursement_Date'] = pd.to_datetime(df['Disbursement_Date'])
        df['Disbursement_Year'] = df['Disbursement_Date'].dt.year

    df['Disbursement_Year'] = df['Disbursement_Year'].astype(int)

    grouped = df.groupby(['Product_Name', 'Disbursement_Year', 'Region_x']).agg({
        'Disbursement_Amount': 'sum',
        'Interest_Paid': 'sum',
        'Total_Fees_Earned': 'sum',
        'Revenue': 'sum',
        'Total_Default': 'sum',
        'Installment/Loan_Ratio': 'mean'
    }).rename(columns={
        'Disbursement_Amount': 'Total_Disbursed',
        'Interest_Paid': 'Total_Interest',
        'Total_Fees_Earned': 'Total_Fees',
        'Revenue': 'Total_Revenue',
        'Total_Default': 'Total_Default_Amount',
        'Installment/Loan_Ratio': 'Avg_Installment_Loan_Ratio'
    }).reset_index()

    grouped['Profit'] = grouped['Total_Revenue'] - grouped['Total_Default_Amount']
    grouped['Profit_Margin_%'] = (grouped['Profit'] / grouped['Total_Revenue']) * 100
    grouped['Default_Rate_%'] = (grouped['Total_Default_Amount'] / grouped['Total_Disbursed']) * 100

    return grouped

# Load and preprocess with progress indicators
st.info("🔄 Loading data...")
df = load_data()

if df.empty:
    st.error("❌ No data loaded. Please check your data sources.")
    st.stop()

st.info("🔄 Preprocessing data...")
grouped = preprocess_data(df)

# Show data info for debugging
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Info")
st.sidebar.write(f"Total rows: {len(df):,}")
st.sidebar.write(f"Products: {len(grouped['Product_Name'].unique())}")
st.sidebar.write(f"Years: {grouped['Disbursement_Year'].min()} - {grouped['Disbursement_Year'].max()}")
st.sidebar.write(f"Regions: {len(grouped['Region_x'].unique())}")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("🔍 Filters")
    product_filter = st.multiselect("Product", options=grouped['Product_Name'].unique(), default=grouped['Product_Name'].unique())
    year_filter = st.slider("Year Range",
                            min_value=int(grouped['Disbursement_Year'].min()),
                            max_value=int(grouped['Disbursement_Year'].max()),
                            value=(int(grouped['Disbursement_Year'].min()), int(grouped['Disbursement_Year'].max())))
    region_filter = st.multiselect("Region", options=grouped['Region_x'].unique(), default=grouped['Region_x'].unique())

# --- APPLY FILTERS ---
filtered_df = grouped[
    (grouped['Product_Name'].isin(product_filter)) &
    (grouped['Disbursement_Year'] >= year_filter[0]) &
    (grouped['Disbursement_Year'] <= year_filter[1]) &
    (grouped['Region_x'].isin(region_filter))
]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("⚠️ No data matches your filter criteria. Please adjust your filters.")
    st.stop()

# --- PLOTTING FUNCTION ---
def plot_line(data, y_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Check if there's data to plot
    if data.empty:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
    else:
        sns.lineplot(data=data, x='Disbursement_Year', y=y_col, hue='Product_Name', marker='o', ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.legend(title='Product Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set integer x-ticks
        years = sorted(data['Disbursement_Year'].unique())
        ax.set_xticks(years)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    st.pyplot(fig)

# --- DASHBOARD SECTIONS ---

st.markdown("## 📈 Profitability Metrics")

col1, col2 = st.columns(2)
with col1:
    plot_line(filtered_df, 'Profit', 'Total Profit by Product Over the Years', 'Profit (GHS)')
with col2:
    plot_line(filtered_df, 'Total_Revenue', 'Revenue Trend by Product', 'Revenue (GHS)')

col3, col4 = st.columns(2)
with col3:
    plot_line(filtered_df, 'Total_Disbursed', 'Disbursement Trend by Product', 'Disbursed Amount (GHS)')
with col4:
    plot_line(filtered_df, 'Total_Interest', 'Interest Paid by Product', 'Interest Paid (GHS)')

st.markdown("## 🏆 Top 5 Most Profitable Products")

top_products = grouped.groupby('Product_Name')['Profit'].mean().nlargest(5).index
filtered_top = grouped[grouped['Product_Name'].isin(top_products)]

if not filtered_top.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_top, x='Disbursement_Year', y='Profit', hue='Product_Name', marker='o', ax=ax)
    ax.set_title("Profit Trend for Top 5 Products")
    ax.set_ylabel("Profit (GHS)")
    
    # Set integer x-ticks
    years = sorted(filtered_top['Disbursement_Year'].unique())
    ax.set_xticks(years)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    st.pyplot(fig)
else:
    st.warning("No data available for top products analysis.")

st.markdown("## 📋 Filtered Data Preview (Top 100 Rows)")
st.dataframe(filtered_df.head(100))

# Download button
if not filtered_df.empty:
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Filtered Data", csv_data, "filtered_profitability.csv")