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
        #st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found at: {path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load data with error handling
repay = load_data()
if repay is None:
    st.error("❌ Failed to load data. Please check your data source.")
    st.stop()

import streamlit as st

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


# Replace spaces with underscores in the column names
# ...existing code...
# Replace spaces with underscores in the column names
# ...existing code...

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
# ...existing code...


# ------------------------------
# Section 1: Repayment Pattern Analysis
# ------------------------------
st.markdown("""
    <style>
        .section-header {
            font-size: 30px;  /* Adjust the size as needed */
            color: #1f77b4;  /* Optional: You can change the color as well */
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

# Display the plot in the second column
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))  
    status_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Repayment Status')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Payment Status')
    ax.set_xticklabels(status_counts.index, rotation=45)
    st.pyplot(fig)

# Payment Performance by Risk Classification
st.markdown("### Repayment Performance by Risk Classification")

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
#st.write("Performance by Risk Classification:")
st.dataframe(risk_performance)

# Regional Repayment Patterns
st.markdown("### Regional Repayment Patterns")

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
#st.write("Regional Performance:")
st.dataframe(regional_performance.sort_values('Payment_Success_Rate', ascending=False))

# Branch-Level Analysis (Top 10 Branches by Loan Volume)
st.markdown("### Branch-Level Analysis (Top 10 Branches by Loan Volume)")

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

# Display the top 10 branches table
#st.write("Top 10 Branches by Loan Volume:")
st.dataframe(top_branches)

# Use Streamlit columns to display regional and branch performance side by side
col1, col2 = st.columns(2)

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

# Group by 'Product_Type' and calculate the performance metrics
product_performance = repay.groupby('Product_Type').agg({
    'Current_Status': lambda x: (x == 'Paid').mean() * 100,
    'Days_Overdue': 'mean',
    'Interest_Rate': 'mean',
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

# Use Streamlit columns to display the plot side by side with the table
col1, col2 = st.columns(2)



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
#st.write("Monthly Repayment Trends:")
st.dataframe(monthly_trends)

# Use Streamlit columns to display the plot
col1, col2 = st.columns(2)



# Plot the Total Amount Due in the second column
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figure size to make it smaller
    ax.plot(monthly_trends['Due_Month'], monthly_trends['Total_Amount_Due'], marker='', color='g', label='Total Amount Due')
    ax.set_title('Total Amount Due Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Amount Due')
    ax.grid(True)
    st.pyplot(fig)

# Collection Probability by Product Type
st.markdown("### Collection Probability by Product Type")

# Group by 'Product_Type' and calculate the average collection probability
collection_prob_by_product = repay.groupby('Product_Type').agg({
    'Collection_Probability': 'mean',
    'Loan_Number': 'nunique'
}).rename(columns={
    'Collection_Probability': 'Average_Collection_Probability',
    'Loan_Number': 'Number_of_Loans'
}).sort_values('Average_Collection_Probability', ascending=False).round(2)

# Display the collection probability table
#st.write("Collection Probability by Product Type:")
st.dataframe(collection_prob_by_product)

# Use Streamlit columns to display the plot
col1, col2 = st.columns(2)



# Historical Monthly Cashflow Analysis
st.markdown("### 2. Expected cashflow")

# Ensure that 'Due_Date' is in datetime format
repay['Due_Date'] = pd.to_datetime(repay['Due_Date'])
repay['Due_Month'] = repay['Due_Date'].dt.to_period('M')

# Group by 'Due_Month' and calculate the required metrics
historical_monthly = repay.groupby('Due_Month').agg({
    'Total_Amount_Due': 'sum',
    'Principal_Component': 'sum',
    'Interest_Component': 'sum',
    'Penalty_Amount': 'sum',
    'Loan_Number': 'nunique',
    'Current_Status': lambda x: (x == 'Current').sum() / len(x) * 100
}).rename(columns={
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
#st.write("Historical Monthly Cashflow Analysis:")
st.dataframe(historical_monthly.head(12))

# Use Streamlit columns to display the plot side by side with the table
col1, col2 = st.columns(2)



import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Historical Monthly Cashflow Analysis
st.markdown("### Historical  Expected Monthly Cashflow Analysis")

# Ensure 'Due_Date' is in datetime format
repay['Due_Date'] = pd.to_datetime(repay['Due_Date'])
repay['Due_Month'] = repay['Due_Date'].dt.to_period('M')

# Group by 'Due_Month' and calculate the required metrics
historical_monthly = repay.groupby('Due_Month').agg({
    'Total_Amount_Due': 'sum',
    'Principal_Component': 'sum',
    'Interest_Component': 'sum',
    'Penalty_Amount': 'sum',
    'Loan_Number': 'nunique',
    'Current_Status': lambda x: (x == 'Current').sum() / len(x) * 100
}).rename(columns={
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
#st.write("Historical Monthly Cashflow Analysis:")
st.dataframe(historical_monthly.head(12))

# Cashflow Forecasting Model
def prepare_forecast_data(df, forecast_horizon=6):
    """Prepare data for cashflow forecasting"""
    
    # Create time series features
    forecast_df = df.set_index('Due_Month').copy()
    forecast_df['time_index'] = range(len(forecast_df))
    
    # Create lag features
    for lag in [1, 2, 3, 6, 12]:
        forecast_df[f'Total_Due_lag_{lag}'] = forecast_df['Total_Due'].shift(lag)
        forecast_df[f'Collection_Rate_lag_{lag}'] = forecast_df['Collection_Rate'].shift(lag)
    
    # Create rolling statistics
    forecast_df['Total_Due_rolling_mean_3'] = forecast_df['Total_Due'].rolling(window=3).mean()
    forecast_df['Total_Due_rolling_std_3'] = forecast_df['Total_Due'].rolling(window=3).std()
    forecast_df['Collection_Rate_rolling_mean_3'] = forecast_df['Collection_Rate'].rolling(window=3).mean()
    
    # Seasonal features
    forecast_df['month'] = forecast_df.index.month
    forecast_df['quarter'] = forecast_df.index.quarter
    
    return forecast_df.dropna()

# Prepare forecasting data
forecast_data = prepare_forecast_data(historical_monthly)

# Split into features and target
X = forecast_data.drop(['Total_Due', 'Actual_Collections', 'Collection_Efficiency'], axis=1)
y = forecast_data['Total_Due']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train forecasting model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Display model performance
st.write(f"Model Performance - MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}")

# Forecast future cashflows
def forecast_future_cashflow(model, last_data, periods=12):
    """Generate future cashflow forecasts"""

    future_forecasts = []
    current_data = last_data.copy()
    # For lag/rolling features, we need to keep track of previous predictions
    prev_total_due = [
        current_data['Total_Due_lag_1'],
        current_data['Total_Due_lag_2'],
        current_data['Total_Due_lag_3']
    ]

    for i in range(periods):
        # Predict next period
        prediction = model.predict(current_data.values.reshape(1, -1))[0]

        # Create new row for next period
        new_period = current_data.copy()

        # Update time index
        new_period['time_index'] = new_period['time_index'] + 1

        # Update lag features
        new_period['Total_Due_lag_1'] = prediction
        new_period['Total_Due_lag_2'] = prev_total_due[0]
        new_period['Total_Due_lag_3'] = prev_total_due[1]

        # Update rolling statistics (simplified)
        new_period['Total_Due_rolling_mean_3'] = (prediction + prev_total_due[0] + prev_total_due[1]) / 3

        # Update prev_total_due for next iteration
        prev_total_due = [prediction, prev_total_due[0], prev_total_due[1]]

        # Update month and quarter
        new_month = (current_data['month'] % 12) + 1
        new_period['month'] = new_month
        new_period['quarter'] = (new_month - 1) // 3 + 1

        future_forecasts.append(prediction)
        current_data = new_period

    return future_forecasts

# Get last available data point
last_data_point = X.iloc[-1]

# Generate 12-month forecast
future_forecast = forecast_future_cashflow(model, last_data_point, periods=12)

# Create future dates
last_date = historical_monthly['Due_Month'].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]

# Create forecast dataframe
future_cashflow = pd.DataFrame({
    'Due_Month': future_dates,
    'Forecasted_Collections': future_forecast,
    'Confidence_Lower': [x * 0.85 for x in future_forecast],  # 15% lower bound
    'Confidence_Upper': [x * 1.15 for x in future_forecast]   # 15% upper bound
})

# Display the 12-month forecast table
st.write("12-Month Expected Cashflow Forecast:")
st.dataframe(future_cashflow.round(2))

# Plot the forecast
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(future_cashflow['Due_Month'], future_cashflow['Forecasted_Collections'], label='Forecasted Collections', color='blue', marker='o')
ax.fill_between(future_cashflow['Due_Month'], future_cashflow['Confidence_Lower'], future_cashflow['Confidence_Upper'], color='lightblue', alpha=0.5, label='Confidence Interval (85% - 115%)')
ax.set_title('12-Month Cashflow Forecast')
ax.set_xlabel('Month')
ax.set_ylabel('Collections')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# Cashflow Analysis by Product Type
st.markdown("### Expected Cashflow Analysis by Product Type")

# Group by 'Due_Month' and 'Product_Type' and aggregate the necessary metrics
product_cashflow = repay.groupby(['Due_Month', 'Product_Type']).agg({
    'Total_Amount_Due': 'sum',
    'Principal_Component': 'sum',
    'Interest_Component': 'sum',
    'Collection_Probability': 'mean',
    'Loan_Number': 'nunique'
}).reset_index()

# Pivot the data for better analysis (Total Amount Due by Product Type)
product_pivot = product_cashflow.pivot_table(
    index='Due_Month', 
    columns='Product_Type', 
    values='Total_Amount_Due', 
    aggfunc='sum'
).fillna(0)

# Display product-wise cashflow distribution table
st.write("Product-wise Cashflow Distribution (First 10 months):")
st.dataframe(product_pivot.head(10))

# Plotting the cashflow trends by product type
fig, ax = plt.subplots(figsize=(10, 6))
product_pivot.plot(kind='line', ax=ax)
ax.set_title('Monthly Expected Cashflow by Product Type')
ax.set_xlabel('Month')
ax.set_ylabel('Total Amount Due')
ax.legend(title='Product Type')
ax.grid(True)
st.pyplot(fig)

# Cashflow Analysis by Risk Classification
st.markdown("### Expected Cashflow Analysis by Risk Classification")

# Group by 'Due_Month' and 'Risk_Classification' and aggregate the necessary metrics
risk_cashflow = repay.groupby(['Due_Month', 'Risk_Classification']).agg({
    'Total_Amount_Due': 'sum',
    'Collection_Probability': 'mean',
    'Days_Overdue': 'mean'
}).reset_index()

# Pivot the data for better analysis (Total Amount Due by Risk Classification)
risk_pivot = risk_cashflow.pivot_table(
    index='Due_Month', 
    columns='Risk_Classification', 
    values='Total_Amount_Due', 
    aggfunc='sum'
).fillna(0)

# Display risk classification cashflow distribution table
st.write("Risk Classification Cashflow Distribution (First 10 months):")
st.dataframe(risk_pivot.head(10))

# Plotting the cashflow trends by risk classification
fig, ax = plt.subplots(figsize=(10, 6))
risk_pivot.plot(kind='line', ax=ax)
ax.set_title('Expected Monthly Cashflow by Risk Classification')
ax.set_xlabel('Month')
ax.set_ylabel('Total Amount Due')
ax.legend(title='Risk Classification')
ax.grid(True)
st.pyplot(fig)


import pandas as pd
import numpy as np
import streamlit as st

# Simulate collection scenarios
def simulate_collection_scenarios(df, months=12):
    """Simulate Expected cashflow under different collection scenarios"""
    
    # Get upcoming installments
    future_installments = df[df['Due_Date'] > pd.Timestamp.now()]
    future_installments = future_installments[future_installments['Due_Date'] <= pd.Timestamp.now() + pd.DateOffset(months=months)]
    
    # Group by month
    monthly_expected = future_installments.groupby(
        future_installments['Due_Date'].dt.to_period('M')
    ).agg({
        'Total_Amount_Due': 'sum',
        'Collection_Probability': 'mean',
        'Loan_Number': 'nunique'
    }).rename(columns={'Loan_Number': 'Number_of_Installments'}).reset_index()
    
    # Calculate expected collections under different scenarios
    scenarios = {
        'Optimistic': 1.2,    # 20% better than expected
        'Expected': 1.0,      # As per collection probability
        'Pessimistic': 0.8,   # 20% worse than expected
        'Worst_Case': 0.6     # 40% worse than expected
    }
    
    scenario_results = {}
    for scenario_name, multiplier in scenarios.items():
        # Apply the multiplier correctly
        monthly_expected[f'Collections_{scenario_name}'] = (
            monthly_expected['Total_Amount_Due'] * 
            (monthly_expected['Collection_Probability'] * multiplier)
        )
        # Store the total for each scenario
        scenario_results[scenario_name] = monthly_expected[f'Collections_{scenario_name}'].sum()
    
    return monthly_expected, scenario_results

# Cashflow data (repay DataFrame from previous steps)
# Replace 'repay' with the actual DataFrame you're using
cashflow_data = repay

# Run collection scenario simulation
monthly_scenarios, total_scenarios = simulate_collection_scenarios(cashflow_data)

# Display results in Streamlit
st.markdown("### Monthly Collection Scenarios")

# Show monthly collection scenarios table
#st.write("Monthly Collection Scenarios:")
st.dataframe(monthly_scenarios[['Due_Date', 'Total_Amount_Due', 'Collection_Probability', 
                                'Collections_Optimistic', 'Collections_Expected',
                                'Collections_Pessimistic', 'Collections_Worst_Case']].head(12))

# Display total collections for each scenario
st.markdown("### Total Expected Collections under Different Scenarios")
for scenario, amount in total_scenarios.items():
    st.write(f"{scenario}: GHC {amount:,.2f}")



import pandas as pd
import streamlit as st

# Stress test cashflow under different scenarios
def stress_test_cashflow(df, stress_factors):
    """Stress test cashflow under adverse conditions"""
    
    stress_results = {}
    
    for factor_name, factor_value in stress_factors.items():
        # Apply stress factor to collection probabilities
        stressed_df = df.copy()
        stressed_df['Stressed_Probability'] = stressed_df['Collection_Probability'] * factor_value
        
        # Calculate stressed collections
        stressed_collections = stressed_df.groupby(
            stressed_df['Due_Date'].dt.to_period('M')
        ).apply(lambda x: (x['Total_Amount_Due'] * x['Stressed_Probability']).sum())
        
        stress_results[factor_name] = {
            'Total_Impact': stressed_collections.sum(),
            'Monthly_Impact': stressed_collections,
            'Reduction_Percentage': (1 - (stressed_collections.sum() / df['Total_Amount_Due'].sum())) * 100
        }
    
    return stress_results

# Define stress scenarios (factors)
stress_factors = {
    'Economic_Downturn': 0.7,      # 30% reduction in collections
    'Interest_Rate_Increase': 0.8, # 20% reduction
    'Seasonal_Decline': 0.9,       # 10% reduction
    'Operational_Issues': 0.85     # 15% reduction
}

# Run stress tests
stress_results = stress_test_cashflow(repay, stress_factors)

# Display results in Streamlit
st.markdown("### Expected Cashflow Stress Test Results")

# Show results for each stress scenario
for scenario, results in stress_results.items():
    st.write(f"**{scenario}:**")
    st.write(f"  - **Total Collections Impact**: ${results['Total_Impact']:,.2f}")
    st.write(f"  - **Reduction in Collections**: {results['Reduction_Percentage']:.1f}%")
    
    # Display monthly impact for the stress test scenario
    #st.write(f"**Monthly Impact:**")
    st.dataframe(results['Monthly_Impact'].reset_index().rename(columns={'Due_Date': 'Month', 0: 'Impact'}))



# Repayment Schedule Adherence Analysis
st.markdown("### 3. Repayment Schedule Adherence Analysis")

# Overall adherence statistics
#st.markdown("#### Overall Payment Status Distribution")

# Using 'repay' dataset for adherence analysis
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
#st.write("Overall Payment Status Distribution:")
st.dataframe(overall_adherence.style.format({
    'Total_Amount': 'GHC {:,.2f}',  
    'Days_Overdue': '{:.1f}',    # Format overdue days to 1 decimal
    'Percentage': '{:.2f}%'      # Format percentage with 2 decimals
}))



# Create Payment Timing Category based on Days_Overdue
def categorize_payment(row):
    if row['Days_Overdue'] < 0:
        return 'Early_Payment'
    elif row['Days_Overdue'] == 0:
        return 'On_Time'
    else:
        return 'Late_Payment'

# Apply the categorization to the DataFrame
repay['Payment_Timing_Category'] = repay.apply(categorize_payment, axis=1)



# Payment Timing Analysis for 'Paid' status
#st.markdown("#### Payment Timing Analysis")

# Filter for paid installments and group by 'Payment_Timing_Category'
timing_analysis = repay[repay['Current_Status'] == 'Paid'].groupby(
    'Payment_Timing_Category'
).agg({
    'Installment_Number': 'count',
    'Total_Amount_Due': 'sum',
    'Days_Overdue': ['mean', 'std']
}).round(2)

# Calculate the percentage of each payment timing category
timing_analysis['Percentage'] = (timing_analysis[('Installment_Number', 'count')] / 
                                timing_analysis[('Installment_Number', 'count')].sum() * 100).round(2)

# Display the payment timing analysis table
st.write("Payment Timing Analysis:")
st.dataframe(timing_analysis.style.format({
    'Total_Amount_Due': 'GHC {:,.2f}',  # Format total amount with currency
    'Days_Overdue': '{:.1f}',        # Format overdue days to 1 decimal
    'Percentage': '{:.2f}%'          # Format percentage with 2 decimals
}))



# Create Payment Timing Categories
def categorize_payment(row):
    if row['Days_Overdue'] < 0:
        return 'Early_Payment'
    elif row['Days_Overdue'] == 0:
        return 'On_Time'
    elif row['Days_Overdue'] > 0:
        return 'Late_Payment'
    else:
        return 'Missed_Payment'

# Apply the categorization to the DataFrame
repay['Payment_Timing_Category'] = repay.apply(categorize_payment, axis=1)

# You can also create the binary columns to track Early, On-Time, and Late Payments:
repay['Early_Payment'] = (repay['Days_Overdue'] < 0).astype(int)
repay['On_Time'] = (repay['Days_Overdue'] == 0).astype(int)
repay['Late_Payment'] = (repay['Days_Overdue'] > 0).astype(int)
repay['Missed_Payment'] = (repay['Days_Overdue'] > 0).astype(int)  # Assuming missed payment based on Days_Overdue




import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Monthly Adherence Trends
st.markdown("### Monthly Adherence Trends")

# Ensure 'Due_Date' is in datetime format (in case it wasn't already)
repay['Due_Date'] = pd.to_datetime(repay['Due_Date'])
repay['Due_Month'] = repay['Due_Date'].dt.to_period('M')

# Group by 'Due_Month' and calculate adherence statistics
monthly_trends = repay.groupby('Due_Month').agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Installment_Number': 'count'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Installment_Number': 'Total_Installments'
})

# Convert Due_Month back to a timestamp for plotting
monthly_trends['Due_Month'] = monthly_trends.index.to_timestamp()

# Multiply adherence rates by 100 to get percentages
monthly_trends[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Calculate rolling averages for smoothing (3-month and 6-month rolling moving averages)
for col in ['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']:
    monthly_trends[f'{col}_3MMA'] = monthly_trends[col].rolling(window=3).mean()
    monthly_trends[f'{col}_6MMA'] = monthly_trends[col].rolling(window=6).mean()

# Display the monthly adherence trends table
st.write("Monthly Adherence Trends (Last 12 months):")
st.dataframe(monthly_trends.head(12).round(2))

# Plotting the adherence rates
fig, ax = plt.subplots(figsize=(10, 6))

# Plot On-Time Rate, Early Payment Rate, Late Payment Rate, Missed Payment Rate
monthly_trends[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']].plot(ax=ax, 
                                                                                                              marker='o')
ax.set_title('Monthly Adherence Trends')
ax.set_xlabel('Month')
ax.set_ylabel('Adherence Rate (%)')
ax.legend(title='Adherence Categories')
ax.grid(True)


import pandas as pd
import numpy as np
import streamlit as st

# Adherence Patterns by Installment Number
st.markdown("### Adherence Patterns by Installment Number")

# Assuming 'repay' is the dataset you're working with (replace 'adherence_data' with 'repay')
installment_adherence = repay.groupby('Installment_Number').agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Installment_Amount': 'mean',
    'Loan_Number': 'count'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Loan_Number': 'Count'
})

# Multiply adherence rates by 100 to get percentages
installment_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Calculate the difference in adherence for each installment number (for identifying drops)
installment_adherence['Adherence_Drop'] = installment_adherence['On_Time_Rate'].diff().fillna(0)

# Identify critical installments where adherence drops significantly (e.g., more than 5%)
critical_installments = installment_adherence[installment_adherence['Adherence_Drop'] < -5]

# Display the full adherence table for the first 15 installment numbers
# st.write("Adherence by Installment Number:")
st.dataframe(installment_adherence.head(15).round(2))

# If there are critical installments with significant adherence drops, display them
if not critical_installments.empty:
    st.write(f"\nCritical Installments with Significant Adherence Drops:")
    st.dataframe(critical_installments[['On_Time_Rate', 'Adherence_Drop']].round(2))
else:
    st.write("\nNo critical installments with significant adherence drops found.")

# Optionally: You can also plot the adherence rates over installment numbers to visualize the trends
fig, ax = plt.subplots(figsize=(10, 6))

# Plot On-Time Rate, Early Payment Rate, Late Payment Rate, Missed Payment Rate over Installment Number
installment_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']].plot(ax=ax, marker='o')

ax.set_title('Adherence Patterns by Installment Number')
ax.set_xlabel('Installment Number')
ax.set_ylabel('Adherence Rate (%)')
ax.legend(title='Adherence Categories')
ax.grid(True)

# Display the plot
st.pyplot(fig)


import pandas as pd
import numpy as np
import streamlit as st

# Customer-level adherence patterns
st.markdown("### Customer-Level Adherence Patterns")

# Assuming 'repay' is the dataset you're working with (replace 'adherence_data' with 'repay')
customer_adherence = repay.groupby('Customer_ID').agg({
    'Loan_Number': 'nunique',
    'Installment_Number': 'count',
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Total_Amount_Due': 'sum',
    'Original_Loan_Amount': 'mean'
}).rename(columns={
    'Loan_Number': 'Number_of_Loans',
    'Installment_Number': 'Total_Installments',
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Total_Amount_Due': 'Total_Amount_Repaid',
    'Original_Loan_Amount': 'Average_Loan_Size'
})

# Multiply adherence rates by 100 to get percentages
customer_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Segment customers by adherence behavior
customer_adherence['Adherence_Segment'] = pd.cut(
    customer_adherence['On_Time_Rate'],
    bins=[0, 50, 75, 90, 95, 100],
    labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'],
    include_lowest=True
)

# Analyze segments
segment_analysis = customer_adherence.groupby('Adherence_Segment').agg({
    'On_Time_Rate': 'mean',
    'Days_Overdue': 'mean',
    'Number_of_Loans': 'mean',
    'Average_Loan_Size': 'mean'
}).round(2)

# Calculate number of customers in each segment
segment_analysis['Number_of_Customers'] = customer_adherence.groupby('Adherence_Segment').size()

# Calculate the percentage of customers in each segment
segment_analysis['Percentage'] = (segment_analysis['Number_of_Customers'] / 
                                 segment_analysis['Number_of_Customers'].sum() * 100).round(2)

# Display the segment analysis table
st.write("Customer Adherence Segments:")
st.dataframe(segment_analysis)

# Plotting the number of customers in each segment
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the number of customers in each segment
segment_analysis['Number_of_Customers'].plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Number of Customers in Each Adherence Segment')
ax.set_xlabel('Adherence Segment')
ax.set_ylabel('Number of Customers')
ax.grid(True)

# Display the plot
st.pyplot(fig)


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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

# Adherence by Product Type
st.markdown("### Adherence by Product Type")

# Group by 'Product_Type' and calculate adherence statistics
product_adherence = repay.groupby('Product_Type').agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Interest_Rate': 'mean',
    'Original_Loan_Amount': 'mean',
    'Loan_Number': 'count'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Loan_Number': 'Number_of_Installments'
}).round(2)

# Multiply adherence rates by 100 to get percentages
product_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Map product codes to names
product_adherence['Product_Name'] = product_adherence.index.map(product_name_mapping)

# Reorder columns to display Product Name first
product_adherence = product_adherence[['Product_Name', 'On_Time_Rate', 'Early_Payment_Rate', 
                                       'Late_Payment_Rate', 'Missed_Payment_Rate', 'Days_Overdue', 
                                       'Interest_Rate', 'Original_Loan_Amount', 'Number_of_Installments']]

# Display adherence by product type
st.write("Adherence by Product Type:")
st.dataframe(product_adherence)

# Plotting the adherence rates by product type
fig, ax = plt.subplots(figsize=(10, 6))

# Plot On-Time Rate, Early Payment Rate, Late Payment Rate, Missed Payment Rate
product_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']].plot(ax=ax, 
                                                                                                              marker='o')
ax.set_title('Adherence by Product Type')
ax.set_xlabel('Product Type')
ax.set_ylabel('Adherence Rate (%)')
ax.legend(title='Adherence Categories')
ax.grid(True)

# Display the plot
st.pyplot(fig)


# Adherence by Risk Classification
st.markdown("### Adherence by Risk Classification")

# Group by 'Risk_Classification' and calculate adherence statistics
risk_adherence = repay.groupby('Risk_Classification').agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Collection_Probability': 'mean',
    'Loan_Number': 'count'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Loan_Number': 'Number_of_Installments'
}).round(2)

# Multiply adherence rates by 100 to get percentages
risk_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Display adherence by risk classification
# st.write("Adherence by Risk Classification:")
st.dataframe(risk_adherence)





# Regional adherence patterns
regional_adherence = repay.groupby('Region').agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Loan_Number': 'nunique',
    'Customer_ID': 'nunique'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Loan_Number': 'Number_of_Loans',
    'Customer_ID': 'Number_of_Customers'
}).round(2)

regional_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

print("Regional Adherence Patterns:")
print(regional_adherence.sort_values('On_Time_Rate', ascending=False))

# Branch-level adherence (top 20 branches by volume)
branch_adherence = repay.groupby(['Branch', 'Region']).agg({
    'On_Time': 'mean',
    'Early_Payment': 'mean',
    'Late_Payment': 'mean',
    'Missed_Payment': 'mean',
    'Days_Overdue': 'mean',
    'Loan_Number': 'nunique'
}).rename(columns={
    'On_Time': 'On_Time_Rate',
    'Early_Payment': 'Early_Payment_Rate',
    'Late_Payment': 'Late_Payment_Rate',
    'Missed_Payment': 'Missed_Payment_Rate',
    'Loan_Number': 'Number_of_Loans'
}).round(2)

branch_adherence[['On_Time_Rate', 'Early_Payment_Rate', 'Late_Payment_Rate', 'Missed_Payment_Rate']] *= 100

# Top and bottom performing branches
top_branches = branch_adherence.nlargest(10, 'On_Time_Rate')
bottom_branches = branch_adherence.nsmallest(10, 'On_Time_Rate')

print("\nTop 10 Performing Branches:")
print(top_branches)

print("\nBottom 10 Performing Branches:")
print(bottom_branches)







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
