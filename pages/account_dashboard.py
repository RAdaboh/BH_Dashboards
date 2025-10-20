import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Account Analytics Dashboard", layout="wide")

st.write("This is the Account Dashboard.")
 
st.set_page_config(page_title="Account Dashboard", layout="wide")

#st.title("📊 Churn Prediction Results (from Test Data)")
 

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"/Users/cogniserve/Desktop/BrightHorizon/data/Featured data/bright_horizons_accounts_enriched.csv")  # Update path if different
    df['Opening Date'] = pd.to_datetime(df['Opening Date'], errors='coerce')
    df['Year Opened'] = df['Opening Date'].dt.year
    return df

df = load_data()

# Sidebar
st.sidebar.title("Filters")
year_filter = st.sidebar.multiselect("Select Year Opened", sorted(df['Year Opened'].dropna().unique()), default=sorted(df['Year Opened'].dropna().unique()))
account_type_filter = st.sidebar.multiselect("Select Account Types", df['Account Type'].unique(), default=df['Account Type'].unique())

df_filtered = df[df['Year Opened'].isin(year_filter) & df['Account Type'].isin(account_type_filter)]

# Title
st.title("💼 Account Analytics Dashboard")

# =========================
# 1. ACCOUNT ACTIVITY ANALYSIS
# =========================
st.header("📊 Account Activity Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Active vs Dormant Accounts")
    active_counts = df_filtered['Active Account'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(active_counts, labels=active_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    st.subheader("Distribution of Account Types")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, y='Account Type', ax=ax2, order=df_filtered['Account Type'].value_counts().index)
    ax2.set_title("Accounts by Type")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Accounts Opened by Year")
    year_counts = df_filtered['Year Opened'].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax3)
    ax3.set_xlabel("Year")
    #ax3.set_xticks(year_counts.index)
    ax3.set_xticklabels(year_counts.index, rotation=45)
    ax3.set_ylabel("Accounts Opened")
    st.pyplot(fig3)

with col4:
    st.subheader("Engagement by Alerts")
    alert_cols = ['Mobile Banking', 'SMS Alerts', 'Email Alerts']
    alert_counts = df_filtered[alert_cols].apply(lambda x: x.value_counts()).T.fillna(0)
    fig4, ax4 = plt.subplots()
    alert_counts.plot(kind='barh', stacked=True, ax=ax4)
    ax4.set_title("Alert Engagement")
    st.pyplot(fig4)

col5, col6 = st.columns(2)
with col5:
    st.subheader("Total Deposits & Withdrawals")
    branch_perf = df.groupby("Branch")[["Total Deposits", "Total Withdrawals"]].sum().reset_index()

    fig5, ax5 = plt.subplots(figsize=(8,5))
    branch_perf.plot(x="Branch", y=["Total Deposits", "Total Withdrawals"], kind="bar", ax=ax5)
    ax5.set_title("Total Deposits and Withdrawals per Branch")
    ax5.set_ylabel("Amount (GHS)")
    ax5.set_xlabel("Branch")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

with col6:
    st.subheader("Deposits & Withdrawals")
    product_perf = df.groupby("Product Code")[["Total Deposits", "Total Withdrawals"]].sum().reset_index()

    fig6, ax6 = plt.subplots(figsize=(8,5))
    product_perf.plot(x="Product Code", y=["Total Deposits", "Total Withdrawals"], kind="bar", ax=ax6)
    ax6.set_title("Deposits and Withdrawals by Product")
    ax6.set_ylabel("Amount (GHS)")
    ax6.set_xlabel("Product")
    plt.xticks(rotation=45)
    st.pyplot(fig6)



# Trend of Account Age and Dormancy Likelihood
st.subheader("Trend of New Accounts and Dormancy Likelihood")

# Create the countplot
fig5, ax5 = plt.subplots()
sns.countplot(data=df_filtered, x='Account Age (Years)', hue='Status', palette='Set1', ax=ax5)

# Customize the plot
ax5.set_title('Account Age vs. Status')
ax5.set_xlabel('Account Age (Years)')
ax5.set_ylabel('Number of Accounts')
ax5.legend(title='Status')
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig5)

# --- 6. Net inflows/outflows per product
st.subheader("Net Inflows/Outflows per Product")

product_netflow = df.groupby("Product Code")["Net Flow"].sum().reset_index()

fig7, ax7 = plt.subplots(figsize=(10,6))
sns.barplot(
    data=product_netflow.sort_values(by="Net Flow", ascending=False),
    x="Product Code", y="Net Flow", ax=ax7, palette="coolwarm"
)

ax7.set_title("Net Flow by Product")
ax7.set_xlabel("Product Code")
ax7.set_ylabel("Total Net Flow")
plt.setp(ax7.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig7)

# --- 8. Which products/customers yield higher interest revenue?
st.subheader("Interest Revenue by Product and Customer")

# Aggregate interest by product
product_interest = df.groupby("Product Code")["Interest Earned"].sum().reset_index().sort_values(by="Interest Earned", ascending=False)

# Aggregate interest by customer
customer_interest = df.groupby("Customer ID")["Interest Earned"].sum().reset_index().sort_values(by="Interest Earned", ascending=False)

# --- Visualization 1: Interest by Product
fig8, ax8 = plt.subplots(figsize=(10,6))
sns.barplot(data=product_interest, x="Product Code", y="Interest Earned", ax=ax8, palette="mako")

ax8.set_title("Interest Revenue by Product")
ax8.set_xlabel("Product Code")
ax8.set_ylabel("Total Interest Earned")
plt.setp(ax8.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig8)


# --- Visualization 2: Top 20 Customers by Interest
fig9, ax9 = plt.subplots(figsize=(10,6))
sns.barplot(data=customer_interest.head(10), x="Customer ID", y="Interest Earned", ax=ax9, palette="crest")

ax9.set_title("Top 10 Customers by Interest Revenue")
ax9.set_xlabel("Customer ID")
ax9.set_ylabel("Total Interest Earned")
plt.setp(ax9.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig9)


# =========================
# 2. BALANCE TREND ANALYSIS
# =========================
# 1. Balance Over Time (Monthly)
#st.subheader("Balance Over Time (Monthly)")

# Ensure 'Date' column is in datetime format
#df_filtered['Opening Date'] = pd.to_datetime(df_filtered['Opening Date'])

# Create YearMonth column
#df_filtered['Year'] = df_filtered['Opening Date']

# Group by Year and compute average balance
#yearly_balance = df_filtered.groupby('Year')['Current Balance'].mean().reset_index()

# Plot yearly average balance
#st.subheader("Balance Over Time (Yearly)")
#fig_yearly, ax_yearly = plt.subplots()
#sns.lineplot(data=yearly_balance, x='Year', y='Current Balance', marker='', ax=ax_yearly)
#ax_yearly.set_title('Average Balance Over Time (Yearly)')
#ax_yearly.set_xlabel('Year')
#ax_yearly.set_ylabel('Average Balance')
#st.pyplot(fig_yearly)



# 2. Balance Growth Rate (Monthly)
# st.subheader("Balance Growth Rate (Monthly)")

# Calculate yearly growth rate
#yearly_balance['Balance Growth Rate (%)'] = yearly_balance['Current Balance'].pct_change() * 100

# Plot growth rate
#st.subheader("Balance Growth Rate (Yearly)")
#fig_growth_yearly, ax_growth_yearly = plt.subplots()
#sns.barplot(data=yearly_balance, x='Year', y='Balance Growth Rate (%)', ax=ax_growth_yearly)
#ax_growth_yearly.set_title('Yearly Balance Growth Rate')
#ax_growth_yearly.set_xlabel('Year')
#ax_growth_yearly.set_ylabel('Growth Rate (%)')
#st.pyplot(fig_growth_yearly)



# =========================
# 3. ACCOUNT PROFITABILITY ANALYSIS
# =========================
st.header("💹 Account Profitability Analysis")

col7, col8 = st.columns(2)

with col7:
    st.subheader("Average Activity Score by Account Type")
    avg_activity = df_filtered.groupby('Account Type')['Activity Score'].mean().sort_values()
    fig9, ax9 = plt.subplots()
    avg_activity.plot(kind='barh', ax=ax9)
    st.pyplot(fig9)

with col8:
    st.subheader("Current Balance Distribution")
    fig10, ax10 = plt.subplots()
    sns.histplot(df_filtered['Current Balance'], bins=30, kde=True, ax=ax10)
    st.pyplot(fig10)

st.subheader("Average Current Balance by Year Opened")
avg_balance = df_filtered.groupby('Year Opened')['Current Balance'].mean().sort_index()

fig11, ax11 = plt.subplots()
avg_balance.plot(marker='o', ax=ax11)

ax11.set_ylabel("Average Balance")
ax11.set_xlabel("Year Opened")

# Ensure x-axis shows as integers, not decimals
ax11.set_xticks(avg_balance.index)
ax11.set_xticklabels(avg_balance.index, rotation=45)

st.pyplot(fig11)

# --- 9. Relationship between balance size and interest contribution
st.subheader("Relationship Between Balance Size and Interest Contribution")

fig10, ax10 = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df, x="Current Balance", y="Interest Earned", alpha=0.6, ax=ax10)

ax10.set_title("Balance Size vs Interest Contribution")
ax10.set_xlabel("Current Balance (GHS)")
ax10.set_ylabel("Interest Earned (GHS)")
plt.tight_layout()

st.pyplot(fig10)


# =========================
# End of Dashboard
# =========================
st.markdown("---")
st.caption("📌 Built with ❤️ using Streamlit | Data: Account Portfolio")
