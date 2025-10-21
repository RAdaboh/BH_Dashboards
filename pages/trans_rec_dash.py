import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Bright  Transaction Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 1.5rem 0rem 1rem 0rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #3a7ca5;
        margin: 1rem 0rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3a7ca5;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin-bottom: 1rem;
    }
    .plot-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e86ab;
        margin: 1rem 0;
    }
    .data-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedTransactionDashboard:
    def __init__(self, transactions_df, customer_df=None):
        self.transactions_df = transactions_df
        self.customer_df = customer_df
        self.setup_data()
        
    def setup_data(self):
        """Prepare and clean data for analysis"""
        # Convert dates
        self.transactions_df['Transaction_Date'] = pd.to_datetime(self.transactions_df['Transaction_Date'])
        self.transactions_df['Value_Date'] = pd.to_datetime(self.transactions_df['Value_Date'], errors='coerce')
        
        # Create time-based features
        self.transactions_df['Transaction_Hour'] = self.transactions_df['Transaction_Date'].dt.hour
        self.transactions_df['Transaction_DayOfWeek'] = self.transactions_df['Transaction_Date'].dt.day_name()
        self.transactions_df['Transaction_Month'] = self.transactions_df['Transaction_Date'].dt.month_name()
        self.transactions_df['Transaction_Year'] = self.transactions_df['Transaction_Date'].dt.year
        self.transactions_df['Year_Month'] = self.transactions_df['Transaction_Date'].dt.to_period('M')
        self.transactions_df['Is_Weekend'] = self.transactions_df['Transaction_Date'].dt.dayofweek >= 5
        self.transactions_df['Week_of_Month'] = (self.transactions_df['Transaction_Date'].dt.day - 1) // 7 + 1
        
        # Part of day categorization
        def get_part_of_day(hour):
            if 5 <= hour < 12: return 'Morning'
            elif 12 <= hour < 17: return 'Afternoon'
            elif 17 <= hour < 21: return 'Evening'
            else: return 'Night'
            
        self.transactions_df['Part_Of_Day'] = self.transactions_df['Transaction_Hour'].apply(get_part_of_day)
        
        # Account type from Account_Number
        if 'Account_Number' in self.transactions_df.columns:
            self.transactions_df['Account_Type'] = self.transactions_df['Account_Number'].str[:3]
        
        # Merge with customer data if available
        if self.customer_df is not None:
            try:
                self.merged_data = self.transactions_df.merge(
                    self.customer_df, 
                    left_on='Customer_ID', 
                    right_on='Customer ID', 
                    how='left'
                )
                
                # Create demographic segments
                income_bins = [0, 2000, 5000, 10000, 20000, np.inf]
                income_labels = ['Low (<2K)', 'Lower Middle (2K-5K)', 'Middle (5K-10K)', 'Upper Middle (10K-20K)', 'High (>20K)']
                self.merged_data['Income_Bracket'] = pd.cut(self.merged_data['Monthly Income (GHS)'], 
                                                           bins=income_bins, labels=income_labels)
                
                age_bins = [18, 25, 35, 45, 55, 65, 100]
                age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                self.merged_data['Age_Cohort'] = pd.cut(self.merged_data['Age'], bins=age_bins, labels=age_labels)
                
                # Urban vs Rural classification
                major_urban_centers = ['Accra', 'Kumasi', 'Tamale', 'Sekondi-Takoradi', 'Sunyani', 
                                     'Cape Coast', 'Obuasi', 'Tema', 'Teshi', 'Madina']
                if 'Town/District' in self.merged_data.columns:
                    self.merged_data['Area_Type'] = self.merged_data['Town/District'].apply(
                        lambda x: 'Urban' if x in major_urban_centers else 'Rural'
                    )
            except Exception as e:
                st.warning(f"Could not merge customer data properly: {e}")

    def display_executive_summary(self):
        """Display comprehensive executive summary"""
        st.markdown('<div class="section-header">🏦 Executive Summary</div>', unsafe_allow_html=True)
        
        # Key Metrics Row 1
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_transactions = len(self.transactions_df)
            st.metric("Total Transactions", f"{total_transactions:,}")
            
        with col2:
            total_amount = self.transactions_df['Transaction_Amount'].sum()
            st.metric("Total Amount", f"GHC {total_amount:,.0f}")
            
        with col3:
            avg_transaction = self.transactions_df['Transaction_Amount'].mean()
            st.metric("Avg Transaction", f"GHC {avg_transaction:,.2f}")
            
        with col4:
            unique_customers = self.transactions_df['Customer_ID'].nunique()
            st.metric("Unique Customers", f"{unique_customers:,}")
            
        with col5:
            fraud_count = self.transactions_df['Fraud_Flag'].sum() if 'Fraud_Flag' in self.transactions_df.columns else 0
            st.metric("Fraud Cases", f"{fraud_count}", delta_color="inverse")

        # Key Metrics Row 2
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            debit_count = (self.transactions_df['Debit_Amount'] > 0).sum()
            st.metric("Debit Transactions", f"{debit_count:,}")
            
        with col2:
            credit_count = (self.transactions_df['Credit_Amount'] > 0).sum()
            st.metric("Credit Transactions", f"{credit_count:,}")
            
        with col3:
            top_channel = self.transactions_df['Channel'].value_counts().index[0] if 'Channel' in self.transactions_df.columns else "N/A"
            st.metric("Top Channel", f"{top_channel}")
            
        with col4:
            top_payment = self.transactions_df['Payment_Method'].value_counts().index[0] if 'Payment_Method' in self.transactions_df.columns else "N/A"
            st.metric("Top Payment Method", f"{top_payment}")
            
        with col5:
            balance_mismatch = self.transactions_df['Balance_Mismatch'].sum() if 'Balance_Mismatch' in self.transactions_df.columns else 0
            st.metric("Balance Mismatches", f"{balance_mismatch:,}", delta_color="inverse")

        # Quick Insights
        st.markdown('<div class="subsection-header">📈 Quick Insights</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            peak_hour = self.transactions_df['Transaction_Hour'].value_counts().index[0]
            st.write(f"**Peak Transaction Hour**: {peak_hour}:00")
            st.write(f"**Busiest Day**: {self.transactions_df['Transaction_DayOfWeek'].value_counts().index[0]}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            if 'Channel' in self.transactions_df.columns:
                channel_dist = self.transactions_df['Channel'].value_counts(normalize=True).head(3)
                st.write("**Top 3 Channels**:")
                for channel, pct in channel_dist.items():
                    st.write(f"- {channel}: {pct:.1%}")
            else:
                st.write("**Transaction Patterns**:")
                st.write("- Analyze temporal trends")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            if hasattr(self, 'merged_data'):
                avg_income = self.merged_data['Monthly Income (GHS)'].mean()
                st.write(f"**Avg Customer Income**: GHC {avg_income:,.0f}")
                st.write(f"**Avg Customer Age**: {self.merged_data['Age'].mean():.1f} years")
            else:
                st.write("**Transaction Types**:")
                if 'Transaction_Type' in self.transactions_df.columns:
                    type_counts = self.transactions_df['Transaction_Type'].value_counts().head(2)
                    for ttype, count in type_counts.items():
                        st.write(f"- {ttype}: {count:,}")
                else:
                    st.write("- Debit/Credit analysis available")
            st.markdown('</div>', unsafe_allow_html=True)

    def transaction_patterns_tab(self):
        """Enhanced Transaction Patterns Analysis"""
        st.markdown('<div class="section-header">📊 Transaction Pattern Analysis</div>', unsafe_allow_html=True)
        
        # Temporal Analysis Section
        st.markdown('<div class="subsection-header">⏰ Temporal Patterns</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Hour of Day")
            hourly_volume = self.transactions_df['Transaction_Hour'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(hourly_volume.index, hourly_volume.values, color='skyblue', alpha=0.7, edgecolor='navy')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Number of Transactions', fontsize=12)
            ax.set_title('Transaction Volume by Hour of Day', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24))
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Day of Week")
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_volume = self.transactions_df['Transaction_DayOfWeek'].value_counts().reindex(weekday_order)
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(weekday_volume.index, weekday_volume.values, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Number of Transactions', fontsize=12)
            ax.set_title('Transaction Volume by Day of Week', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(weekday_volume.values)*0.01,
                       f'{height:,}', ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Week of Month and Part of Day Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Week of Month")
            week_of_month_data = self.transactions_df['Week_of_Month'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            weeks = [f'Week {i}' for i in week_of_month_data.index]
            bars = ax.bar(weeks, week_of_month_data.values, alpha=0.7, color='purple')
            ax.set_title('Transaction Volume by Week of Month', fontsize=14, fontweight='bold')
            ax.set_xlabel('Week of Month')
            ax.set_ylabel('Number of Transactions')
            ax.grid(True, alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(week_of_month_data.values)*0.01,
                       f'{height:,}', ha='center', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Part of Day")
            part_of_day_data = self.transactions_df['Part_Of_Day'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            part_of_day_data.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
            ax.set_title('Transaction Volume by Part of Day', fontsize=14, fontweight='bold')
            ax.set_xlabel('Part of Day')
            ax.set_ylabel('Number of Transactions')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
            # Monthly Trends - FIXED VERSION
            col1, col2 = st.columns(2)
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Monthly Transaction Trends")

            # Fixed aggregation - using proper column names
            monthly_trends = self.transactions_df.groupby('Year_Month').agg(
                Transaction_Count=('Transaction_ID', 'count'),
                Total_Amount=('Transaction_Amount', 'sum'),
                Avg_Amount=('Transaction_Amount', 'mean')
            ).reset_index()

            monthly_trends['Year_Month'] = monthly_trends['Year_Month'].astype(str)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Extract unique years for cleaner x-axis labels
            unique_years = sorted(self.transactions_df['Transaction_Year'].unique())

            # Create positions and labels for years only
            year_positions = []
            year_labels = []

            for year in unique_years:
                # Find the first occurrence of each year in the monthly data
                year_str = str(year)
                matching_indices = monthly_trends.index[monthly_trends['Year_Month'].str.startswith(year_str)].tolist()
                if matching_indices:
                    year_positions.append(matching_indices[0])  # First month of each year
                    year_labels.append(year_str)

            # Transaction Count - use numeric indexing for plotting but show only year labels
            x_values = range(len(monthly_trends))
            ax1.plot(x_values, monthly_trends['Transaction_Count'], 
                    color='blue', marker='o', linewidth=2, markersize=4)
            ax1.set_ylabel('Transaction Count', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('Monthly Transaction Volume Trend', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Set x-axis to show only year labels at the first month of each year
            ax1.set_xticks(year_positions)
            ax1.set_xticklabels(year_labels, rotation=0)  # No rotation for years

            # Total Amount (secondary axis)
            ax3 = ax1.twinx()
            ax3.plot(x_values, monthly_trends['Total_Amount'], 
                    color='red', marker='s', linewidth=2, markersize=4, linestyle='--')
            ax3.set_ylabel('Total Amount (GHC)', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
        
    
        
        # Amount Distribution Analysis
        st.markdown('<div class="subsection-header">💰 Amount Distribution Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Amount Distribution")
            
            # Filter positive amounts for better visualization
            transaction_amounts = self.transactions_df[self.transactions_df['Transaction_Amount'] > 0]['Transaction_Amount']
            amount_95 = transaction_amounts[transaction_amounts <= transaction_amounts.quantile(0.95)]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Histogram
            ax1.hist(amount_95, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Transaction Amount (GHC)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Transaction Amounts (95th Percentile)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Boxplot
            ax2.boxplot(transaction_amounts, vert=False)
            ax2.set_xlabel('Transaction Amount (GHC)')
            ax2.set_title('Boxplot of Transaction Amounts', fontweight='bold')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Debit vs Credit Analysis")
            
            debit_count = (self.transactions_df['Debit_Amount'] > 0).sum()
            credit_count = (self.transactions_df['Credit_Amount'] > 0).sum()
            total = len(self.transactions_df)
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Pie chart
            ax1.pie([debit_count, credit_count], labels=['Debit', 'Credit'], 
                   autopct='%1.1f%%', startangle=90, colors=['#ff6b6b', '#4ecdc4'],
                   textprops={'fontsize': 12})
            ax1.set_title('Debit vs Credit Transactions', fontweight='bold')
            
            # Ratio by Account Type
            if 'Account_Type' in self.transactions_df.columns:
                ratio_by_type = self.transactions_df.groupby('Account_Type').apply(
                    lambda x: pd.Series({
                        'Debit_Ratio': len(x[x['Debit_Amount'] > 0]) / len(x),
                        'Credit_Ratio': len(x[x['Credit_Amount'] > 0]) / len(x)
                    })
                ).reset_index()
                
                x = np.arange(len(ratio_by_type))
                width = 0.35
                
                ax2.bar(x - width/2, ratio_by_type['Debit_Ratio'] * 100, width, label='Debit %', alpha=0.7)
                ax2.bar(x + width/2, ratio_by_type['Credit_Ratio'] * 100, width, label='Credit %', alpha=0.7)
                ax2.set_xlabel('Account Type')
                ax2.set_ylabel('Percentage (%)')
                ax2.set_title('Debit/Credit Ratio by Account Type', fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(ratio_by_type['Account_Type'])
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Account Type data\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Debit/Credit Ratio by Account Type', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", f"{total:,}")
            with col2:
                st.metric("Debit Transactions", f"{debit_count:,} ({debit_count/total*100:.1f}%)")
            with col3:
                st.metric("Credit Transactions", f"{credit_count:,} ({credit_count/total*100:.1f}%)")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Transaction Type Analysis
        if 'Transaction_Type' in self.transactions_df.columns:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Type Distribution")
            
            transaction_type_counts = self.transactions_df['Transaction_Type'].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            transaction_type_counts.plot(kind='bar', ax=ax, color='lightcoral', alpha=0.7)
            ax.set_title('Transaction Type Distribution', fontweight='bold')
            ax.set_xlabel('Transaction Type')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(transaction_type_counts.values):
                ax.text(i, v + max(transaction_type_counts.values)*0.01, f'{v:,}', 
                       ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Location Analysis
        if 'Location' in self.transactions_df.columns:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Location")
            
            location_counts = self.transactions_df['Location'].value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            location_counts.plot(kind='bar', ax=ax, color='lightblue', alpha=0.7)
            ax.set_title('Transactions by Location (Top 15)', fontweight='bold')
            ax.set_xlabel('Location')
            ax.set_ylabel('Transaction Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    def channel_preference_tab(self):
        """Enhanced Channel Preference Analysis"""
        st.markdown('<div class="section-header">🌐 Channel & Payment Method Analysis</div>', unsafe_allow_html=True)
        
        if 'Channel' not in self.transactions_df.columns or 'Payment_Method' not in self.transactions_df.columns:
            st.warning("Channel or Payment Method data not available in the dataset.")
            return
        
        # Channel Analysis
        st.markdown('<div class="subsection-header">📱 Channel Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Total Transaction Amount by Channel")
            channel_amounts = self.transactions_df.groupby('Channel')['Transaction_Amount'].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(channel_amounts.index, channel_amounts.values, color='lightblue', alpha=0.7, edgecolor='navy')
            ax.set_title('Total Transaction Amount by Channel', fontweight='bold', fontsize=14)
            ax.set_xlabel('Channel', fontsize=12)
            ax.set_ylabel('Total Amount (GHC)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(channel_amounts.values)*0.01,
                       f'GHC {height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Channel")
            channel_volume = self.transactions_df['Channel'].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(channel_volume.index, channel_volume.values, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax.set_title('Transaction Volume by Channel', fontweight='bold', fontsize=14)
            ax.set_xlabel('Channel', fontsize=12)
            ax.set_ylabel('Number of Transactions', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(channel_volume.values)*0.01,
                       f'{height:,}', ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Payment Method Analysis
        st.markdown('<div class="subsection-header">💳 Payment Method Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Total Amount by Payment Method")
            payment_amounts = self.transactions_df.groupby('Payment_Method')['Transaction_Amount'].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(payment_amounts.index, payment_amounts.values, color='salmon', alpha=0.7, edgecolor='darkred')
            ax.set_title('Total Transaction Amount by Payment Method', fontweight='bold', fontsize=14)
            ax.set_xlabel('Payment Method', fontsize=12)
            ax.set_ylabel('Total Amount (GHC)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(payment_amounts.values)*0.01,
                       f'GHC {height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Payment Method Usage")
            payment_methods = self.transactions_df['Payment_Method'].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(payment_methods.index, payment_methods.values, color='gold', alpha=0.7, edgecolor='darkorange')
            ax.set_title('Payment Method Usage', fontweight='bold', fontsize=14)
            ax.set_xlabel('Payment Method', fontsize=12)
            ax.set_ylabel('Number of Transactions', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(payment_methods.values)*0.01,
                       f'{height:,}', ha='center', va='bottom', fontweight='bold')
                
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Channel vs Payment Method Analysis
        st.markdown('<div class="subsection-header">🔄 Channel vs Payment Method Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Transaction Volume by Channel and Payment Method")
            
            channel_payment_combo = self.transactions_df.groupby(['Channel', 'Payment_Method'])['Transaction_ID'].count().unstack().fillna(0)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            channel_payment_combo.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
            ax.set_title('Transaction Volume by Channel and Payment Method', fontweight='bold', fontsize=14)
            ax.set_xlabel('Channel', fontsize=12)
            ax.set_ylabel('Number of Transactions', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Channel vs Payment Method Heatmap")
            
            channel_payment_combo = self.transactions_df.groupby(['Channel', 'Payment_Method'])['Transaction_ID'].count().unstack().fillna(0)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(channel_payment_combo.values, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(channel_payment_combo.columns)))
            ax.set_yticks(range(len(channel_payment_combo.index)))
            ax.set_xticklabels(channel_payment_combo.columns, rotation=45)
            ax.set_yticklabels(channel_payment_combo.index)
            ax.set_xlabel('Payment Method', fontsize=12)
            ax.set_ylabel('Channel', fontsize=12)
            ax.set_title('Transaction Count: Channel vs Payment Method', fontweight='bold', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Transaction Count', fontsize=12)
            
            # Add text annotations
            for i in range(len(channel_payment_combo.index)):
                for j in range(len(channel_payment_combo.columns)):
                    text = ax.text(j, i, f'{channel_payment_combo.iloc[i, j]:.0f}',
                                  ha="center", va="center", color="black", fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    def customer_behavior_tab(self):
        """Enhanced Customer Behavior Insights"""
        st.markdown('<div class="section-header">👥 Customer Behavior Insights</div>', unsafe_allow_html=True)
        
        if not hasattr(self, 'merged_data'):
            st.warning("Customer demographic data not available. Loading basic customer behavior from transaction data only.")
            
            # Show basic customer behavior from transaction data only
            st.markdown('<div class="subsection-header">📈 Basic Customer Behavior</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.subheader("Transactions per Customer")
                transactions_per_customer = self.transactions_df.groupby('Customer_ID')['Transaction_ID'].count()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(transactions_per_customer, bins=50, alpha=0.7, color='lightblue', edgecolor='black', log=True)
                ax.set_title('Distribution of Transactions per Customer', fontweight='bold')
                ax.set_xlabel('Number of Transactions per Customer')
                ax.set_ylabel('Number of Customers (Log Scale)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.subheader("Customer Spending Patterns")
                customer_spending = self.transactions_df.groupby('Customer_ID')['Transaction_Amount'].sum()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(customer_spending, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.set_title('Distribution of Total Spending per Customer', fontweight='bold')
                ax.set_xlabel('Total Spending (GHC)')
                ax.set_ylabel('Number of Customers')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            
            return
        
        # Demographic Analysis
        st.markdown('<div class="subsection-header">👨‍👩‍👧‍👦 Demographic Analysis</div>', unsafe_allow_html=True)
        
        # Check if required columns exist
        required_cols = ['Income_Bracket', 'Age_Cohort']
        missing_cols = [col for col in required_cols if col not in self.merged_data.columns]
        
        if missing_cols:
            st.warning(f"Missing demographic columns: {missing_cols}. Some visualizations may not be available.")
        
        if 'Income_Bracket' in self.merged_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.subheader("Customer Distribution by Income Bracket")
                
                income_dist = self.merged_data['Income_Bracket'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(income_dist.index, income_dist.values, color='lightblue', alpha=0.7, edgecolor='navy')
                ax.set_title('Customer Distribution by Income Bracket', fontweight='bold')
                ax.set_xlabel('Income Bracket')
                ax.set_ylabel('Number of Customers')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(income_dist.values)*0.01,
                           f'{height:,}', ha='center', va='bottom', fontweight='bold')
                    
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
        if 'Age_Cohort' in self.merged_data.columns:        
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.subheader("Customer Distribution by Age Cohort")
                
                age_dist = self.merged_data['Age_Cohort'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(age_dist.index, age_dist.values, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
                ax.set_title('Customer Distribution by Age Cohort', fontweight='bold')
                ax.set_xlabel('Age Cohort')
                ax.set_ylabel('Number of Customers')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(age_dist.values)*0.01,
                           f'{height:,}', ha='center', va='bottom', fontweight='bold')
                    
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Main dashboard
    st.markdown('<div class="main-header">🏦 Bank Transaction Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        transactions_df = pd.read_csv(r'/Users/cogniserve/Desktop/BrightHorizon/data/Featured data/bright_horizons_transaction_recs_enriched.csv')
        customer_df = pd.read_csv(r'/Users/cogniserve/Desktop/BrightHorizon/data/Featured data/bright_horizons_customers_enriched.csv')
        
        # Data preprocessing
        transactions_df['Transaction_Date'] = pd.to_datetime(transactions_df['Transaction_Date'])
        if 'Customer Since' in customer_df.columns:
            customer_df['Customer Since'] = pd.to_datetime(customer_df['Customer Since'])
        if 'Date of Birth' in customer_df.columns:
            customer_df['Date of Birth'] = pd.to_datetime(customer_df['Date of Birth'])
            
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.info("Using sample data for demonstration...")
        
        # Create comprehensive sample data
        np.random.seed(42)
        n_transactions = 50000
        n_customers = 5000
        
        # Sample transactions data
        transactions_df = pd.DataFrame({
            'Transaction_ID': [f'TXN{i:06d}' for i in range(n_transactions)],
            'Customer_ID': [f'CUST{i:05d}' for i in np.random.randint(0, n_customers, n_transactions)],
            'Transaction_Date': pd.date_range('2022-01-01', periods=n_transactions, freq='H'),
            'Transaction_Amount': np.random.exponential(500, n_transactions),
            'Debit_Amount': np.random.exponential(300, n_transactions),
            'Credit_Amount': np.random.exponential(400, n_transactions),
            'Channel': np.random.choice(['Mobile', 'Internet Banking', 'Branch', 'ATM', 'POS'], n_transactions, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            'Payment_Method': np.random.choice(['Transfer', 'Cash', 'Card', 'Cheque'], n_transactions, p=[0.5, 0.3, 0.15, 0.05]),
            'Location': np.random.choice(['Accra', 'Kumasi', 'Takoradi', 'Tamale', 'Cape Coast', 'Tema', 'Madina', 'Obuasi'], n_transactions),
            'Transaction_Type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Bill'], n_transactions),
            'Fraud_Flag': np.random.choice([True, False], n_transactions, p=[0.01, 0.99]),
            'Balance_Mismatch': np.random.choice([True, False], n_transactions, p=[0.02, 0.98]),
            'Account_Number': [f'ACC{i:08d}' for i in np.random.randint(10000000, 99999999, n_transactions)]
        })
        
        # Sample customer data
        customer_df = pd.DataFrame({
            'Customer ID': [f'CUST{i:05d}' for i in range(n_customers)],
            'Age': np.random.randint(18, 70, n_customers),
            'Monthly Income (GHS)': np.random.exponential(3000, n_customers),
            'Region': np.random.choice(['Greater Accra', 'Ashanti', 'Western', 'Northern', 'Eastern', 'Central'], n_customers),
            'Town/District': np.random.choice(['Accra', 'Kumasi', 'Takoradi', 'Tamale', 'Cape Coast', 'Tema', 'Madina', 'Obuasi', 'Sunyani'], n_customers),
            'Education Level': np.random.choice(['Primary', 'Secondary', 'Tertiary', 'Postgraduate'], n_customers, p=[0.1, 0.4, 0.4, 0.1]),
            'Employment Sector': np.random.choice(['Private', 'Public', 'Self-Employed', 'Unemployed', 'Student'], n_customers, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            'Customer Since': pd.to_datetime(np.random.choice(pd.date_range('2018-01-01', '2023-12-31', n_customers))),
            'Date of Birth': pd.to_datetime(np.random.choice(pd.date_range('1950-01-01', '2004-12-31', n_customers)))
        })
    
    # Initialize dashboard
    dashboard = EnhancedTransactionDashboard(transactions_df, customer_df)
    
    # Display executive summary
    dashboard.display_executive_summary()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Transaction Patterns", 
        "🌐 Channel Preferences", 
        "👥 Customer Behavior"
    ])
    
    with tab1:
        dashboard.transaction_patterns_tab()
        
    with tab2:
        dashboard.channel_preference_tab()
        
    with tab3:
        dashboard.customer_behavior_tab()

if __name__ == "__main__":
    main()