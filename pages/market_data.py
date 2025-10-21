import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(page_title="Market Data Dashboard", layout="wide")

# Title
st.title("Ghanaian Market Data Dashboard")

# Load data function
@st.cache_data
def load_data():
    industry_data = pd.read_excel('https://docs.google.com/spreadsheets/d/1Swv5f8TpawfsSln29Vai5TRcA98eOE5B/edit?usp=drive_link&ouid=113829564228791815939&rtpof=true&sd=true')
    industry_pop = pd.read_excel('https://docs.google.com/spreadsheets/d/1KV3WexpuHm_h5uEsj4yY8P5OwaVH8owr/edit?usp=drive_link&ouid=113829564228791815939&rtpof=true&sd=true')
    region_pop = pd.read_excel('https://docs.google.com/spreadsheets/d/1D-PokWv6szmbmjnHW_qpS-y3JKmEzV-d/edit?usp=drive_link&ouid=113829564228791815939&rtpof=true&sd=true')
    regbyindustry = pd.read_excel('https://docs.google.com/spreadsheets/d/1q-TEGK_Gk4lW6B7Sg_TGV00oqfbfkL1Y/edit?usp=drive_link&ouid=113829564228791815939&rtpof=true&sd=true')
    return industry_data, industry_pop, region_pop, regbyindustry

# Load data
industry_data, industry_pop, region_pop, regbyindustry = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_industry = st.sidebar.selectbox(
    "Select Industry", 
    industry_pop['Industry'].unique()
)

# Main content
tab1, tab2, tab3 = st.tabs(["Age Distribution", "Industry Population", "Region Analysis"])

with tab1:
    st.header("Age Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution in Industry Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=industry_data, x='Age Groups', y='Total', palette='Blues_d', ax=ax)
        plt.title('Age Distribution in Industry Data')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        #st.subheader("Age Group Statistics")
        #st.dataframe(industry_data.describe())
        
        # Calculate percentages
        industry_data['Percentage'] = (industry_data['Total'] / industry_data['Total'].sum()) * 100
        st.write("Percentage Distribution")
        st.dataframe(industry_data[['Age Groups', 'Percentage']])

with tab2:
    st.header("Industry Population Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Population by Industry")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=industry_pop, x='Industry', y='Total population', palette='Greens_d', ax=ax)
        plt.title('Total Population by Industry')
        plt.xlabel('Industry')
        plt.ylabel('Total Population')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Selected Industry Details")
        selected_data = industry_pop[industry_pop['Industry'] == selected_industry]
        st.write(f"Total Population for {selected_industry}: {selected_data['Total population'].values[0]:,}")
        
        # Calculate percentages
        industry_pop['Percentage'] = (industry_pop['Total population'] / industry_pop['Total population'].sum()) * 100
        st.write("Industry Distribution:")
        st.dataframe(industry_pop[['Industry', 'Total population', 'Percentage']].sort_values('Total population', ascending=False))

with tab3:
    st.header("Region Analysis")
    
    st.subheader("Region Population Overview")
    st.dataframe(region_pop.head())
    
    st.subheader("Region by Industry Overview")
    st.dataframe(regbyindustry.head())
    
    # You can add more visualizations here based on the region data
    if not regbyindustry.empty:
        # Step 1: Melt the data
        regbyindustry_melted = regbyindustry.melt(id_vars='Region', var_name='Industry', value_name='Count')

    
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        regbyindustry_melted['Count'] = pd.to_numeric(regbyindustry_melted['Count'], errors='coerce')
        regbyindustry_melted = regbyindustry_melted.dropna(subset=['Count'])
        sns.barplot(data=regbyindustry_melted, x='Region', y='Count', hue='Industry', ax=ax)
        plt.title('Industry Distribution by Region')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)



# Additional insights
st.sidebar.header("Insights")
if st.sidebar.checkbox("Show Key Insights"):
    st.sidebar.write("### Age Group Insights")
    max_age_group = industry_data.loc[industry_data['Total'].idxmax(), 'Age Groups']
    st.sidebar.write(f"- Most populated age group: {max_age_group}")
    
    st.sidebar.write("### Industry Insights")
    max_industry = industry_pop.loc[industry_pop['Total population'].idxmax(), 'Industry']
    st.sidebar.write(f"- Largest industry by population: {max_industry}")

# Data download option
st.sidebar.header("Data Export")
if st.sidebar.button("Download Sample Data"):
    csv = industry_data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Age Data as CSV",
        data=csv,
        file_name='age_distribution.csv',
        mime='text/csv'
    )

# About section
st.sidebar.header("About")
st.sidebar.info(
    "Powered By Bright Horizon\n"
    
)