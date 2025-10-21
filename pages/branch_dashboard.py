# branch_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Load Data
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1azAAqoepNKIj7gqOkj6mpEkq9-1rRpCd"  # or Dropbox link
    df = pd.read_csv(url)
    return df

df = load_data()
st.title("📍 Branch Operations Dashboard")

# Sidebar Navigation
section = st.sidebar.radio("Select Analysis", ["Branch Performance", "Customer Footfall", "Geographic Coverage"])

# ============================== Branch Performance Analysis ==============================
if section == "Branch Performance":
    st.header("📈 Branch Performance Analysis")

    st.subheader("Top 5 Branches by Profitability Index")
    top_profit = df.sort_values(by="Profitability Index", ascending=False).head(5)
    st.dataframe(top_profit[["Branch Name", "Region", "Profitability Index", "Loan Portfolio (GHS Million)", "Deposits (GHS Million)"]])

    st.subheader("📊 Performance Category Distribution")
    st.bar_chart(df["Performance Category"].value_counts())

    st.subheader("🔗 Correlation of Key Metrics")
    corr_cols = ['Loan Portfolio (GHS Million)', 'Deposits (GHS Million)', 
                 'Profitability Index', 'Loan Portfolio Quality (%)', 
                 'Collection Rate (%)', 'NPL Rate (%)', 'Market Share (%)']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ============================== Customer Footfall Analysis ==============================
elif section == "Customer Footfall":
    st.header("👥 Customer Footfall Analysis")

    df["Customer_to_Staff"] = df["Active Customers"] / df["Staff Count"]
    df["Service Capacity Score"] = df["Customer Service Desks"] / df["Active Customers"]

    st.subheader("📌 Customer to Staff Ratio")
    st.bar_chart(df.set_index("Branch Name")["Customer_to_Staff"])

    st.subheader("⭐ Satisfaction vs Load")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Customer_to_Staff", y="Customer Satisfaction (5.0)", hue="Performance Category", ax=ax)
    st.pyplot(fig)

    st.subheader("🧮 Service Capacity Score")
    st.dataframe(df[["Branch Name", "Customer_to_Staff", "Service Capacity Score"]].sort_values(by="Customer_to_Staff", ascending=False))

# ============================== Geographic Coverage Analysis ==============================
elif section == "Geographic Coverage":
    st.header("🗺️ Geographic Coverage Analysis")

    st.subheader("Branch Distribution by Region")
    st.bar_chart(df["Region"].value_counts())

    st.subheader("Interactive Branch Map")

    branch_map = folium.Map(location=[df["GPS Latitude"].mean(), df["GPS Longitude"].mean()], zoom_start=6)

    for _, row in df.iterrows():
        popup = f"{row['Branch Name']} ({row['Region']})<br>Perf: {row['Performance Category']}<br>Profit Index: {row['Profitability Index']}"
        folium.Marker(
            location=[row["GPS Latitude"], row["GPS Longitude"]],
            popup=popup,
            icon=folium.Icon(color="blue" if row["ATM Available"] == "Yes" else "gray")
        ).add_to(branch_map)

    folium_static(branch_map)

    st.subheader("♿ Accessibility Summary")
    st.write(df["Disability Access"].value_counts())

    st.subheader("📊 Disability Access vs Active Customers")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df, x='Disability Access', y='Active Customers', ax=ax)
    ax.set_title("Active Customers by Disability Access")
    st.pyplot(fig)


    # ====================== Coverage Percentage Table ======================
    st.subheader("📊 Coverage % of District Population")
    
    # Mapping dictionary (you can store this outside to reuse across sections)
    branch_to_district = {
        'Accra Central': 'Accra Metropolitan Area (AMA)',
        'Adenta': 'Adentan Municipal',
        'Madina': 'La Nkwantanang Madina Municipal',
        'Tema': 'Tema Metropolitan Area (TMA)',
        'Kumasi Central': 'Kumasi Metropolitan Area (KMA)',
        'Adum': 'KMA-Bantama',
        'Bantama': 'KMA-Bantama',
        'Cape Coast': 'Cape Cape Metropolitan Area (CCMA)',
        'Koforidua': 'Juaben Municipal',
        'Takoradi': 'Sekondi Takoradi Metropolitan Area (STMA)',
        'Tamale': 'TMA-Tamale Central',
        'Ho': 'Ho Municipal',
        'Wa': 'Wa Municipal',
        'Bolgatanga': 'Bolgatanga Municipal',
        'Dambai': 'Krachi East Municipal',
        'Damongo': 'West Gonja',
        'Sunyani': 'Sunyani Municipal',
        'Sefwi Wiawso': 'Sefwi Wiawso Municipal',
        'Techiman': 'Techiman Municipal',
        'Head Office': 'Accra Metropolitan Area (AMA)',
    }
    
    # Create District column
    df['Matched_District'] = df['Branch Name'].map(branch_to_district)

    # Bring in district population (load or merge from district_df)
    # NOTE: You can load district_df at top of the script from CSV or define it programmatically
    district_df = pd.DataFrame({
        "Districts": [
            "Sekondi Takoradi Metropolitan Area (STMA)",
            "Cape Cape Metropolitan Area (CCMA)",
            "Accra Metropolitan Area (AMA)",
            "Adentan Municipal",
            "La Nkwantanang Madina Municipal",
            "Tema Metropolitan Area (TMA)",
            "Ho Municipal",
            "Juaben Municipal",
            "Kumasi Metropolitan Area (KMA)",
            "KMA-Bantama",
            "Sefwi Wiawso Municipal",
            "Sunyani Municipal",
            "Techiman Municipal",
            "Krachi East Municipal",
            "TMA-Tamale Central",
            "West Gonja",
            "Bolgatanga Municipal",
            "Wa Municipal"
        ],
        "Population": [
            85174, 67570, 121103, 108423, 83831, 69392, 61676, 21253, 164504,
            43026, 51890, 64799, 89186, 38222, 38878, 10305, 41187, 55802
        ]
    })
    
    df_merged = pd.merge(df, district_df, left_on='Matched_District', right_on='Districts', how='left')

    # Compute Coverage
    df_merged['Coverage (%)'] = (df_merged['Active Customers'] / df_merged['Population']) * 100
    df_merged['Coverage (%)'] = df_merged['Coverage (%)'].round(2)

    coverage_table = df_merged[['Branch Name', 'Matched_District', 'Active Customers', 'Population', 'Coverage (%)']]

    # Sort or show top/bottom coverage
    st.dataframe(coverage_table.sort_values(by='Coverage (%)', ascending=False))

