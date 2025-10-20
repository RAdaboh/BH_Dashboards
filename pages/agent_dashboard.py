# agent_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\Evelyn Wullar\\Desktop\\BRIGHT HORIZON\\BrightHorizon\\data\\Featured data\\bright_horizons_agents_enriched.csv")  # Replace with actual file name

df = load_data()
st.title("Agent Performance & Coverage Dashboard")

# Sidebar Navigation
section = st.sidebar.radio("Choose Analysis", [
    "Agent Performance",
    "Service Coverage",
    "Customer Interaction"
])

# ====================== SECTION 1: Agent Performance ======================
if section == "Agent Performance":
    st.header("📊 Agent Performance Analysis")

    st.subheader("Top 10 Agents by Monthly Volume")
    top_agents = df.sort_values(by="monthly_volume_(ghs)", ascending=False).head(10)
    st.dataframe(top_agents[["agent_id", "region", "monthly_volume_(ghs)", "monthly_commission_(ghs)", "performance_tier_clean"]])

    st.subheader("Performance by Tier")
    tier_summary = df.groupby("performance_tier_clean")[["monthly_transactions", "monthly_volume_(ghs)", "monthly_commission_(ghs)"]].mean().round(2)
    st.dataframe(tier_summary)

    st.subheader("Commission Rate vs Monthly Volume")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="commission_rate_(%)", y="monthly_volume_(ghs)", hue="performance_tier_clean", ax=ax)
    st.pyplot(fig)

    #st.subheader("Compliance Score by Tier")
    #fig2, ax2 = plt.subplots()
    #sns.boxplot(data=df, x="performance_tier_clean", y="compliance_score", ax=ax2)
    #st.pyplot(fig2)



# ====================== SECTION 2: Service Coverage ======================
elif section == "Service Coverage":
    st.header("🗺️ Agent Service Coverage")

    st.subheader("Agent Count by Region")
    region_count = df['region'].value_counts()
    st.bar_chart(region_count)

   # ================= Regional Performance =================
    st.subheader("📊 Regional Performance")

    # Compute regional performance
    regional_performance = df.groupby("region")[[
        "monthly_volume_(ghs)", 
        "monthly_transactions", 
        "monthly_commission_(ghs)"
    ]].sum().round(2)

    # Sort by highest monthly transactions
    regional_performance = regional_performance.sort_values(
        by="monthly_transactions", ascending=False
    )

    # Show as table
    st.dataframe(regional_performance)

    # Optional visualization: bar chart for average monthly transactions
    st.subheader("🧾 Avg Monthly Transactions by Region")
    st.bar_chart(regional_performance["monthly_transactions"])

    

    st.subheader("Agent Locations Map")
    map_center = [df["gps_latitude"].mean(), df["gps_longitude"].mean()]
    agent_map = folium.Map(location=map_center, zoom_start=6)
    marker_cluster = MarkerCluster().add_to(agent_map)

    for _, row in df.iterrows():
        popup = f"""<b>ID:</b> {row['agent_id']}<br>
                    <b>Region:</b> {row['region']}<br>
                    <b>Tier:</b> {row['performance_tier_clean']}<br>
                    <b>Service:</b> {row['services_offered']}"""
        folium.Marker(
            location=[row["gps_latitude"], row["gps_longitude"]],
            popup=popup
        ).add_to(marker_cluster)

    folium_static(agent_map)

    

# ====================== SECTION 3: Customer Interaction ======================
elif section == "Customer Interaction":
    st.header("👥 Customer Interaction Analysis")

    st.subheader("Customer Base by Tier")
    fig4, ax4 = plt.subplots()


    sns.barplot(data=df, x='performance_tier_clean', y='active_customer_base', ax=ax4)

    ax4.set_ylabel("Active Customer Base")
    ax4.set_xlabel("Performance Tier")

    st.pyplot(fig4)



    st.subheader("Working Hours vs Customer Base")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df, x='working_hours', y='active_customer_base', hue='performance_tier_clean', ax=ax5)
    st.pyplot(fig5)

    st.subheader("Training Completed vs Customer Base")
    trained_group = df.groupby("training_completed")["active_customer_base"].mean().round(2)
    st.dataframe(trained_group)

# ====================== NEW INFO INSERTED HERE ======================
    st.subheader("Regional Performance by Training Completion")

    training_region_performance = (
        df.groupby(["region", "training_completed"])[["monthly_volume_(ghs)"]]
        .mean()
        .unstack()
    )

    st.dataframe(training_region_performance)

    # Grouped bar chart
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    training_region_performance.plot(kind="bar", ax=ax6)

    ax6.set_title("Regional Monthly Volume by Training Completion")
    ax6.set_ylabel("Average Monthly Volume (GHS)")
    ax6.set_xlabel("Region")
    ax6.tick_params(axis="x", rotation=45)

    st.pyplot(fig6)

    st.subheader("Engagement Score (Custom)")
    df['engagement_score'] = (df['monthly_transactions'] + df['active_customer_base']) / df['months_active']
    st.bar_chart(df.set_index("agent_id")["engagement_score"].sort_values(ascending=False).head(10))

    st.subheader("Engagement vs Performance")

    # Option 1: Scatter/Bubble Plot
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="engagement_score",
        y="monthly_volume_(ghs)",
        hue="performance_tier_clean",
        #size="active_customer_base",   # bubble size
        alpha=0.7,
        ax=ax7
    )
    ax7.set_title("Engagement Index vs Monthly Volume (Bubble Plot)")
    st.pyplot(fig7)

    summary_table = (
    df.groupby(["engagement_score", "performance_tier_clean"])["monthly_volume_(ghs)"]
      .sum()
      .reset_index()
      .rename(columns={"monthly_volume_(ghs)": "total_monthly_volume_(ghs)"})
      .sort_values(by="total_monthly_volume_(ghs)", ascending=False)
      .head(10)
)

st.dataframe(summary_table)


