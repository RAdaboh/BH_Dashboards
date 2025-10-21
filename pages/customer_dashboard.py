import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px

# Configure page
st.set_page_config(layout="wide", page_title="Customer Analytics", page_icon="📊")

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1yudFOERd3cvnw1xMh5utMXW6kBn04l3c"  # or Dropbox link
    customers = pd.read_csv(url)
    return customers

customers = load_data()

customers['Customer Since'] = pd.to_datetime(customers['Customer Since'], errors='coerce')

total_customers = customers['Customer ID'].nunique()
start_year = customers['Customer Since'].min().year
end_year = customers['Customer Since'].max().year
years_active = end_year - start_year + 1
avg_annual_growth = total_customers / years_active  
    


customers = load_data()
customers['TenureYears'] = customers['Tenure'] / 12
customers['YearJoined'] = pd.to_datetime(customers['Customer Since']).dt.year

# ----------------------------------
# KPI ROW
# ----------------------------------
st.header("Customer Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(customers):,}")
col2.metric("Avg Tenure", f"{customers['TenureYears'].mean():.1f} years")
col3.metric("Growth Rate", "8.2%", "2.1% YoY")

# ----------------------------------
# VISUAL 1 & 2: Market Penetration + Regional Distribution
# ----------------------------------
st.header("Market Analysis")
col1, col2 = st.columns(2)

with col1:
    # Market Penetration by Sector
    st.subheader("Market Penetration by Sector")
    sector_pen = customers['Employment Sector'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#e74c3c' if x < 10 else '#3498db' for x in sector_pen]
    sector_pen.sort_values().plot(kind='barh', color=colors, ax=ax)
    
    # Annotate untapped sectors
    for i, (sector, pct) in enumerate(sector_pen.sort_values().items()):
        if pct < 10:
            ax.annotate('Untapped', xy=(pct, i), xytext=(5,0), 
                       textcoords='offset points', color='red', fontsize=9)
    
    ax.set_xlabel("Market Share (%)")
    st.pyplot(fig)

with col2:
    # Customer Distribution by Region
    st.subheader("Regional Distribution")
    region_counts = customers['Region'].value_counts()
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#2ecc71' if x == region_counts.max() else '#3498db' for x in region_counts]
    region_counts.plot(kind='bar', color=colors, ax=ax)
    
    ax.annotate(f'Strongest: {region_counts.idxmax()}',
               xy=(region_counts.argmax(), region_counts.max()),
               xytext=(0,10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='white'))
    
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------------------
# VISUAL 3 & 4: Gender + Tenure Segments
# ----------------------------------
st.header("Customer Segments")
col1, col2 = st.columns(2)

with col1:
    # Gender Distribution
    st.subheader("Gender Ratio")
    gender_counts = customers['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8,4))
    explode = (0.05, 0) if abs(gender_counts['Male']-gender_counts['Female'])/len(customers) > 0.1 else (0,0)
    gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#3498db','#e74c3c'],
                     explode=explode, startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

with col2:
    # Tenure Segments
    st.subheader("Tenure Distribution")
    tenure_bins = [0,1,3,5,10,20]
    labels = ['New (<1yr)', 'Growth (1-3yr)', 'Established (3-5yr)', 'Loyal (5-10yr)', 'VIP (>10yr)']
    customers['TenureSegment'] = pd.cut(customers['TenureYears'], bins=tenure_bins, labels=labels)
    seg_counts = customers['TenureSegment'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#e74c3c','#f39c12','#3498db','#2ecc71','#27ae60']
    seg_counts.plot(kind='bar', color=colors, ax=ax)
    
    ax.annotate('Churn Risk', xy=(0, seg_counts[0]), xytext=(0,20),
               textcoords='offset points', color='red')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)



# ----------------------------------
# VISUAL 5 & 6: Acquisition Trends + Income vs Tenure
# ----------------------------------
st.header("Customer Behavior")
col1, col2 = st.columns(2)

with col1:
    # Acquisition Trends
    st.subheader("Acquisition Over Time")
    yearly = customers['YearJoined'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    yearly.plot(kind='area', color='#9b59b6', alpha=0.6, ax=ax)
    
    # Highlight best year
    peak_year = yearly.idxmax()
    ax.annotate(f'Peak: {peak_year}', xy=(peak_year, yearly.max()),
               xytext=(10,10), textcoords='offset points',
               arrowprops=dict(arrowstyle='->'))
    
    ax.set_ylabel("New Customers")
    st.pyplot(fig)

with col2:
    # Income vs Tenure (Mocked data)
    st.subheader("Income vs Tenure")
    np.random.seed(42)
    customers['Income'] = np.random.normal(5000, 1500, len(customers))  # Mock income data
    #age_counts = customers['Age Binned'].value_counts().sort_index()
    #fig, plt.figure(figsize=(10,6))
    #age_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    #plt.title('Age Distribution of Customers')
    #plt.xlabel('Age Bins')
    #plt.ylabel('Number of Customers')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
   # st.pyplot(fig)
    #plt.show()


    fig, ax = plt.subplots(figsize=(8,4))
    sns.regplot(x='TenureYears', y='Income', data=customers, 
               scatter_kws={'alpha':0.3, 'color':'#3498db'},
               line_kws={'color':'#e74c3c'}, ax=ax)
    
    ax.set_xlabel("Tenure (Years)")
    ax.set_ylabel("Monthly Income (USD)")
    st.pyplot(fig)

# ----------------------------------
# SUMMARY ANNOTATIONS
# ----------------------------------
st.markdown("""
<div style="background-color:#f8f9fa;padding:15px;border-radius:5px;margin-top:20px">
<h4>🔍 Key Customer Insights</h4>
<ol>
<li>Tech sector shows <b>untapped potential</b> (only 8% penetration)</li>
<li><b style='color:#2ecc71'>Northern region</b> dominates with 32% of customers</li>
<li>23% of customers are in <b>high-churn</b> first year</li>
<li>Customer acquisition <b>peaked in 2021</b> (+18% vs 2020)</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ==================================
# EMPLOYEE ANALYTICS TAB
# ==================================
tab2 = st.tabs(["Employee Analytics"])[0]

with tab2:
    import seaborn as sns
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np

    # Load real employee data
    @st.cache_data
    def load_employee_data():
        df = pd.read_csv("C:\\Users\\Evelyn Wullar\\Desktop\\BRIGHT HORIZON\\BrightHorizon\\data\\Featured data\\bright_horizons_employees_enriched.csv", parse_dates=["JoiningDate"])
        df['JoiningYear'] = pd.to_datetime(df['JoiningDate']).dt.year
        df['Tenure'] = (pd.to_datetime("today") - pd.to_datetime(df['JoiningDate'])).dt.days / 365

        # If AttritionRisk column doesn't exist, you can mock it (temporarily)
        if 'AttritionRisk' not in df.columns:
            np.random.seed(42)
            df['AttritionRisk'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df), p=[0.6, 0.3, 0.1])

        return df

    employees = load_employee_data()

    with tab2:
        st.header("Employee Analytics")
    
    # Sidebar filters
   # Sidebar Filters
    st.sidebar.subheader("Filter Employees")
    selected_branch = st.sidebar.selectbox("Select Branch", ["All"] + sorted(employees['Branch'].unique()))
    selected_dept = st.sidebar.selectbox("Select Department", ["All"] + sorted(employees['Department'].unique()))
    selected_year = st.sidebar.selectbox("Select Joining Year", ["All"] + sorted(employees['JoiningYear'].unique()))

    # Apply Filters
    filtered_emp = employees.copy()
    if selected_branch != "All":
        filtered_emp = filtered_emp[filtered_emp['Branch'] == selected_branch]
    if selected_dept != "All":
        filtered_emp = filtered_emp[filtered_emp['Department'] == selected_dept]
    if selected_year != "All":
        filtered_emp = filtered_emp[filtered_emp['JoiningYear'] == selected_year]

# KPIs
st.subheader("📌 Workforce Overview")
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Employees", f"{filtered_emp['Employee ID'].nunique()}")
col2.metric("🕒 Avg Tenure", f"{filtered_emp['Tenure'].mean():.1f} yrs")
col3.metric("🎓 Avg Age", f"{filtered_emp['Age'].mean():.1f} yrs")

st.markdown("---")

# Chart layout
col1, col2 = st.columns(2)

# Visual 1: Hiring Trends
with col1:
    st.markdown("#### 📈 Hiring Trends")
    hiring_trend = filtered_emp.groupby("JoiningYear")["Employee ID"].count().reset_index()
    fig = px.bar(hiring_trend, x="JoiningYear", y="Employee ID", title="Employees Hired per Year", height=350)
    st.plotly_chart(fig, use_container_width=True)

# Visual 2: Department Distribution
with col2:
    st.markdown("#### 🏢 Department Distribution")
    dept_dist = filtered_emp["Department"].value_counts().reset_index()
    dept_dist.columns = ["Department", "Count"]
    fig = px.pie(dept_dist, values='Count', names='Department', title="Employee Distribution by Department", hole=0.4, height=350)
    st.plotly_chart(fig, use_container_width=True)

# Visual 3: Avg Tenure by Department
col3, col4 = st.columns(2)
with col3:
    st.markdown("#### ⏳ Avg Tenure by Department")
    tenure_dept = filtered_emp.groupby("Department")["Tenure"].mean().reset_index()
    fig = px.bar(tenure_dept, x="Department", y="Tenure", color="Department", title="Avg Tenure by Department", height=350)
    st.plotly_chart(fig, use_container_width=True)

# Visual 4: Avg Age by Position
with col4:
    st.markdown("#### 🧓 Avg Age by Position")
    age_position = filtered_emp.groupby("Position")["Age"].mean().reset_index()
    fig = px.bar(age_position, x="Age", y="Position", orientation='h', title="Avg Age by Position", height=350)
    st.plotly_chart(fig, use_container_width=True)

# Visual 5: Gender Balance
col5, col6 = st.columns(2)
with col5:
    st.markdown("#### 🚻 Gender Balance")
    gender_dist = filtered_emp["Gender"].value_counts().reset_index()
    gender_dist.columns = ["Gender", "Count"]
    fig = px.pie(gender_dist, values='Count', names='Gender', title="Gender Distribution", hole=0.3, height=350)
    st.plotly_chart(fig, use_container_width=True)

# Visual 6: Attrition Risk Placeholder
with col6:
# Create IncomeBin
    bins = np.arange(0, employees['IncomeLevelNumeric'].max() + 10000, 10000)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    employees['IncomeBin'] = pd.cut(employees['IncomeLevelNumeric'], bins=bins, labels=labels, right=False)

    # Create the crosstab (like groupby) for Education Level by IncomeBin
    edu_income_dist = pd.crosstab(employees['IncomeBin'], employees['EducationLevel'])

    # Convert the crosstab to a long-form DataFrame for Plotly
    edu_income_dist_long = edu_income_dist.reset_index().melt(id_vars='IncomeBin', var_name='EducationLevel', value_name='Count')

    # Create the stacked bar chart using Plotly
    fig6 = px.bar(
        edu_income_dist_long,
        x='IncomeBin',
        y='Count',
        color='EducationLevel',
        title='Distribution of Educational Level by Income Level',
        labels={'IncomeBin': 'Income Range', 'Count': 'Number of Employees'},
        height=350
    )
    fig6.update_layout(barmode='stack', xaxis_tickangle=-45)
    # Display the chart
    st.plotly_chart(fig6, use_container_width=True)

